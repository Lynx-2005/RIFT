//! Discovery for model servers running on this machine.
//!
//! This mirrors OpenCode's local provider plugins: each runtime keeps its
//! native metadata endpoint, while completions still go through the ordinary
//! OpenAI-compatible `/v1` API. Ollama answers `/api/tags` and `/api/show`
//! (completion capability, tool support, context length), LM Studio answers
//! `/api/v1/models`, and vLLM is checked with `/health` before `/v1/models`.
//!
//! Discovery is best-effort and never blocks the prompt: a failed probe keeps
//! the last successful inventory, and a successful empty list clears it so a
//! stopped server stops offering stale models.

use crate::provider::{Client, ThinkingSupport};
use crate::store::Store;
use anyhow::{Context, Result};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

/// How often the background thread repeats discovery. OpenCode uses the same
/// interval for its local providers.
pub const REFRESH_SECS: u64 = 30;

/// One discovery round. The three runtimes are probed in parallel so a
/// stopped server costs one timeout, not one per runtime.
pub fn discover_all(store: &Store, client: &Client) {
    // A file entry with an explicit model list is authoritative. Drop any
    // inventory discovered earlier so its hand-written list is the whole menu.
    for name in crate::config::LOCAL_PROVIDERS {
        if crate::provider::file_provider(name).is_some_and(|f| !f.models.is_empty()) {
            store.clear_models(name);
        }
    }
    let targets: Vec<Target> = crate::config::LOCAL_PROVIDERS
        .iter()
        .filter_map(|name| target(name, store))
        .collect();
    if targets.is_empty() {
        return;
    }
    let results: Vec<(&'static str, Option<Discovered>)> = std::thread::scope(|scope| {
        let handles: Vec<_> = targets
            .into_iter()
            .map(|target| {
                scope.spawn(move || {
                    let name = target.name;
                    (name, discover_one(&target, client))
                })
            })
            .collect();
        handles
            .into_iter()
            .filter_map(|handle| handle.join().ok())
            .collect()
    });
    for (name, result) in results {
        let Some(found) = result else { continue };
        if found.ids.is_empty() {
            store.clear_models(name);
            store.clear_contexts(name);
            store.save_provider_thinking(name, &[]);
        } else {
            store.save_models(name, &found.ids);
            if found.contexts.is_empty() {
                store.clear_contexts(name);
            } else {
                store.save_contexts(name, &found.contexts);
            }
            store.save_provider_thinking(name, &found.thinking);
        }
    }
}

struct Target {
    name: &'static str,
    root: String,
    key: String,
}

#[derive(Default)]
struct Discovered {
    ids: Vec<String>,
    contexts: Vec<(String, u64)>,
    /// Reasoning support per model, when the runtime described it.
    thinking: Vec<(String, ThinkingSupport)>,
}

/// A local runtime is probed only when it has an endpoint and no explicit
/// model list. A `providers.json` entry that names its models is the source of
/// truth; overriding it every 30 seconds would be surprising.
fn target(name: &'static str, store: &Store) -> Option<Target> {
    if crate::config::Provider::local(name).is_none() {
        return None;
    }
    if crate::provider::file_provider(name).is_some_and(|f| !f.models.is_empty()) {
        return None;
    }
    let state = crate::provider::state(name, store)?;
    let root = root_of(&state.base_url, name)?;
    Some(Target {
        name,
        root,
        key: state.api_key,
    })
}

/// Strip the completion API suffix from a configured base URL. The native
/// discovery endpoints sit beside `/v1`, not under it: a proxy path such as
/// `http://host/proxy/v1` keeps its `/proxy` prefix.
fn root_of(base_url: &str, name: &str) -> Option<String> {
    let mut url = reqwest::Url::parse(base_url).ok()?;
    let path = url.path().trim_end_matches('/').to_string();
    let stripped = match name {
        // Ollama accepts /v1 (OpenAI) or /api (native) as the base.
        "ollama" => path
            .strip_suffix("/v1")
            .or_else(|| path.strip_suffix("/api")),
        // LM Studio native metadata lives at /api/v1/models.
        "lmstudio" => path
            .strip_suffix("/api/v1")
            .or_else(|| path.strip_suffix("/v1")),
        _ => path.strip_suffix("/v1"),
    }
    .unwrap_or(&path);
    url.set_path(if stripped.is_empty() { "/" } else { stripped });
    url.set_query(None);
    url.set_fragment(None);
    Some(url.to_string().trim_end_matches('/').to_string())
}

fn discover_one(target: &Target, client: &Client) -> Option<Discovered> {
    match target.name {
        "ollama" => discover_ollama(target, client),
        "lmstudio" => discover_lmstudio(target, client),
        "vllm" => discover_vllm(target, client),
        _ => None,
    }
}

fn discover_ollama(target: &Target, client: &Client) -> Option<Discovered> {
    let tags = get_json(client, &format!("{}/api/tags", target.root), &target.key).ok()?;
    let summaries = ollama_summaries(&tags)?;
    let mut found = Discovered::default();
    // OpenCode limits /api/show to four concurrent requests. Model metadata is
    // cached by digest, so a refresh only asks about models that changed.
    for chunk in summaries.chunks(4) {
        let shown: Vec<(String, Option<ShowInfo>)> = std::thread::scope(|scope| {
            let handles: Vec<_> = chunk
                .iter()
                .map(|(model, digest)| {
                    scope.spawn(move || {
                        (
                            model.clone(),
                            show_info(client, &target.root, &target.key, model, digest),
                        )
                    })
                })
                .collect();
            handles
                .into_iter()
                .filter_map(|handle| handle.join().ok())
                .collect()
        });
        for (id, info) in shown {
            let Some(info) = info else { continue };
            if !info.completion {
                continue;
            }
            if let Some(context) = info.context {
                found.contexts.push((id.clone(), context));
            }
            found.thinking.push((id.clone(), info.thinking.clone()));
            found.ids.push(id);
        }
    }
    found.ids.sort();
    found.ids.dedup();
    found
        .contexts
        .retain(|(id, _)| found.ids.binary_search(id).is_ok());
    found
        .thinking
        .retain(|(id, _)| found.ids.binary_search(id).is_ok());
    Some(found)
}

fn discover_lmstudio(target: &Target, client: &Client) -> Option<Discovered> {
    let body = get_json(
        client,
        &format!("{}/api/v1/models", target.root),
        &target.key,
    )
    .ok()?;
    lmstudio_models(&body)
}

fn discover_vllm(target: &Target, client: &Client) -> Option<Discovered> {
    // vLLM may accept the port before a model is ready. /health is the
    // readiness gate OpenCode uses; without it, an empty /v1/models would look
    // like a real empty catalog and hide the last good inventory.
    let mut health = client
        .http
        .get(format!("{}/health", target.root))
        .timeout(Duration::from_secs(1));
    if !target.key.is_empty() {
        health = health.bearer_auth(&target.key);
    }
    health.send().ok()?.error_for_status().ok()?;
    let body = get_json(client, &format!("{}/v1/models", target.root), &target.key).ok()?;
    vllm_models(&body)
}

#[derive(Clone)]
struct ShowInfo {
    completion: bool,
    context: Option<u64>,
    /// What this model says about reasoning. Always `known`: the show response
    /// either described thinking controls, listed the thinking capability, or
    /// had no thinking capability at all.
    thinking: crate::provider::ThinkingSupport,
    /// Whether the model accepts image (vision) inputs.
    vision: bool,
}

type ShowCache = Mutex<HashMap<String, (String, ShowInfo)>>;

fn show_cache() -> &'static ShowCache {
    static CACHE: OnceLock<ShowCache> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn show_info(
    client: &Client,
    root: &str,
    key: &str,
    model: &str,
    digest: &str,
) -> Option<ShowInfo> {
    let cache_key = format!("{root}\0{key}\0{model}");
    if let Ok(cache) = show_cache().lock() {
        if let Some((seen, info)) = cache.get(&cache_key) {
            if seen == digest {
                return Some(info.clone());
            }
        }
    }
    let body = post_json(
        client,
        &format!("{root}/api/show"),
        key,
        &json!({"model": model}),
    )
    .ok()?;
    let info = ollama_show_info(&body);
    if let Ok(mut cache) = show_cache().lock() {
        cache.insert(cache_key, (digest.to_string(), info.clone()));
    }
    Some(info)
}

fn get_json(client: &Client, url: &str, key: &str) -> Result<Value> {
    let mut request = client
        .http
        .get(url)
        .timeout(Duration::from_secs(1))
        .header("accept", "application/json");
    if !key.is_empty() {
        request = request.bearer_auth(key);
    }
    request
        .send()
        .with_context(|| format!("reaching {url}"))?
        .error_for_status()
        .with_context(|| format!("{url} rejected discovery"))?
        .json()
        .context("parsing discovery response")
}

fn post_json(client: &Client, url: &str, key: &str, body: &Value) -> Result<Value> {
    let mut request = client
        .http
        .post(url)
        .timeout(Duration::from_secs(1))
        .header("accept", "application/json")
        .json(body);
    if !key.is_empty() {
        request = request.bearer_auth(key);
    }
    request
        .send()
        .with_context(|| format!("reaching {url}"))?
        .error_for_status()
        .with_context(|| format!("{url} rejected discovery"))?
        .json()
        .context("parsing discovery response")
}

// --- parsing ---------------------------------------------------------------

/// Ollama `/api/tags`: `{models: [{model, digest, ...}]}`. The `model` field is
/// the tag used for requests, so it wins over the display `name`.
fn ollama_summaries(body: &Value) -> Option<Vec<(String, String)>> {
    let models = body.get("models")?.as_array()?;
    let mut out: Vec<(String, String)> = models
        .iter()
        .filter_map(|model| {
            let id = model.get("model").and_then(Value::as_str)?;
            if id.is_empty() {
                return None;
            }
            let digest = model
                .get("digest")
                .and_then(Value::as_str)
                .unwrap_or_default();
            Some((id.to_string(), digest.to_string()))
        })
        .collect();
    out.sort();
    out.dedup_by(|a, b| a.0 == b.0);
    Some(out)
}

/// Ollama `/api/show`: capabilities gate chat models, the architecture's
/// `<family>.context_length` entry supplies the context window, and the
/// `thinking` descriptor names the levels a thinking model accepts.
fn ollama_show_info(body: &Value) -> ShowInfo {
    let capability = |name: &str| {
        body.get("capabilities")
            .and_then(Value::as_array)
            .is_some_and(|capabilities| {
                capabilities
                    .iter()
                    .any(|capability| capability.as_str() == Some(name))
            })
    };
    let completion = capability("completion");
    let vision = capability("vision");
    let context = body
        .get("model_info")
        .and_then(Value::as_object)
        .and_then(|info| {
            info.iter().find_map(|(key, value)| {
                if key.ends_with(".context_length") {
                    value.as_u64().filter(|context| *context > 0)
                } else {
                    None
                }
            })
        });
    let mut thinking = if capability("thinking") {
        ollama_thinking(body).unwrap_or_else(legacy_thinking)
    } else {
        ThinkingSupport {
            known: true,
            reasoning: false,
            ..Default::default()
        }
    };
    thinking.vision = vision;
    ShowInfo {
        completion,
        context,
        thinking,
        vision,
    }
}

/// Ollama `/api/show` `thinking`: `{values: [true|false|"low"|...], default}`.
/// Named values map onto the canonical ladder; a plain boolean switch becomes
/// one "on" level, the same way every other toggle-capable model is shown.
/// `none` is the documented off switch on the OpenAI-compatible endpoint, so
/// it is offered whenever any level is. A descriptor whose default is not one
/// of its own values is not trusted.
fn ollama_thinking(body: &Value) -> Option<ThinkingSupport> {
    let descriptor = body.get("thinking")?;
    let values = descriptor.get("values")?.as_array()?;
    let default = descriptor.get("default")?;
    if values.is_empty() || !values.iter().any(|value| value == default) {
        return None;
    }
    let named: Vec<String> = values
        .iter()
        .filter_map(Value::as_str)
        .filter(|name| crate::config::THINKING_LEVELS.contains(name))
        .map(str::to_string)
        .collect();
    let on = values.iter().any(|value| value.as_bool() == Some(true));
    if named.is_empty() && !on {
        // The descriptor exists but offers no way to enable thinking.
        return Some(ThinkingSupport {
            known: true,
            reasoning: false,
            ..Default::default()
        });
    }
    let mut support = ThinkingSupport {
        known: true,
        reasoning: true,
        effort: if named.is_empty() {
            // A boolean switch has no graded levels: one on level, presented
            // like every other toggle-capable model.
            vec!["high".to_string()]
        } else {
            named
        },
        ..Default::default()
    };
    support.effort.push("none".to_string());
    Some(support)
}

/// A thinking-capable model whose `/api/show` predates the `thinking`
/// descriptor. The OpenAI-compatible endpoint still resolves the canonical
/// names (aliasing `minimal` and `xhigh`), so the whole ladder is expressible
/// and `none` disables.
fn legacy_thinking() -> ThinkingSupport {
    ThinkingSupport {
        known: true,
        reasoning: true,
        effort: crate::config::THINKING_LEVELS
            .iter()
            .skip(1)
            .map(|level| level.to_string())
            .chain(["none".to_string()])
            .collect(),
        ..Default::default()
    }
}

/// LM Studio `/api/v1/models`: only `type: "llm"` entries are chat models. A
/// loaded instance can shrink the usable window, so the smallest loaded
/// context wins; otherwise the model's maximum is the best available answer.
fn lmstudio_models(body: &Value) -> Option<Discovered> {
    let models = body.get("models")?.as_array()?;
    let mut parsed: Vec<(String, Option<u64>)> = models
        .iter()
        .filter_map(|model| {
            if model.get("type").and_then(Value::as_str) != Some("llm") {
                return None;
            }
            let id = model.get("key").and_then(Value::as_str)?;
            if id.is_empty() {
                return None;
            }
            let loaded: Vec<u64> = model
                .get("loaded_instances")
                .and_then(Value::as_array)
                .map(|instances| {
                    instances
                        .iter()
                        .filter_map(|instance| {
                            instance
                                .get("config")
                                .and_then(|config| config.get("context_length"))
                                .and_then(Value::as_u64)
                                .filter(|context| *context > 0)
                        })
                        .collect()
                })
                .unwrap_or_default();
            let context = if loaded.is_empty() {
                model
                    .get("max_context_length")
                    .and_then(Value::as_u64)
                    .filter(|context| *context > 0)
            } else {
                loaded.into_iter().min()
            };
            Some((id.to_string(), context))
        })
        .collect();
    parsed.sort_by(|a, b| a.0.cmp(&b.0));
    let mut found = Discovered::default();
    for (id, context) in parsed {
        if let Some(context) = context {
            found.contexts.push((id.clone(), context));
        }
        found.ids.push(id);
    }
    Some(found)
}

/// vLLM `/v1/models`: `owned_by` distinguishes the models this server hosts
/// from proxy entries. vLLM reports its context window as `max_model_len`.
fn vllm_models(body: &Value) -> Option<Discovered> {
    let models = body.get("data")?.as_array()?;
    let mut parsed: Vec<(String, Option<u64>)> = models
        .iter()
        .filter_map(|model| {
            if model.get("owned_by").and_then(Value::as_str) != Some("vllm") {
                return None;
            }
            let id = model.get("id").and_then(Value::as_str)?;
            if id.is_empty() {
                return None;
            }
            let context = model
                .get("max_model_len")
                .and_then(Value::as_u64)
                .filter(|context| *context > 0);
            Some((id.to_string(), context))
        })
        .collect();
    parsed.sort_by(|a, b| a.0.cmp(&b.0));
    let mut found = Discovered::default();
    for (id, context) in parsed {
        if let Some(context) = context {
            found.contexts.push((id.clone(), context));
        }
        found.ids.push(id);
    }
    Some(found)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    /// A minimal HTTP server for one discovery round: accept exactly
    /// `expected` requests, hand each to `handler`, then exit.
    fn serve(
        expected: usize,
        mut handler: impl FnMut(&str, &str, &str) -> String + Send + 'static,
    ) -> (String, std::thread::JoinHandle<()>) {
        let listener = TcpListener::bind(("127.0.0.1", 0)).expect("bind test server");
        let origin = format!("http://{}", listener.local_addr().unwrap());
        let handle = std::thread::spawn(move || {
            for _ in 0..expected {
                let Ok((mut stream, _)) = listener.accept() else {
                    return;
                };
                let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
                let mut raw = Vec::new();
                let mut buf = [0u8; 2048];
                loop {
                    let Ok(n) = stream.read(&mut buf) else { break };
                    if n == 0 {
                        break;
                    }
                    raw.extend_from_slice(&buf[..n]);
                    let Some(head_end) = raw.windows(4).position(|w| w == b"\r\n\r\n") else {
                        continue;
                    };
                    let head = String::from_utf8_lossy(&raw[..head_end]).to_string();
                    let length = head
                        .lines()
                        .find_map(|line| {
                            let (key, value) = line.split_once(':')?;
                            key.eq_ignore_ascii_case("content-length")
                                .then(|| value.trim().parse::<usize>().ok())
                                .flatten()
                        })
                        .unwrap_or(0);
                    if raw.len() < head_end + 4 + length {
                        continue;
                    }
                    let request_line = head.lines().next().unwrap_or_default();
                    let mut parts = request_line.split_whitespace();
                    let method = parts.next().unwrap_or_default();
                    let path = parts.next().unwrap_or_default();
                    let body = String::from_utf8_lossy(&raw[head_end + 4..head_end + 4 + length]);
                    let body = handler(method, path, &body);
                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    );
                    let _ = stream.write_all(response.as_bytes());
                    let _ = stream.flush();
                    break;
                }
            }
        });
        (origin, handle)
    }

    #[test]
    fn native_roots_strip_only_the_completion_suffix() {
        assert_eq!(
            root_of("http://localhost:11434/v1", "ollama").as_deref(),
            Some("http://localhost:11434")
        );
        assert_eq!(
            root_of("http://localhost:11434/api", "ollama").as_deref(),
            Some("http://localhost:11434")
        );
        assert_eq!(
            root_of("https://gpu.example/proxy/v1", "ollama").as_deref(),
            Some("https://gpu.example/proxy")
        );
        assert_eq!(
            root_of("http://localhost:1234/api/v1", "lmstudio").as_deref(),
            Some("http://localhost:1234")
        );
        assert_eq!(
            root_of("http://localhost:8000/v1", "vllm").as_deref(),
            Some("http://localhost:8000")
        );
    }

    #[test]
    fn ollama_keeps_only_completion_models_with_context() {
        let tags = json!({
            "models": [
                {"model": "gemma3:4b", "digest": "d1"},
                {"model": "nomic-embed", "digest": "d2"},
                {"model": "", "digest": "d3"}
            ]
        });
        assert_eq!(
            ollama_summaries(&tags),
            Some(vec![
                ("gemma3:4b".to_string(), "d1".to_string()),
                ("nomic-embed".to_string(), "d2".to_string())
            ])
        );
        assert!(ollama_summaries(&json!({})).is_none());

        let chat = json!({
            "capabilities": ["completion", "tools", "vision"],
            "model_info": {"gemma3.context_length": 131072}
        });
        let info = ollama_show_info(&chat);
        assert!(info.completion);
        assert_eq!(info.context, Some(131072));
        assert!(
            info.thinking.known && !info.thinking.reasoning,
            "a model without the thinking capability is explicitly non-reasoning"
        );

        let embedding = json!({
            "capabilities": ["embedding"],
            "model_info": {"nomic.context_length": 8192}
        });
        assert!(!ollama_show_info(&embedding).completion);
        assert!(!ollama_show_info(&json!({})).completion);
    }

    #[test]
    fn ollama_thinking_reads_the_models_own_levels() {
        // The server names the levels it accepts: only those become /thinking
        // choices; `none` rides along as the documented off switch.
        let named = json!({
            "capabilities": ["completion", "thinking"],
            "thinking": {
                "values": ["low", "medium", "high", "max"],
                "default": "medium"
            }
        });
        let info = ollama_show_info(&named);
        assert!(info.thinking.reasoning);
        assert_eq!(
            info.thinking.effort,
            vec!["low", "medium", "high", "max", "none"]
        );

        // A plain boolean switch has one on level, like any toggle model.
        let toggle = json!({
            "capabilities": ["completion", "thinking"],
            "thinking": {"values": [true, false], "default": true}
        });
        let info = ollama_show_info(&toggle);
        assert!(info.thinking.reasoning);
        assert_eq!(info.thinking.effort, vec!["high", "none"]);

        // A descriptor that cannot be trusted falls back to the capability,
        // whose endpoint still resolves the canonical aliases.
        let broken = json!({
            "capabilities": ["completion", "thinking"],
            "thinking": {"values": ["low", "high"], "default": "medium"}
        });
        let info = ollama_show_info(&broken);
        assert!(info.thinking.reasoning);
        assert!(
            info.thinking.effort.len() > 3,
            "the legacy ladder keeps thinking usable: {:?}",
            info.thinking.effort
        );

        // A thinking capability with no descriptor (older server) does the
        // same rather than hiding the model's ability.
        let legacy = json!({"capabilities": ["completion", "thinking"]});
        let info = ollama_show_info(&legacy);
        assert!(info.thinking.reasoning);
        assert!(info.thinking.effort.contains(&"high".to_string()));

        // A descriptor offering no way to enable thinking is not reasoning.
        let off_only = json!({
            "capabilities": ["completion", "thinking"],
            "thinking": {"values": [false], "default": false}
        });
        let info = ollama_show_info(&off_only);
        assert!(info.thinking.known && !info.thinking.reasoning);
    }

    #[test]
    fn lmstudio_uses_the_smallest_loaded_context_and_skips_embeddings() {
        let body = json!({
            "models": [
                {
                    "type": "llm",
                    "key": "qwen3-coder",
                    "loaded_instances": [
                        {"config": {"context_length": 32768}},
                        {"config": {"context_length": 16384}}
                    ],
                    "max_context_length": 262144
                },
                {
                    "type": "llm",
                    "key": "deepseek-r1",
                    "loaded_instances": [],
                    "max_context_length": 131072
                },
                {
                    "type": "embedding",
                    "key": "nomic-embed",
                    "max_context_length": 2048
                }
            ]
        });
        let found = lmstudio_models(&body).unwrap();
        assert_eq!(found.ids, vec!["deepseek-r1", "qwen3-coder"]);
        assert_eq!(
            found.contexts,
            vec![
                ("deepseek-r1".to_string(), 131072),
                ("qwen3-coder".to_string(), 16384)
            ]
        );
        assert!(lmstudio_models(&json!({})).is_none());
    }

    #[test]
    fn discovery_asks_ollama_for_native_model_metadata() {
        let (origin, server) = serve(3, |method, path, body| match path {
            "/api/tags" => {
                assert_eq!(method, "GET");
                json!({
                    "models": [
                        {"model": "chat:7b", "digest": "d1"},
                        {"model": "embed", "digest": "d2"}
                    ]
                })
                .to_string()
            }
            "/api/show" => {
                assert_eq!(method, "POST");
                let model = serde_json::from_str::<Value>(body)
                    .unwrap()
                    .get("model")
                    .and_then(Value::as_str)
                    .unwrap()
                    .to_string();
                if model == "chat:7b" {
                    json!({
                        "capabilities": ["completion", "tools", "thinking"],
                        "thinking": {"values": ["low", "medium", "high"], "default": "medium"},
                        "model_info": {"llama.context_length": 32768}
                    })
                    .to_string()
                } else {
                    json!({
                        "capabilities": ["embedding"],
                        "model_info": {"bert.context_length": 8192}
                    })
                    .to_string()
                }
            }
            other => panic!("unexpected discovery request: {other}"),
        });
        let client = Client::new().unwrap();
        let target = Target {
            name: "ollama",
            root: origin,
            key: String::new(),
        };
        let found = discover_ollama(&target, &client).expect("ollama discovery should succeed");
        server.join().unwrap();
        assert_eq!(found.ids, vec!["chat:7b"]);
        assert_eq!(found.contexts, vec![("chat:7b".to_string(), 32768)]);
        assert_eq!(found.thinking.len(), 1);
        let (model, support) = &found.thinking[0];
        assert_eq!(model, "chat:7b");
        assert!(support.reasoning);
        assert_eq!(support.effort, vec!["low", "medium", "high", "none"]);
    }

    #[test]
    fn vllm_skips_models_it_does_not_own() {
        let body = json!({
            "data": [
                {"id": "Qwen/Qwen3-Coder", "owned_by": "vllm", "max_model_len": 65536},
                {"id": "foreign", "owned_by": "other", "max_model_len": 4096},
                {"id": "no-limit", "owned_by": "vllm", "max_model_len": 0}
            ]
        });
        let found = vllm_models(&body).unwrap();
        assert_eq!(found.ids, vec!["Qwen/Qwen3-Coder", "no-limit"]);
        assert_eq!(
            found.contexts,
            vec![("Qwen/Qwen3-Coder".to_string(), 65536)]
        );
        assert!(vllm_models(&json!({})).is_none());
    }
}
