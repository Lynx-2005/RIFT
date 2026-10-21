//! Provider adapters. Everything is normalized to `Msg` / `Turn`, so the agent
//! loop never cares which vendor is on the other end.
use crate::config::{Config, Provider, Wire};
use anyhow::{anyhow, bail, Context, Result};

/// The community model catalog (models.dev) that opencode and pi derive their
/// context windows from: provider-keyless, one JSON blob, ~9000 models.
const MODELS_DEV_URL: &str = "https://models.dev/api.json";
use serde_json::{json, Value};
use crate::stats::Usage;
use std::thread;
use std::time::{Duration, Instant};

pub const SYSTEM: &str = include_str!("system_prompt.md");

/// The instructions plus the one fact the model cannot infer: where commands
/// run. Built per request so a `-C` change is never stale.
///
/// `context` is what the session adds: today the skill catalog, which is a
/// list of names and one-line descriptions, never a skill's body. A body
/// arrives only when the model asks for it by name.
pub fn system(cfg: &Config, context: &str) -> String {
    let base = format!(
        "{SYSTEM}\n\nYou are on {os}, and the `shell` tool runs every command through {shell}.\nYour working directory is {cwd}.\nEvery command runs there, and relative paths in your commands resolve there. A `cd` inside one command does not carry to the next: each command starts in the working directory again. Stay in that directory; leave it only if the user asks.",
        os = std::env::consts::OS,
        shell = crate::shell::SHELL,
        cwd = cfg.cwd.display()
    );
    if context.trim().is_empty() {
        base
    } else {
        format!("{base}\n\n{}", context.trim_end())
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Chunk<'a> {
    Text(&'a str),
    Think(&'a str),
}

/// One tool the model may call, as the request must describe it: a name, a
/// description, and a JSON-Schema for the arguments. The harness owns `shell`
/// and `skill`; everything else arrives from an MCP server, so this list is
/// assembled per session instead of being a constant.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub schema: serde_json::Value,
}

/// A call the model made: which tool, and the arguments it passed. Arguments
/// are kept as JSON because that is what every provider speaks and what an MCP
/// server is handed back verbatim.
#[derive(Debug, Clone)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub input: serde_json::Value,
}

impl ToolCall {
    /// A shell call, for tests that need one.
    #[cfg(test)]
    pub fn shell(id: impl Into<String>, command: impl Into<String>) -> Self {
        let command = command.into();
        ToolCall {
            id: id.into(),
            name: "shell".to_string(),
            input: serde_json::json!({"command": command}),
        }
    }

    /// The command, when this is a shell call.
    pub fn command(&self) -> Option<&str> {
        if self.name != "shell" {
            return None;
        }
        self.input.get("command").and_then(|c| c.as_str())
    }

    /// What the tool echo prints. A shell command reads as itself; anything
    /// else reads as its name plus a one-line rendering of the arguments, so a
    /// call to a server tool is still legible on one row.
    pub fn label(&self) -> String {
        if let Some(c) = self.command() {
            return c.to_string();
        }
        let args = match &self.input {
            serde_json::Value::Object(o) if !o.is_empty() => {
                let pairs: Vec<String> = o
                    .iter()
                    .map(|(k, v)| {
                        let v = match v {
                            serde_json::Value::String(s) => s.clone(),
                            other => other.to_string(),
                        };
                        format!("{k}={v}")
                    })
                    .collect();
                format!(" {}", pairs.join(" "))
            }
            _ => String::new(),
        };
        format!("{}{}", self.name, args)
    }
}

#[derive(Debug, Clone)]
pub enum Msg {
    User(String),
    Assistant { text: String, calls: Vec<ToolCall> },
    ToolResult { id: String, output: String },
}

/// Transcript encoding for the sessions table. Plain JSON, no schema version:
/// old rows still parse because every field has a default.
pub fn msg_json(m: &Msg) -> serde_json::Value {
    match m {
        Msg::User(t) => serde_json::json!({"role": "user", "text": t}),
        Msg::Assistant { text, calls } => serde_json::json!({
            "role": "assistant", "text": text,
            "calls": calls.iter().map(|c| serde_json::json!({"id": c.id, "name": c.name, "input": c.input})).collect::<Vec<_>>(),
        }),
        Msg::ToolResult { id, output } => serde_json::json!({"role": "tool", "id": id, "output": output}),
    }
}

pub fn msg_from_json(v: &Value) -> Option<Msg> {
    match v["role"].as_str()? {
        "user" => Some(Msg::User(v["text"].as_str().unwrap_or_default().to_string())),
        "assistant" => Some(Msg::Assistant {
            text: v["text"].as_str().unwrap_or_default().to_string(),
            calls: v["calls"]
                .as_array()
                .cloned()
                .unwrap_or_default()
                .iter()
                .filter_map(|c| {
                    // Rows written before tools had names stored a bare
                    // `command`: those are shell calls, and reading them back
                    // is what keeps an old session resumable.
                    let name = c["name"].as_str().unwrap_or("shell").to_string();
                    let input = if c["input"].is_object() {
                        c["input"].clone()
                    } else {
                        serde_json::json!({"command": c["command"].as_str().unwrap_or_default()})
                    };
                    if name.is_empty() {
                        return None;
                    }
                    Some(ToolCall { id: c["id"].as_str().unwrap_or_default().to_string(), name, input })
                })
                .collect(),
        }),
        "tool" => Some(Msg::ToolResult {
            id: v["id"].as_str().unwrap_or_default().to_string(),
            output: v["output"].as_str().unwrap_or_default().to_string(),
        }),
        _ => None,
    }
}

#[derive(Debug, Default)]
pub struct Turn {
    pub text: String,
    pub thinking: String,
    pub calls: Vec<ToolCall>,
    /// What the provider billed for this call. Zeroed by providers that do not
    /// report it, so `/stats` shows what is known instead of inventing a figure.
    pub usage: Usage,
}

#[derive(Clone)]
pub struct Client {
    pub(crate) http: reqwest::blocking::Client,
}

/// The provider's model ids plus the context lengths it reported for
/// them (OpenRouter annotates every model; OpenAI annotates none).
pub struct ModelList {
    pub ids: Vec<String>,
    pub contexts: Vec<(String, u64)>,
}

impl Client {
    pub fn new() -> Result<Self> {
        let http = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(300))
            .build()?;
        Ok(Self { http })
    }

    /// One model call, streamed: `on_text` fires per text delta so the UI
    /// prints tokens as they arrive. Takes the config per request so slash
    /// commands take effect immediately.
    pub fn complete_streaming(
        &self,
        cfg: &Config,
        history: &[Msg],
        tools: &[ToolSpec],
        context: &str,
        mut on_chunk: impl FnMut(Chunk<'_>),
    ) -> Result<Turn> {
        if cfg.api_key.is_empty() && cfg.provider.requires_key() {
            bail!(
                "no API key for {}: set {} or run /provider",
                cfg.provider.name(),
                cfg.provider.key_env()
            );
        }
        let (url, body, headers) = match cfg.provider.wire() {
            Wire::Anthropic => anthropic_request(cfg, history, tools, context),
            Wire::OpenAi => openai_request(cfg, history, tools, context),
        };
        let started = Instant::now();
        let mut first: Option<Instant> = None;
        let mut timed = |c: Chunk<'_>| {
            if first.is_none() {
                first = Some(Instant::now());
            }
            on_chunk(c);
        };
        let mut resp = send_with_retry(&self.http, &url, &body, &headers)?;
        let turn = match cfg.provider.wire() {
            Wire::Anthropic => stream_anthropic(&mut resp, &mut timed),
            Wire::OpenAi => stream_openai(&mut resp, &mut timed),
        };
        // Time to first token is where the wait is felt; the rest is generation
        // speed. A failed call records nothing: `/stats` is billing, not errors.
        if let Ok(t) = &turn {
            let ttft = first.map_or(Duration::ZERO, |f| f.saturating_duration_since(started));
            let gen = first.map_or(Duration::ZERO, |f| f.elapsed());
            crate::stats::record(t.usage, ttft, gen);
        }
        turn
    }

    /// `GET {base}/models` for OpenAI-compatible providers. Best-effort:
    /// callers keep serving the cached list on error, but the error itself is
    /// surfaced so a silent empty menu is never mistaken for the truth.
    pub fn fetch_models(&self, base_url: &str, api_key: &str) -> Result<ModelList> {
        let url = format!("{base_url}/models");
        let mut req = self.http.get(&url).timeout(Duration::from_secs(8));
        if !api_key.is_empty() {
            req = req.bearer_auth(api_key);
        }
        let v: Value = req
            .send()
            .with_context(|| format!("reaching {url}"))?
            .error_for_status()
            .with_context(|| format!("{base_url} rejected the model-list request"))?
            .json()
            .context("parsing the model list")?;
        let mut ids = Vec::new();
        let mut contexts: Vec<(String, u64)> = Vec::new();
        if let Some(data) = v.get("data").and_then(|d| d.as_array()) {
            for item in data {
                let Some(id) = item.get("id").and_then(|i| i.as_str()) else { continue };
                if id.is_empty() {
                    continue;
                }
                ids.push(id.to_string());
                // Context length when the provider tells us (OpenRouter
                // does): `context_length` directly, or under `top_provider`.
                let ctx = item
                    .get("context_length")
                    .and_then(Value::as_u64)
                    .or_else(|| {
                        item.get("top_provider")
                            .and_then(|t| t.get("context_length"))
                            .and_then(Value::as_u64)
                    })
                    .filter(|c| *c > 0);
                if let Some(ctx) = ctx {
                    contexts.push((id.to_string(), ctx));
                }
            }
        }
        if ids.is_empty() {
            bail!("empty model list");
        }
        ids.sort();
        ids.dedup();
        Ok(ModelList { ids, contexts })
    }

    /// Context windows for every model known to models.dev, keyed by bare
    /// model id (the same source opencode uses; no API key). Best-effort:
    /// callers proceed without it on any error.
    pub fn fetch_model_contexts(&self) -> Vec<(String, u64)> {
        self.fetch_model_contexts_from(MODELS_DEV_URL)
    }

    /// The same, against an explicit URL (tests point it at a local server).
    pub fn fetch_model_contexts_from(&self, url: &str) -> Vec<(String, u64)> {
        let Ok(body) = self
            .http
            .get(url)
            .timeout(Duration::from_secs(15))
            .send()
        else {
            return Vec::new();
        };
        let Ok(v) = body.json::<Value>() else { return Vec::new() };
        let Some(providers) = v.as_object() else { return Vec::new() };
        let mut out = Vec::new();
        for p in providers.values() {
            let Some(models) = p.get("models").and_then(Value::as_object) else { continue };
            for (id, m) in models {
                let Some(ctx) = m.get("limit").and_then(|l| l.get("context")).and_then(Value::as_u64)
                else {
                    continue;
                };
                if ctx > 0 {
                    // The same id is listed by several providers with
                    // different windows (200k anthropic, 1M on a 1m tier,
                    // 198k bedrock). Keep the smallest: a window too small
                    // compacts one request early, one too large loses the
                    // request - the reason the heuristic below also errs low.
                    match out.iter_mut().find(|(seen, _)| seen == id) {
                        Some(entry) if ctx < entry.1 => entry.1 = ctx,
                        Some(_) => {}
                        None => out.push((id.clone(), ctx)),
                    }
                }
            }
        }
        out.sort();
        out
    }
}


/// Pi-style effort ladder (`off|minimal|low|medium|high|xhigh|max`).
/// Anthropic's API speaks token budgets for thinking, so each level maps to one
/// here. The OpenAI-compatible path takes level names instead, and only the
/// four it documents: see `api_effort`.
pub fn thinking_budget(level: &str) -> Option<u32> {
    match level {
        "minimal" => Some(1024),
        "low" => Some(4096),
        "medium" => Some(10240),
        "high" => Some(20480),
        "xhigh" => Some(24576),
        "max" => Some(31999),
        _ => None, // "off" and unknown: no thinking block
    }
}

/// The effort levels an OpenAI-compatible API accepts. The ladder above is
/// wider than this API's vocabulary, and an unsupported value is a rejected
/// request rather than something a server tolerates, so the top of the ladder
/// clamps to `high` and says so (`effort_note`).
pub fn api_effort(level: &str) -> Option<&'static str> {
    match level {
        "minimal" => Some("minimal"),
        "low" => Some("low"),
        "medium" => Some("medium"),
        "high" | "xhigh" | "max" => Some("high"),
        _ => None, // "off" and anything unrecognized: no effort field at all
    }
}

/// How a clamped level is reported, so the request never quietly changes the
/// setting behind the user's back. `None` when nothing is being changed.
pub fn effort_note(level: &str) -> Option<String> {
    let sent = api_effort(level)?;
    (sent != level).then(|| format!("this API takes minimal, low, medium, high: sending {sent}"))
}

fn send_with_retry(
    http: &reqwest::blocking::Client,
    url: &str,
    body: &Value,
    headers: &[(&str, String)],
) -> Result<reqwest::blocking::Response> {
    let mut attempt = 0u32;
    loop {
        attempt += 1;
        let mut req = http.post(url).json(body);
        for (k, v) in headers {
            req = req.header(*k, v);
        }
        match req.send() {
            Ok(resp) => {
                let status = resp.status();
                if status.is_success() {
                    return Ok(resp);
                }
                let retryable = status.as_u16() == 429 || status.is_server_error();
                if retryable && attempt < 5 {
                    backoff(attempt);
                    continue;
                }
                let text = resp.text().unwrap_or_default();
                bail!("HTTP {status}: {}", truncate_err(&text));
            }
            Err(e) if attempt < 5 && (e.is_timeout() || e.is_connect()) => backoff(attempt),
            Err(e) => return Err(anyhow!(e).context("request failed")),
        }
    }
}

/// Incremental SSE decoder: yields `data:` payloads, one per blank-line
/// dispatch. Survives events split across TCP chunks.
#[derive(Default)]
struct SseDecoder {
    buf: String,
    pending: String,
}

impl SseDecoder {
    fn push(&mut self, chunk: &[u8]) -> Vec<String> {
        self.buf.push_str(&String::from_utf8_lossy(chunk));
        let mut out = Vec::new();
        while let Some(i) = self.buf.find('\n') {
            let line = self.buf[..i].trim_end_matches('\r').to_string();
            self.buf.drain(..=i);
            if line.is_empty() {
                if !self.pending.is_empty() {
                    out.push(std::mem::take(&mut self.pending));
                }
            } else if let Some(d) = line.strip_prefix("data:") {
                let d = d.trim_start();
                if !self.pending.is_empty() {
                    self.pending.push('\n');
                }
                self.pending.push_str(d);
            }
        }
        out
    }

    fn finish(&mut self) -> Option<String> {
        // Flush an unterminated trailing line (streams cut mid-flush).
        let rest = std::mem::take(&mut self.buf);
        let rest = rest.trim_end_matches(['\r', '\n']);
        if let Some(d) = rest.strip_prefix("data:") {
            if !self.pending.is_empty() {
                self.pending.push('\n');
            }
            self.pending.push_str(d.trim_start());
        }
        if self.pending.is_empty() {
            None
        } else {
            Some(std::mem::take(&mut self.pending))
        }
    }
}

#[derive(Default)]
struct AnthropicAsm {
    text: String,
    thinking: String,
    ids: Vec<String>,
    names: Vec<String>,
    json: Vec<String>,
    is_tool: Vec<bool>,
    is_think: Vec<bool>,
    usage: Usage,
}

impl AnthropicAsm {
    fn ensure(&mut self, idx: usize) {
        while self.ids.len() <= idx {
            self.ids.push(String::new());
            self.names.push(String::new());
            self.json.push(String::new());
            self.is_tool.push(false);
            self.is_think.push(false);
        }
    }

    fn event<F: FnMut(Chunk<'_>)>(&mut self, ev: &Value, on_chunk: &mut F) -> Result<()> {
        // Every event that carries usage replaces the running totals: Anthropic
        // sends input on `message_start` and the final output count on
        // `message_delta`, and the later figure is the authoritative one.
        if let Some(u) = usage_anthropic(&ev["usage"]) {
            self.usage = u;
        }
        match ev["type"].as_str() {
            Some("content_block_start") => {
                let idx = ev["index"].as_u64().unwrap_or(0) as usize;
                self.ensure(idx);
                match ev["content_block"]["type"].as_str() {
                    Some("tool_use") => {
                        self.is_tool[idx] = true;
                        self.ids[idx] = ev["content_block"]["id"].as_str().unwrap_or_default().to_string();
                        self.names[idx] = ev["content_block"]["name"].as_str().unwrap_or_default().to_string();
                    }
                    Some("thinking") => self.is_think[idx] = true,
                    _ => {}
                }
            }
            Some("content_block_delta") => {
                let idx = ev["index"].as_u64().unwrap_or(0) as usize;
                self.ensure(idx);
                match ev["delta"]["type"].as_str() {
                    Some("text_delta") => {
                        let t = ev["delta"]["text"].as_str().unwrap_or("");
                        if !t.is_empty() {
                            self.text.push_str(t);
                            on_chunk(Chunk::Text(t));
                        }
                    }
                    Some("thinking_delta") => {
                        let t = ev["delta"]["thinking"].as_str().unwrap_or("");
                        if !t.is_empty() {
                            self.thinking.push_str(t);
                            on_chunk(Chunk::Think(t));
                        }
                    }
                    Some("input_json_delta") => {
                        self.json[idx].push_str(ev["delta"]["partial_json"].as_str().unwrap_or(""));
                    }
                    _ => {}
                }
            }
            Some("error") => bail!("stream error: {}", ev["error"]["message"].as_str().unwrap_or("unknown")),
            _ => {}
        }
        Ok(())
    }

    fn finish(self) -> Result<Turn> {
        let mut calls = Vec::new();
        for i in 0..self.ids.len() {
            if self.is_tool[i] && !self.names[i].is_empty() {
                // A tool call whose JSON never parsed is kept with an empty
                // object: the model gets a result saying so, and the history
                // stays valid instead of losing a call the provider billed for.
                let args: Value = serde_json::from_str(&self.json[i]).unwrap_or(Value::Null);
                let input = if args.is_null() { serde_json::json!({}) } else { args };
                calls.push(ToolCall { id: self.ids[i].clone(), name: self.names[i].clone(), input });
            }
        }
        Ok(Turn { text: self.text, thinking: self.thinking, calls, usage: self.usage })
    }
}

#[derive(Default)]
struct OpenaiAsm {
    text: String,
    thinking: String,
    ids: Vec<String>,
    names: Vec<String>,
    args: Vec<String>,
    usage: Usage,
}

impl OpenaiAsm {
    fn ensure(&mut self, idx: usize) {
        while self.ids.len() <= idx {
            self.ids.push(String::new());
            self.names.push(String::new());
            self.args.push(String::new());
        }
    }

    fn event<F: FnMut(Chunk<'_>)>(&mut self, data: &Value, on_chunk: &mut F) -> Result<()> {
        if !data["error"].is_null() {
            bail!("stream error: {}", data["error"]["message"].as_str().unwrap_or("unknown"));
        }
        // With `include_usage`, the final chunk carries the totals and has an
        // empty `choices`, so it is read before the delta is looked for.
        if let Some(u) = usage_openai(&data["usage"]) {
            self.usage = u;
        }
        let Some(delta) = data["choices"].as_array().and_then(|c| c.first()).and_then(|c| c.get("delta")) else {
            return Ok(());
        };
        if let Some(t) = delta["content"].as_str() {
            if !t.is_empty() {
                self.text.push_str(t);
                on_chunk(Chunk::Text(t));
            }
        }
        // Reasoning models (deepseek-r1, grok, qwen…) stream thought here.
        for key in ["reasoning_content", "reasoning"] {
            if let Some(t) = delta[key].as_str() {
                if !t.is_empty() {
                    self.thinking.push_str(t);
                    on_chunk(Chunk::Think(t));
                }
            }
        }
        for tc in delta["tool_calls"].as_array().cloned().unwrap_or_default() {
            let idx = tc["index"].as_u64().unwrap_or(0) as usize;
            self.ensure(idx);
            if let Some(id) = tc["id"].as_str() {
                self.ids[idx] = id.to_string();
            }
            if let Some(name) = tc["function"]["name"].as_str() {
                self.names[idx] = name.to_string();
            }
            if let Some(a) = tc["function"]["arguments"].as_str() {
                self.args[idx].push_str(a);
            }
        }
        Ok(())
    }

    fn finish(self) -> Result<Turn> {
        let mut calls = Vec::new();
        for i in 0..self.ids.len() {
            if self.names[i].is_empty() {
                continue;
            }
            let args: Value = serde_json::from_str(&self.args[i]).unwrap_or(Value::Null);
            let input = if args.is_null() { serde_json::json!({}) } else { args };
            calls.push(ToolCall { id: self.ids[i].clone(), name: self.names[i].clone(), input });
        }
        Ok(Turn { text: self.text, thinking: self.thinking, calls, usage: self.usage })
    }
}

/// Drain one SSE stream; `handle` folds each `data:` payload into `asm`.
/// Providers that ignore `stream: true` answer with plain JSON: if zero SSE
/// events arrive, fall back to parsing the whole body (`parse`).
/// Returns the fallback Turn, or `None` when the stream was consumed into `asm`.
fn drain_stream<T, H, F>(
    resp: &mut reqwest::blocking::Response,
    asm: &mut T,
    mut handle: H,
    on_chunk: &mut F,
    parse: impl FnOnce(&Value) -> Result<Turn>,
) -> Result<Option<Turn>>
where
    H: FnMut(&mut T, &Value, &mut F) -> Result<()>,
    F: for<'x> FnMut(Chunk<'x>),
{
    use std::io::Read;
    let mut dec = SseDecoder::default();
    let mut raw = Vec::new();
    let mut tmp = [0u8; 8192];
    let mut events = 0u32;
    let mut payload = |payload: &str, asm: &mut T, on_chunk: &mut F, events: &mut u32| -> Result<()> {
        if payload == "[DONE]" {
            return Ok(());
        }
        let v: Value = serde_json::from_str(payload).context("parsing stream event")?;
        *events += 1;
        handle(asm, &v, on_chunk)
    };
    loop {
        let n = resp.read(&mut tmp).context("reading stream")?;
        if n == 0 {
            break;
        }
        raw.extend_from_slice(&tmp[..n]);
        for p in dec.push(&tmp[..n]) {
            payload(&p, asm, on_chunk, &mut events)?;
        }
    }
    if let Some(p) = dec.finish() {
        payload(&p, asm, on_chunk, &mut events)?;
    }
    if events == 0 {
        let v: Value = serde_json::from_slice(&raw).context("parsing response JSON")?;
        return parse(&v).map(Some);
    }
    Ok(None)
}

fn stream_anthropic(
    resp: &mut reqwest::blocking::Response,
    on_chunk: &mut impl for<'x> FnMut(Chunk<'x>),
) -> Result<Turn> {
    let mut asm = AnthropicAsm::default();
    if let Some(t) = drain_stream(resp, &mut asm, |a, v, c| a.event(v, c), on_chunk, parse_anthropic)? {
        return Ok(t);
    }
    asm.finish()
}

fn stream_openai(
    resp: &mut reqwest::blocking::Response,
    on_chunk: &mut impl for<'x> FnMut(Chunk<'x>),
) -> Result<Turn> {
    let mut asm = OpenaiAsm::default();
    if let Some(t) = drain_stream(resp, &mut asm, |a, v, c| a.event(v, c), on_chunk, parse_openai)? {
        return Ok(t);
    }
    asm.finish()
}

fn anthropic_request(
    cfg: &Config,
    history: &[Msg],
    tools: &[ToolSpec],
    context: &str,
) -> (String, Value, Vec<(&'static str, String)>) {
        let mut messages: Vec<Value> = Vec::new();
        for m in history {
            match m {
                Msg::User(t) => messages.push(json!({"role": "user", "content": t})),
                Msg::Assistant { text, calls } => {
                    let mut blocks = Vec::new();
                    if !text.is_empty() {
                        blocks.push(json!({"type": "text", "text": text}));
                    }
                    for c in calls {
                        blocks.push(json!({"type": "tool_use", "id": c.id, "name": c.name, "input": c.input}));
                    }
                    if blocks.is_empty() {
                        blocks.push(json!({"type": "text", "text": "(no output)"}));
                    }
                    messages.push(json!({"role": "assistant", "content": blocks}));
                }
                Msg::ToolResult { id, output } => {
                    let block = json!({"type": "tool_result", "tool_use_id": id, "content": output});
                    // Consecutive tool results must share one user message.
                    match messages.last_mut() {
                        Some(last) if last["role"] == "user" && last["content"].is_array() => {
                            last["content"].as_array_mut().unwrap().push(block);
                        }
                        _ => messages.push(json!({"role": "user", "content": [block]})),
                    }
                }
            }
        }
        let mut body = json!({
            "model": cfg.model,
            "max_tokens": 8192,
            "system": system(cfg, context),
            "stream": true,
            "messages": messages,
        });
        // No tools at all is a valid request (a server-less session), and an
        // empty `tools` array is rejected by the API, so the field is omitted.
        if !tools.is_empty() {
            body["tools"] = Value::Array(tools.iter().map(anthropic_schema).collect());
        }
        // Budget-based thinking (older Claude API); newer models ignore it.
        if let Some(budget) = thinking_budget(&cfg.thinking) {
            body["thinking"] = json!({"type": "enabled", "budget_tokens": budget});
            body["max_tokens"] = json!(8192 + budget as i64);
        }
        let headers = vec![
            ("x-api-key", cfg.api_key.clone()),
            ("anthropic-version", "2023-06-01".to_string()),
        ];
        // A third-party Anthropic-compatible endpoint may already end in /v1
        // (the OpenAI convention); adding another would double it.
        let base = cfg.base_url.trim_end_matches('/');
        let url = if base.ends_with("/v1") {
            format!("{base}/messages")
        } else {
            format!("{base}/v1/messages")
        };
        (url, body, headers)
    }

fn openai_request(
    cfg: &Config,
    history: &[Msg],
    tools: &[ToolSpec],
    context: &str,
) -> (String, Value, Vec<(&'static str, String)>) {
        let mut messages = vec![json!({"role": "system", "content": system(cfg, context)})];
        for m in history {
            match m {
                Msg::User(t) => messages.push(json!({"role": "user", "content": t})),
                Msg::Assistant { text, calls } => {
                    let mut msg = json!({"role": "assistant", "content": text});
                    if !calls.is_empty() {
                        msg["tool_calls"] = Value::Array(
                            calls
                                .iter()
                                .map(|c| json!({
                                    "id": c.id,
                                    "type": "function",
                                    "function": {"name": c.name, "arguments": c.input.to_string()}
                                }))
                                .collect(),
                        );
                    }
                    messages.push(msg);
                }
                Msg::ToolResult { id, output } => {
                    messages.push(json!({"role": "tool", "tool_call_id": id, "content": output}));
                }
            }
        }
        let mut body = json!({
            "model": cfg.model,
            "messages": messages,
            "stream": true,
            // Without this an OpenAI-compatible stream never reports usage.
            "stream_options": {"include_usage": true},
        });
        if !tools.is_empty() {
            body["tools"] = Value::Array(tools.iter().map(openai_schema).collect());
        }
        // Clamped to what this API documents: `xhigh` and `max` are real
        // levels elsewhere, and a rejected request here.
        if let Some(effort) = api_effort(&cfg.thinking) {
            body["reasoning_effort"] = json!(effort);
        }
        // OpenRouter honors these optional headers for attribution/rate limits.
        // A keyless custom endpoint gets no Authorization header at all: an
        // empty `Bearer ` is a 401 on most local servers, not "no auth".
        let headers = match (cfg.api_key.is_empty(), cfg.provider.name()) {
            (true, _) => Vec::new(),
            (false, "openrouter") => vec![
                ("authorization", format!("Bearer {}", cfg.api_key)),
                ("HTTP-Referer", "https://github.com/rift-shell".to_string()),
                ("X-Title", "rift".to_string()),
            ],
            (false, _) => vec![("authorization", format!("Bearer {}", cfg.api_key))],
        };
        (format!("{}/chat/completions", cfg.base_url), body, headers)
    }

/// The model's context window, for autocompact's percentage to be a fraction
/// of. Providers do not report it on the wire, so this is a guess from the
/// model id. Every guess errs low: compacting a little early costs one
/// summarization, while guessing a window too wide can overflow the request
/// and lose the turn.
/// The model's context window. An exact id learned from a real source (the
/// provider's own `/models` list, or models.dev) wins; without one the old
/// substring guess applies, erring low. `provider` selects the store's table;
/// `None` skips the lookup (tests, one-shot paths before the cache exists).
pub fn context_window_for(store: Option<&crate::store::Store>, provider: &str, model: &str) -> u64 {
    if let Some(store) = store {
        if let Some(ctx) = store.context_of(provider, model) {
            return ctx;
        }
        // The shared models.dev table is keyed by bare model id. It is the
        // fallback for every provider whose own `/models` list carries no
        // context annotation (OpenAI, local endpoints, custom gateways).
        if let Some(ctx) = store.context_of("models.dev", model) {
            return ctx;
        }
    }
    context_window(model)
}

pub fn context_window(model: &str) -> u64 {
    let m = model.to_ascii_lowercase();
    // The `1m` suffix is Anthropic's own naming for the wide-window tier.
    if m.contains("1m") || m.contains("gemini") || m.contains("gpt-4.1") {
        return 1_000_000;
    }
    if m.contains("gpt-5") {
        return 400_000;
    }
    if m.contains("gpt-4o") {
        return 128_000;
    }
    if m.contains("claude") || m.contains("o3") || m.contains("o4") || m.contains("gpt") {
        return 200_000;
    }
    // DeepSeek, GLM and anything unrecognized: the older, smaller window.
    128_000
}

/// Anthropic's usage block. Cache fields are absent on older models, and the
/// output count is often missing from `message_start`: both default to zero.
fn usage_anthropic(u: &Value) -> Option<Usage> {
    if u.is_null() {
        return None;
    }
    let get = |k: &str| u.get(k).and_then(Value::as_u64);
    // `message_start` reports output_tokens as a placeholder (usually 1), so a
    // lone output figure with no input beside it is not treated as a total.
    if get("input_tokens").is_none() && get("cache_read_input_tokens").is_none()
        && get("cache_creation_input_tokens").is_none()
    {
        return None;
    }
    Some(Usage {
        input: get("input_tokens").unwrap_or(0),
        cache_read: get("cache_read_input_tokens").unwrap_or(0),
        cache_write: get("cache_creation_input_tokens").unwrap_or(0),
        output: get("output_tokens").unwrap_or(0),
    })
}

/// OpenAI's usage block. `prompt_tokens` includes the cached ones, so the
/// cached count is subtracted to keep `input` uncached and avoid double
/// counting it in `Usage::prompt`.
fn usage_openai(u: &Value) -> Option<Usage> {
    if u.is_null() {
        return None;
    }
    let prompt = u.get("prompt_tokens").and_then(Value::as_u64)?;
    let cached = u
        .get("prompt_tokens_details")
        .and_then(|d| d.get("cached_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0)
        .min(prompt);
    Some(Usage {
        input: prompt.saturating_sub(cached),
        cache_read: cached,
        cache_write: 0,
        output: u.get("completion_tokens").and_then(Value::as_u64).unwrap_or(0),
    })
}

fn parse_anthropic(v: &Value) -> Result<Turn> {
    let mut turn = Turn { usage: usage_anthropic(&v["usage"]).unwrap_or_default(), ..Turn::default() };
    for b in v["content"].as_array().ok_or_else(|| anyhow!("no content in response"))? {
        match b["type"].as_str() {
            Some("text") => turn.text.push_str(b["text"].as_str().unwrap_or("")),
            Some("tool_use") => {
                let name = b["name"].as_str().unwrap_or_default();
                if !name.is_empty() {
                    let input = b["input"].clone();
                    turn.calls.push(ToolCall {
                        id: b["id"].as_str().unwrap_or_default().to_string(),
                        name: name.to_string(),
                        input: if input.is_null() { serde_json::json!({}) } else { input },
                    });
                }
            }
            _ => {}
        }
    }
    Ok(turn)
}

fn parse_openai(v: &Value) -> Result<Turn> {
    let msg = &v["choices"][0]["message"];
    if msg.is_null() {
        bail!("no choices in response");
    }
    let mut turn = Turn {
        text: msg["content"].as_str().unwrap_or_default().to_string(),
        thinking: msg["reasoning_content"].as_str().unwrap_or_default().to_string(),
        calls: vec![],
        usage: usage_openai(&v["usage"]).unwrap_or_default(),
    };
    for c in msg["tool_calls"].as_array().unwrap_or(&vec![]) {
        let name = c["function"]["name"].as_str().unwrap_or_default();
        if name.is_empty() {
            continue;
        }
        let args: Value = serde_json::from_str(c["function"]["arguments"].as_str().unwrap_or("{}"))
            .unwrap_or(Value::Null);
        turn.calls.push(ToolCall {
            id: c["id"].as_str().unwrap_or_default().to_string(),
            name: name.to_string(),
            input: if args.is_null() { serde_json::json!({}) } else { args },
        });
    }
    Ok(turn)
}

/// One tool spec in Anthropic's shape.
fn anthropic_schema(t: &ToolSpec) -> Value {
    json!({"name": t.name, "description": t.description, "input_schema": t.schema})
}

/// The same spec in OpenAI's shape. One spec list serves both vendors, which is
/// why it is a plain struct rather than a request fragment.
fn openai_schema(t: &ToolSpec) -> Value {
    json!({
        "type": "function",
        "function": {"name": t.name, "description": t.description, "parameters": t.schema},
    })
}

/// The harness's own two tools. `skill` exists because a skill body is
/// instructions, and instructions the model cannot read are instructions it
/// cannot follow.
pub fn builtin_tools() -> Vec<ToolSpec> {
    vec![
        ToolSpec {
            name: "shell".to_string(),
            description: format!(
                "Run a {} command. Returns stdout, [stderr] and [exit N].",
                crate::shell::SHELL
            ),
            schema: json!({
                "type": "object",
                "properties": {"command": {"type": "string", "description": format!("The {} command to run", crate::shell::SHELL)}},
                "required": ["command"],
            }),
        },
        ToolSpec {
            name: "skill".to_string(),
            description:
                "Load a skill: the full instructions of one listed under Skills. Use it before                  doing work the skill covers."
                    .to_string(),
            schema: json!({
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The skill's name, as listed"},
                },
                "required": ["name"],
            }),
        },
    ]
}

fn backoff(attempt: u32) {
    thread::sleep(Duration::from_millis(500 * 2u64.pow(attempt.min(5))));
}

fn truncate_err(s: &str) -> String {
    s.chars().take(500).collect()
}

/// A provider declared in `providers.json`. Its shape matches a stored row so
/// the rest of the app does not care where a provider came from.
#[derive(Clone)]
pub struct FileProvider {
    pub name: String,
    pub state: crate::store::ProviderState,
    pub models: Vec<String>,
}

fn text_field<'a>(
    spec: &'a serde_json::Map<String, Value>,
    keys: &[&str],
) -> Option<&'a str> {
    keys.iter()
        .find_map(|k| spec.get(*k).and_then(Value::as_str))
        .map(str::trim)
        .filter(|s| !s.is_empty())
}

/// Model ids from either an array of strings/objects or an object keyed by id.
fn models_from(v: Option<&Value>) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    match v {
        Some(Value::Array(items)) => {
            for item in items {
                let id = match item {
                    Value::String(s) => s.trim().to_string(),
                    Value::Object(map) => text_field(map, &["id", "model", "name"])
                        .unwrap_or("")
                        .to_string(),
                    _ => String::new(),
                };
                if !id.is_empty() && !out.contains(&id) {
                    out.push(id);
                }
            }
        }
        Some(Value::Object(map)) => {
            for id in map.keys() {
                let id = id.trim();
                if !id.is_empty() && !out.iter().any(|m| m == id) {
                    out.push(id.to_string());
                }
            }
        }
        _ => {}
    }
    out
}

/// Parse the provider file. One object maps each provider name to its settings,
/// optionally wrapped in `{"providers": ...}`. Entries that cannot be used (no
/// URL, unknown wire, bad name) are skipped, so one typo never hides the rest.
pub(crate) fn parse_providers(text: &str) -> Result<Vec<FileProvider>, String> {
    let root: Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
    let map = root
        .get("providers")
        .and_then(Value::as_object)
        .or_else(|| root.as_object())
        .ok_or_else(|| "expected a JSON object of providers".to_string())?;
    let mut out = Vec::new();
    for (name, spec) in map {
        let name = name.trim().to_ascii_lowercase();
        if !crate::config::valid_custom_name(&name)
            && crate::config::Provider::builtin(&name).is_none()
        {
            continue;
        }
        let Some(spec) = spec.as_object() else { continue };
        let wire = if spec.contains_key("type") || spec.contains_key("wire") {
            match text_field(spec, &["type", "wire"]).and_then(Wire::parse) {
                Some(w) => w,
                None => continue,
            }
        } else {
            Wire::OpenAi
        };
        let url = text_field(spec, &["url", "base_url", "baseURL", "baseUrl"])
            .map(crate::config::normalize_url)
            .or_else(|| crate::config::local_default_url(&name).map(str::to_string));
        let Some(url) = url else { continue };
        if !url.starts_with("http://") && !url.starts_with("https://") {
            continue;
        }
        let models = models_from(spec.get("models"));
        let model = text_field(spec, &["model"])
            .map(str::to_string)
            .or_else(|| models.first().cloned())
            .unwrap_or_default();
        let key = text_field(spec, &["api_key", "apiKey"])
            .map(str::to_string)
            .or_else(|| {
                text_field(spec, &["api_key_env", "apiKeyEnv"])
                    .and_then(|env| std::env::var(env).ok())
            })
            .unwrap_or_default();
        out.push(FileProvider {
            name,
            state: crate::store::ProviderState {
                model,
                base_url: url,
                api_key: key,
                kind: wire.as_str().to_string(),
            },
            models,
        });
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

/// Every provider the file declares. Read on every call so an edit is picked
/// up by the next `/provider`; a malformed file is treated as empty and the
/// startup check reports it.
pub fn file_providers() -> Vec<FileProvider> {
    if cfg!(test) {
        return Vec::new();
    }
    let Ok(text) = std::fs::read_to_string(crate::config::providers_path()) else {
        return Vec::new();
    };
    parse_providers(&text).unwrap_or_default()
}

pub fn file_provider(name: &str) -> Option<FileProvider> {
    let name = name.trim().to_ascii_lowercase();
    file_providers().into_iter().find(|p| p.name == name)
}

/// A URL on this machine means a local model server, never a built-in's
/// official endpoint.
fn local_endpoint(url: &str) -> bool {
    let url = url.trim().to_ascii_lowercase();
    ["://127.0.0.1", "://localhost", "://[::1]", "://0.0.0.0"]
        .iter()
        .any(|host| url.contains(host))
}

/// True when a saved row describes some endpoint other than the built-in it is
/// filed under. Older builds persisted per-run `--base-url`/`--model` flags, so
/// an OpenAI row can still be wearing a local server's model.
fn stale_builtin(provider: &Provider, saved: &crate::store::ProviderState) -> bool {
    if provider.is_custom() {
        return false;
    }
    if local_endpoint(&saved.base_url) {
        return true;
    }
    !saved.model.is_empty()
        && file_providers()
            .iter()
            .any(|f| f.name != provider.name() && f.models.contains(&saved.model))
}

/// A provider's connection details: the file wins over a stored row, so
/// editing the file is always authoritative.
///
/// A built-in owns its endpoint. If its row points at a local server or names a
/// model another provider declares, the row is stale and the built-in's own
/// defaults are returned instead. The stale catalog is dropped with it.
pub fn state(
    name: &str,
    store: &crate::store::Store,
) -> Option<crate::store::ProviderState> {
    // A built-in is resolved before the file everywhere else, so a file entry
    // using its name must not shadow it here either.
    let builtin = Provider::builtin(name);
    if builtin.is_none() {
        if let Some(p) = file_provider(name) {
            return Some(p.state);
        }
    }
    let saved = store.provider_state(name)?;
    let Some(provider) = builtin else { return Some(saved) };
    if !stale_builtin(&provider, &saved) {
        return Some(saved);
    }
    store.clear_models(name);
    Some(crate::store::ProviderState {
        model: provider.default_model().to_string(),
        base_url: provider.default_url().to_string(),
        api_key: String::new(),
        kind: saved.kind,
    })
}

pub fn is_file_backed(name: &str) -> bool {
    file_provider(name).is_some()
}

/// A startup check for a broken file: `None` when absent or valid.
pub fn file_problem() -> Option<String> {
    if cfg!(test) {
        return None;
    }
    let path = crate::config::providers_path();
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return None,
        Err(e) => return Some(format!("{}: {e}", path.display())),
    };
    parse_providers(&text)
        .err()
        .map(|e| format!("{}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Provider;

    #[test]
    fn the_context_window_guess_errs_low_for_unknown_models() {
        assert_eq!(context_window("claude-sonnet-4-5"), 200_000);
        assert_eq!(context_window("claude-3-5-sonnet"), 200_000);
        // GPT-4o is a 128k model: the guess must not call it 200k.
        assert_eq!(context_window("gpt-4o"), 128_000);
        assert_eq!(context_window("claude-sonnet-4-5-1m"), 1_000_000);
        assert_eq!(context_window("gemini-2.5-pro"), 1_000_000);
        assert_eq!(context_window("gpt-4.1"), 1_000_000);
        assert_eq!(context_window("gpt-5"), 400_000);
        // Unrecognized: the smallest common window, never an overestimate.
        assert_eq!(context_window("totally-unknown-model"), 128_000);
        assert_eq!(context_window("deepseek-v4.1-flash"), 128_000);
    }


    #[test]
    fn anthropic_usage_reads_both_halves_of_the_stream() {
        // `message_start` carries input and cache; `message_delta` the output.
        let start = json!({"input_tokens": 1200, "cache_read_input_tokens": 900,
                           "cache_creation_input_tokens": 100, "output_tokens": 1});
        let u = usage_anthropic(&start).unwrap();
        assert_eq!(u.input, 1200);
        assert_eq!(u.cache_read, 900);
        assert_eq!(u.cache_write, 100);
        assert_eq!(u.prompt(), 2200);
        // No usage at all (older model, or a different event) is not a total.
        assert!(usage_anthropic(&Value::Null).is_none());
        assert!(usage_anthropic(&json!({"output_tokens": 5})).is_none());
    }

    #[test]
    fn openai_usage_does_not_double_count_the_cached_prompt() {
        // `prompt_tokens` is the whole prompt; the cached part is inside it.
        let u = usage_openai(&json!({
            "prompt_tokens": 12000,
            "completion_tokens": 300,
            "prompt_tokens_details": {"cached_tokens": 9000},
        }))
        .unwrap();
        assert_eq!(u.input, 3000, "cached tokens must not be counted as fresh input");
        assert_eq!(u.cache_read, 9000);
        assert_eq!(u.output, 300);
        assert_eq!(u.prompt(), 12000, "the prompt is still 12000 tokens wide");
        // A provider that reports no usage is reported as unknown, not zero.
        assert!(usage_openai(&Value::Null).is_none());
        assert!(usage_openai(&json!({"completion_tokens": 3})).is_none());
    }

    #[test]
    fn the_stream_asks_for_usage() {
        // Without `stream_options.include_usage`, an OpenAI-compatible stream
        // never sends a usage block and `/stats` has nothing to report.
        let cfg = crate::config::Config::for_test();
        let (_, body, _) = openai_request(&cfg, &[Msg::User("hi".into())], &builtin_tools(), "");
        assert_eq!(body["stream_options"]["include_usage"], json!(true));
    }

    #[test]
    fn thinking_budget_ladder() {
        assert_eq!(thinking_budget("off"), None);
        assert_eq!(thinking_budget("bogus"), None);
        assert_eq!(thinking_budget("minimal"), Some(1024));
        assert_eq!(thinking_budget("medium"), Some(10240));
        assert_eq!(thinking_budget("max"), Some(31999));
    }

    #[test]
    fn openai_effort_is_clamped_to_the_levels_that_api_documents() {
        assert_eq!(api_effort("off"), None, "off must send no effort field");
        assert_eq!(api_effort("unknown"), None);
        assert_eq!(api_effort("medium"), Some("medium"));
        assert_eq!(api_effort("xhigh"), Some("high"));
        assert_eq!(api_effort("max"), Some("high"));

        assert!(effort_note("medium").is_none(), "nothing was changed");
        let note = effort_note("max").unwrap();
        assert!(note.contains("sending high"), "{note}");
    }

    #[test]
    fn msg_json_roundtrips_all_variants() {
        use crate::provider::{msg_from_json, msg_json, Msg, ToolCall};
        let hist = vec![
            Msg::User("hi".into()),
            Msg::Assistant {
                text: "t".into(),
                calls: vec![ToolCall::shell("c1", "ls")],
            },
            Msg::ToolResult { id: "c1".into(), output: "o".into() },
        ];
        for m in &hist {
            let v = msg_json(m);
            let back = msg_from_json(&v).expect("parses");
            assert_eq!(msg_json(&back), v);
        }
        assert!(msg_from_json(&serde_json::json!({"role": "zzz"})).is_none());
    }

    fn test_cfg(provider: Provider) -> Config {
        Config {
            provider,
            model: "m".into(),
            base_url: "http://localhost".into(),
            api_key: "k".into(),
            thinking: "medium".into(),
            timeout: std::time::Duration::from_secs(1),
            max_output: 100,
            jobs: 1,
            cwd: std::env::temp_dir(),
            log: None,
            prompt: None,
            print_system: false,
            spinner: true,
            model_flagged: false,
            url_flagged: false,
            key_flagged: false,
        }
    }

    #[test]
    fn every_result_of_a_batch_reaches_the_model_under_its_own_id() {
        // Parallel commands share a wait, never a result: the model must get
        // one block per call, with the body that call produced, so a batch
        // costs it nothing in information compared with running in sequence.
        let cfg = test_cfg(Provider::builtin("anthropic").unwrap());
        let hist = vec![
            Msg::User("three things".into()),
            Msg::Assistant {
                text: String::new(),
                calls: vec![
                    ToolCall::shell("c1", "echo one"),
                    ToolCall::shell("c2", "echo two"),
                    ToolCall::shell("c3", "echo three"),
                ],
            },
            Msg::ToolResult { id: "c1".into(), output: "one\n\n[exit 0]".into() },
            Msg::ToolResult { id: "c2".into(), output: "two\n\n[exit 0]".into() },
            Msg::ToolResult { id: "c3".into(), output: "three\n\n[exit 0]".into() },
        ];
        let (_, body, _) = anthropic_request(&cfg, &hist, &builtin_tools(), "");
        let mut blocks: Vec<Value> = Vec::new();
        for m in body["messages"].as_array().unwrap() {
            for b in m["content"].as_array().into_iter().flatten() {
                if b["type"] == "tool_result" {
                    blocks.push(b.clone());
                }
            }
        }
        assert_eq!(blocks.len(), 3, "not every result was sent");
        assert_eq!(blocks[0]["tool_use_id"], "c1");
        assert_eq!(blocks[1]["tool_use_id"], "c2");
        assert_eq!(blocks[2]["tool_use_id"], "c3");
        assert!(blocks[0]["content"].as_str().unwrap().contains("one"));
        assert!(blocks[2]["content"].as_str().unwrap().contains("three"));
    }

    #[test]
    fn anthropic_request_includes_thinking_budget() {
        let mut cfg = test_cfg(Provider::builtin("anthropic").unwrap());
        let (_, body, _) = anthropic_request(&cfg, &[], &builtin_tools(), "");
        assert_eq!(body["thinking"]["budget_tokens"], 10240);
        cfg.thinking = "off".into();
        let (_, body, _) = anthropic_request(&cfg, &[], &builtin_tools(), "");
        assert!(body.get("thinking").is_none());
    }

    #[test]
    fn openai_request_passes_effort_through() {
        let cfg = test_cfg(Provider::builtin("openai").unwrap());
        let (_, body, _) = openai_request(&cfg, &[], &builtin_tools(), "");
        assert_eq!(body["reasoning_effort"], "medium");
        let mut cfg = test_cfg(Provider::builtin("tokenharbor").unwrap());
        cfg.thinking = "off".into();
        let (_, body, _) = openai_request(&cfg, &[], &builtin_tools(), "");
        assert!(body.get("reasoning_effort").is_none());
    }

    #[test]
    fn a_custom_provider_uses_its_url_and_skips_auth_without_a_key() {
        let mut cfg = test_cfg(Provider::custom("ollama", crate::config::Wire::OpenAi));
        cfg.base_url = "http://localhost:11434/v1".into();
        cfg.api_key = String::new();
        let (url, _, headers) = openai_request(&cfg, &[], &builtin_tools(), "");
        assert_eq!(url, "http://localhost:11434/v1/chat/completions");
        assert!(
            headers.is_empty(),
            "a keyless local endpoint must not send an empty bearer header"
        );
    }

    #[test]
    fn a_custom_provider_can_speak_the_anthropic_shape() {
        let cfg = test_cfg(Provider::custom("claude-proxy", crate::config::Wire::Anthropic));
        let (url, _, headers) = anthropic_request(&cfg, &[], &builtin_tools(), "");
        assert!(url.ends_with("/v1/messages"), "{url}");
        assert_eq!(headers[0].0, "x-api-key");
    }

    #[test]
    fn an_anthropic_proxy_url_that_already_ends_in_v1_is_not_doubled() {
        let mut cfg = test_cfg(Provider::custom("claude-proxy", crate::config::Wire::Anthropic));
        cfg.base_url = "https://proxy.example/v1".into();
        let (url, _, _) = anthropic_request(&cfg, &[], &builtin_tools(), "");
        assert_eq!(url, "https://proxy.example/v1/messages");
    }

    #[test]
    fn parses_anthropic_tool_use() {
        let v = json!({"content": [
            {"type": "text", "text": "checking"},
            {"type": "tool_use", "id": "t1", "name": "shell", "input": {"command": "ls"}}
        ]});
        let t = parse_anthropic(&v).unwrap();
        assert_eq!(t.text, "checking");
        assert_eq!(t.calls[0].command().unwrap(), "ls");
        assert_eq!(t.calls[0].id, "t1");
    }

    #[test]
    fn parses_openai_tool_call() {
        let v = json!({"choices": [{"message": {"content": null, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "shell", "arguments": "{\"command\":\"pwd\"}"}}
        ]}}]});
        let t = parse_openai(&v).unwrap();
        assert_eq!(t.calls[0].command().unwrap(), "pwd");
    }

    #[test]
    fn parses_plain_text_reply_without_calls() {
        let v = json!({"choices": [{"message": {"content": "done"}}]});
        let t = parse_openai(&v).unwrap();
        assert!(t.calls.is_empty());
        assert_eq!(t.text, "done");
    }

    #[test]
    fn sse_decoder_survives_split_chunks() {
        let mut d = SseDecoder::default();
        // One event split mid-line across chunks, plus a complete one.
        let a = d.push(b"data: {\"type\":\"message");
        let b = d.push(b"_start\"}\n\ndata: [DONE]\n\n");
        assert!(a.is_empty());
        assert_eq!(b.len(), 2);
        assert_eq!(b[0], "{\"type\":\"message_start\"}");
        assert_eq!(b[1], "[DONE]");
        assert!(d.finish().is_none());
    }

    #[test]
    fn sse_decoder_flushes_trailing_event() {
        let mut d = SseDecoder::default();
        assert!(d.push(b"data: {\"a\":1}\n\n").len() == 1);
        assert!(d.push(b"data: {\"b\":2}").is_empty());
        assert_eq!(d.finish().as_deref(), Some("{\"b\":2}"));
    }

    fn ev(s: &str) -> Value {
        serde_json::from_str(s).unwrap()
    }

    #[test]
    fn anthropic_stream_assembles_text_and_split_tool_json() {
        let mut asm = AnthropicAsm::default();
        let mut got = String::new();
        let mut think = String::new();
        let mut on_chunk = |c: Chunk| match c {
            Chunk::Text(d) => got.push_str(d),
            Chunk::Think(d) => think.push_str(d),
        };
        asm.event(&ev(r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}"#), &mut on_chunk).unwrap();
        asm.event(&ev(r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"t1","name":"shell"}}"#), &mut on_chunk).unwrap();
        asm.event(&ev(r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"comm"}}"#), &mut on_chunk).unwrap();
        asm.event(&ev(r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"and\":\"ls\"}"}}"#), &mut on_chunk).unwrap();
        // Another tool call is kept too: the harness no longer assumes the
        // only tool is `shell`, because MCP servers add their own.
        asm.event(&ev(r#"{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"t2","name":"other"}}"#), &mut on_chunk).unwrap();
        asm.event(&ev(r#"{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"x\":1}"}}"#), &mut on_chunk).unwrap();
        let t = asm.finish().unwrap();
        assert_eq!(got, "hi");
        assert_eq!(t.text, "hi");
        assert_eq!(t.calls.len(), 2);
        assert_eq!((t.calls[0].id.as_str(), t.calls[0].command().unwrap()), ("t1", "ls"));
        assert_eq!(t.calls[1].name, "other");
        assert_eq!(t.calls[1].input["x"], 1);
    }

    #[test]
    fn anthropic_stream_collects_thinking() {
        let mut asm = AnthropicAsm::default();
        let mut think = String::new();
        let mut on_chunk = |c: Chunk| {
            if let Chunk::Think(d) = c {
                think.push_str(d);
            }
        };
        asm.event(&ev(r#"{"type":"content_block_start","index":0,"content_block":{"type":"thinking"}}"#), &mut on_chunk).unwrap();
        asm.event(&ev(r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"let me "}}"#), &mut on_chunk).unwrap();
        asm.event(&ev(r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"think"}}"#), &mut on_chunk).unwrap();
        let t = asm.finish().unwrap();
        assert_eq!(think, "let me think");
        assert_eq!(t.thinking, "let me think");
        assert!(t.text.is_empty() && t.calls.is_empty());
    }

    #[test]
    fn anthropic_stream_error_bails() {
        let mut asm = AnthropicAsm::default();
        let mut on_chunk = |_: Chunk| {};
        assert!(asm.event(&ev(r#"{"type":"error","error":{"message":"overloaded"}}"#), &mut on_chunk).is_err());
    }

    #[test]
    fn openai_stream_assembles_split_tool_args() {
        let mut asm = OpenaiAsm::default();
        let mut got = String::new();
        let mut think = String::new();
        let mut on_chunk = |c: Chunk| match c {
            Chunk::Text(d) => got.push_str(d),
            Chunk::Think(d) => think.push_str(d),
        };
        asm.event(&ev(r#"{"choices":[{"delta":{"content":"Hel"}}]}"#), &mut on_chunk).unwrap();
        asm.event(&ev(r#"{"choices":[{"delta":{"content":"lo"}}]}"#), &mut on_chunk).unwrap();
        asm.event(&ev(r#"{"choices":[{"delta":{"reasoning_content":"because"}}]}"#), &mut on_chunk).unwrap();
        asm.event(&ev(r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"shell","arguments":""}}]}}]}"#), &mut on_chunk).unwrap();
        asm.event(&ev(r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"command\":"}}]}}]}"#), &mut on_chunk).unwrap();
        asm.event(&ev(r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"pwd\"}"}}]}}]}"#), &mut on_chunk).unwrap();
        let t = asm.finish().unwrap();
        assert_eq!(got, "Hello");
        assert_eq!(think, "because");
        assert_eq!(t.text, "Hello");
        assert_eq!(t.thinking, "because");
        assert_eq!(t.calls.len(), 1);
        assert_eq!((t.calls[0].id.as_str(), t.calls[0].command().unwrap()), ("c1", "pwd"));
    }
}


#[cfg(test)]
mod catalog_tests {
    use super::*;

    /// A one-request HTTP server on localhost, so the catalog parsers are
    /// tested against real wire responses without a network or new deps.
    fn serve_once(body: &'static str) -> String {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        std::thread::spawn(move || {
            let (mut sock, _) = l.accept().unwrap();
            let mut buf = [0u8; 4096];
            let _ = std::io::Read::read(&mut sock, &mut buf);
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = std::io::Write::write_all(&mut sock, resp.as_bytes());
        });
        format!("http://{addr}")
    }

    #[test]
    fn the_provider_model_list_yields_ids_and_annotated_contexts() {
        let base = serve_once(
            r#"{"data":[
                {"id":"anthropic/claude-sonnet-4.5","context_length":1000000},
                {"id":"openai/gpt-4o","top_provider":{"context_length":128000}},
                {"id":"no-annotation/model"},
                {"id":""},
                {"id":"zero/model","context_length":0}
            ]}"#,
        );
        let client = Client::new().unwrap();
        let list = client.fetch_models(&base, "k").unwrap();
        assert_eq!(list.ids, vec!["anthropic/claude-sonnet-4.5", "no-annotation/model", "openai/gpt-4o", "zero/model"]);
        // Only positive, well-formed annotations land; empty ids never do.
        assert_eq!(list.contexts, vec![("anthropic/claude-sonnet-4.5".to_string(), 1_000_000), ("openai/gpt-4o".to_string(), 128_000)]);
    }

    #[test]
    fn models_dev_yields_bare_ids_with_context_windows() {
        let base = serve_once(
            r#"{"anthropic":{"models":{"claude-sonnet-4-5":{"limit":{"context":200000,"output":64000}}}},
                "openrouter":{"models":{"anthropic/claude-sonnet-4.5":{"limit":{"context":1000000}}}},
                "broken":{"models":{"x":{"limit":{"context":0}}},"nomodels":1}}"#,
        );
        let client = Client::new().unwrap();
        let map = client.fetch_model_contexts_from(&base);
        assert_eq!(
            map,
            vec![
                ("anthropic/claude-sonnet-4.5".to_string(), 1_000_000),
                ("claude-sonnet-4-5".to_string(), 200_000)
            ]
        );
    }

    #[test]
    fn the_shared_models_dev_table_is_a_fallback_for_every_provider() {
        let store = crate::store::Store::memory();
        store.save_contexts(
            "models.dev",
            &[("claude-sonnet-4-5".to_string(), 123_456)],
        );
        // A custom endpoint and an OpenRouter id both read the shared table.
        assert_eq!(
            context_window_for(Some(&store), "local-proxy", "claude-sonnet-4-5"),
            123_456
        );
        assert_eq!(
            context_window_for(Some(&store), "openrouter", "anthropic/claude-sonnet-4-5"),
            123_456
        );
        // A provider-specific figure still wins over the shared one.
        store.save_contexts("local-proxy", &[("claude-sonnet-4-5".to_string(), 200_000)]);
        assert_eq!(
            context_window_for(Some(&store), "local-proxy", "claude-sonnet-4-5"),
            200_000
        );
    }

    #[test]
    fn an_exact_context_beats_the_substring_guess() {
        let store = crate::store::Store::memory();
        // The guess alone would say 200_000 for this id.
        assert_eq!(context_window("claude-sonnet-4-5"), 200_000);
        store.save_contexts("anthropic", &[("claude-sonnet-4-5".to_string(), 1_000_000)]);
        assert_eq!(
            context_window_for(Some(&store), "anthropic", "claude-sonnet-4-5"),
            1_000_000
        );
        // Nothing learned: the guess applies unchanged.
        assert_eq!(context_window_for(Some(&store), "anthropic", "claude-haiku-4.5"), 200_000);
        assert_eq!(context_window_for(None, "anthropic", "claude-sonnet-4-5"), 200_000);
    }

    #[test]
    fn the_provider_file_accepts_a_bare_map_and_a_wrapper() {
        let bare = r#"{"local":{"url":"http://127.0.0.1:8787/v1","model":"local-model"}}"#;
        let p = parse_providers(bare).unwrap();
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].name, "local");
        assert_eq!(p[0].state.model, "local-model");
        assert_eq!(p[0].state.base_url, "http://127.0.0.1:8787/v1");
        assert_eq!(p[0].state.kind, "openai", "openai-compatible is the default");
        assert!(p[0].state.api_key.is_empty(), "local servers need no key");

        let wrapped = r#"{"providers":{"proxy":{
            "type":"anthropic","base_url":"https://proxy.example/v1",
            "models":["claude-a","claude-b"]
        }}}"#;
        let p = parse_providers(wrapped).unwrap();
        assert_eq!(p[0].state.kind, "anthropic");
        assert_eq!(p[0].models, vec!["claude-a", "claude-b"]);
        // One model declared and no `model`: it becomes the default.
        assert_eq!(p[0].state.model, "claude-a");
    }

    #[test]
    fn the_provider_file_reads_models_in_both_common_shapes() {
        let array = r#"{"x":{"url":"http://localhost:1/v1",
            "models":["a",{"id":"b"},{"name":"c"}]}}"#;
        assert_eq!(parse_providers(array).unwrap()[0].models, vec!["a", "b", "c"]);
        let object = r#"{"x":{"url":"http://localhost:1/v1",
            "models":{"a":{},"b":{"name":"B"}}}}"#;
        assert_eq!(parse_providers(object).unwrap()[0].models, vec!["a", "b"]);
    }

    #[test]
    fn a_broken_entry_does_not_hide_the_rest_of_the_file() {
        let text = r#"{
            "good": {"url":"http://localhost:1/v1","models":["a","b"]},
            "no-url": {"model":"x"},
            "bad-wire": {"url":"http://localhost:2/v1","type":"smoke-signals"},
            "not a name!": {"url":"http://localhost:3/v1"}
        }"#;
        let p = parse_providers(text).unwrap();
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].name, "good");
    }

    #[test]
    fn a_conventional_local_name_can_omit_its_url() {
        let p = parse_providers(r#"{"ollama":{"model":"llama3.2"}}"#).unwrap();
        assert_eq!(p[0].state.base_url, "http://localhost:11434/v1");
    }

    #[test]
    fn a_builtin_row_pointing_at_a_local_server_returns_its_own_defaults() {
        let store = crate::store::Store::memory();
        store.save_provider("openai", "lfm2.5-vl-3b", "http://127.0.0.1:8787/v1", "thk");
        store.save_models("openai", &["lfm2.5-vl-3b".to_string()]);
        let s = state("openai", &store).expect("built-in state");
        assert_eq!(s.model, "gpt-4.1");
        assert_eq!(s.base_url, "https://api.openai.com/v1");
        assert_eq!(s.api_key, "", "a local endpoint's key is not sent to OpenAI");
        assert!(
            store.cached_models("openai").is_none(),
            "the local endpoint's catalog was left behind"
        );
    }

    #[test]
    fn a_builtin_row_with_a_real_endpoint_is_left_alone() {
        let store = crate::store::Store::memory();
        store.save_provider("openai", "real-model", "https://real.example/v1", "sk-real");
        let s = state("openai", &store).expect("built-in state");
        assert_eq!(s.model, "real-model");
        assert_eq!(s.base_url, "https://real.example/v1");
        assert_eq!(s.api_key, "sk-real");
    }
}

#[cfg(test)]
mod catalog_live_tests {
    use super::*;

    /// The real models.dev payload: shape and coverage drift on their side,
    /// so this only runs on demand (`cargo test -- --ignored`).
    #[test]
    #[ignore = "needs the network; run explicitly with --ignored"]
    fn the_real_models_dev_catalog_parses_and_is_populous() {
        let client = Client::new().unwrap();
        let map = client.fetch_model_contexts();
        assert!(map.len() > 3000, "only {} models parsed", map.len());
        // The same id is published with several windows (200k first-party,
        // 198k bedrock, 1M on the 1m tier); the merge keeps the smallest, so
        // the figure errs low like the heuristic it replaces.
        let sonnet = map.iter().find(|(id, _)| id == "claude-sonnet-4-5");
        let c = sonnet.expect("sonnet missing from models.dev").1;
        assert!(c <= 200_000 && c > 0, "sonnet window {c}");
    }
}
