//! Slash commands typed at the `>` prompt. They change the session, not the files on disk.
use crate::config::{normalize_url, Config, Provider, LOCAL_PROVIDERS};

pub enum Action {
    /// Not a slash command: send to the model.
    NotCommand,
    /// Handled. The string is a report: printed neutrally.
    Message(String),
    /// Rejected input (unknown name, bad value, wrong usage): printed through
    /// the error role so a failure never looks like a report.
    Problem(String),
    /// Clear the conversation history (starts a fresh stored session).
    Reset(String),
    /// Change configuration (already applied to `cfg`); message to print.
    Changed(String),
    /// Provider switched but has no key: the loop prompts for one inline.
    NeedKey { msg: String, env: String },
    /// Leave the REPL.
    Exit,
    /// Bare `/model`: the loop refreshes catalogs, then opens the live picker.
    Models,
    /// Bare `/provider`: open the provider picker (built-in + custom).
    Providers,
    /// Show or set the thinking effort level (already validated by agent).
    Thinking(String),
    /// Summarize history into one line of context. Arg = extra instructions.
    Compact(String),
    /// List past sessions (`None`) or load one (`Some(id)`).
    Resume(Option<String>),
    /// Drop the last `n` user turns (default 1).
    Rewind(Option<String>),
    /// Session telemetry: tokens, speed, cache hit rate, cpu and memory.
    Stats,
    /// The skill catalog, one line per skill.
    Skills,
    /// Every configured MCP server, with its connection verdict.
    Mcp,
}

/// Single source of truth for every slash command: name + one-line description.
/// The input completer in `ui` reads this table, so adding a row here
/// automatically teaches Tab-completion, hints and `/help`.
pub const COMMANDS: &[(&str, &str)] = &[
    ("/help", "show this list"),
    ("/status", "show provider, model, thinking and key status"),
    ("/provider", "pick a provider; custom endpoints live in providers.json"),
    ("/model", "pick from every configured provider, or set one id"),
    ("/thinking", "show or set thinking effort (levels depend on the model)"),
    ("/compact", "summarize history to free context"),
    ("/skills", "list the skills this session can load"),
    ("/mcp", "list MCP servers, their state and their tools"),
    ("/resume", "list this directory's sessions or resume one by id (/resume all)"),
    ("/rewind", "undo the last turn(s): messages and files, e.g. /rewind 2"),
    ("/stats", "session totals: tokens, speed, cache hit, cpu, memory"),
    ("/reset", "clear conversation history"),
    ("/exit", "quit"),
];

pub const PROVIDERS: &[&str] = &["anthropic", "openai", "tokenharbor", "openrouter"];

/// Anthropic's model ids. Anthropic has no list API, so this static catalog
/// is the real menu, not a fallback. Freeform ids still work via
/// `/model <id>`, and every other provider shows only what its `/models`
/// endpoint returned.
const ANTHROPIC_MODELS: &[&str] = &["claude-sonnet-4-5", "claude-opus-5.5", "claude-haiku-4.5"];

/// Models matching `frag` anywhere in the id (case-insensitive search);
/// empty `frag` lists all. Current model is marked via the description.
pub fn model_candidates(
    catalog: &[(String, String)],
    current_provider: &str,
    current_model: &str,
    frag: &str,
) -> Vec<(String, String)> {
    let f = frag.to_ascii_lowercase();
    catalog
        .iter()
        .filter(|(_, id)| id.to_ascii_lowercase().contains(&f))
        .map(|(p, id)| {
            let desc = if p == current_provider && id == current_model {
                format!("{p}, current")
            } else {
                p.clone()
            };
            (id.clone(), desc)
        })
        .collect()
}

/// Every provider the user can select: the four hosted built-ins, the local
/// runtimes discovery has found, and every custom endpoint declared on disk.
/// Custom rows carry their wire shape, so this is the one list every provider
/// command and the picker share.
pub fn known_providers(store: &crate::store::Store) -> Vec<Provider> {
    let mut out: Vec<Provider> = PROVIDERS.iter().filter_map(|n| Provider::parse(n)).collect();
    for name in LOCAL_PROVIDERS {
        if crate::provider::local_visible(name, store) {
            if let Some(provider) = Provider::resolve(name, store) {
                out.push(provider);
            }
        }
    }
    let mut seen: std::collections::HashSet<String> =
        out.iter().map(|p| p.name().to_string()).collect();
    for fp in crate::provider::file_providers() {
        if seen.insert(fp.name.clone()) {
            if let Some(p) = Provider::from_state(&fp.name, &fp.state.kind) {
                out.push(p);
            }
        }
    }
    for (name, state) in store.provider_rows() {
        if !seen.insert(name.clone()) {
            continue;
        }
        if let Some(p) = Provider::from_state(&name, &state.kind) {
            out.push(p);
        }
    }
    out
}

/// The key that would be used for this provider right now: a key entered in
/// the app first, then the provider's environment variable. A key set through
/// `/provider` is an explicit choice, so it must be able to replace a stale
/// environment key on the next switch.
pub fn provider_key(p: &Provider, store: &crate::store::Store) -> Option<String> {
    crate::provider::state(p.name(), store)
        .map(|s| s.api_key)
        .filter(|k| !k.is_empty())
        .or_else(|| p.env_key())
}

/// Picker rows for `/provider`: every known provider, with whether it is
/// current, custom, and ready to use.
pub fn provider_candidates(store: &crate::store::Store, active: &str) -> Vec<(String, String)> {
    known_providers(store)
        .into_iter()
        .map(|p| {
            // A stored key is one the user chose; an environment key is
            // ambient. Say which one the picker is showing so a surprising
            // "key set" can be traced to its source.
            let stored = crate::provider::file_provider(p.name())
                .map(|f| f.state.api_key)
                .or_else(|| store.provider_state(p.name()).map(|s| s.api_key))
                .filter(|k| !k.is_empty());
            let from_env = p.env_key().is_some();
            let from_file = crate::provider::file_provider(p.name()).is_some();
            let mut desc = if from_file {
                format!("from providers.json, {} API", p.wire().as_str())
            } else if p.is_local() {
                "local, auto-discovered".to_string()
            } else if p.is_custom() {
                format!("custom {} API", p.wire().as_str())
            } else {
                "built-in".to_string()
            };
            if p.name() == active {
                desc = format!("current, {desc}");
            }
            if stored.is_some() {
                desc.push_str(", key set");
            } else if from_env {
                desc.push_str(", key set from env");
            } else if p.is_local() || p.is_custom() {
                desc.push_str(", no key needed");
            } else {
                desc.push_str(", no key");
            }
            (p.name().to_string(), desc)
        })
        .collect()
}

/// Every configured provider's cached catalog, or the static fallback for
/// Anthropic. This is what `/model` lists and searches: models from all
/// connected providers show up together, and picking one switches to its
/// owner.
pub fn connected_catalog(store: &crate::store::Store) -> Vec<(String, String)> {
    let files = crate::provider::file_providers();
    let file_names: std::collections::HashSet<&str> =
        files.iter().map(|f| f.name.as_str()).collect();
    // The same id can be listed by several providers (a local server is often
    // also in a stale built-in row). Show it once, owned by the provider file
    // entry, which is the one the user actually wrote down.
    let mut seen: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut out: Vec<(String, String)> = Vec::new();
    for p in known_providers(store) {
        let name = p.name();
        let usable = !p.requires_key() || provider_key(&p, store).is_some();
        if !usable {
            continue;
        }
        // The live catalog or nothing: a provider whose list has never been
        // fetched (or whose fetch keeps failing) shows no guesses, so a short
        // menu is always a fact about the connection, never a fake default.
        // Anthropic is the exception: it has no list API, so its static
        // catalog is the real one. A file provider may declare its models, and
        // those are treated as the complete catalog when present.
        let declared = files
            .iter()
            .find(|f| f.name == *name)
            .map(|f| f.models.clone())
            .unwrap_or_default();
        let cached = store.cached_models(name).map(|(ids, _)| ids).unwrap_or_default();
        let mut ids: Vec<String> = Vec::new();
        for id in declared.into_iter().chain(cached) {
            if !id.is_empty() && !ids.contains(&id) {
                ids.push(id);
            }
        }
        if ids.is_empty() && name == "anthropic" {
            ids.extend(ANTHROPIC_MODELS.iter().map(|s| s.to_string()));
        }
        for id in ids {
            if let Some(&i) = seen.get(&id) {
                if file_names.contains(name)
                    && !file_names.contains(out[i].0.as_str())
                {
                    out[i] = (name.to_string(), id);
                }
                continue;
            }
            seen.insert(id.clone(), out.len());
            out.push((name.to_string(), id));
        }
    }
    out
}

pub fn help() -> String {
    help_at(crate::term::cols() < 60)
}

fn help_at(narrow: bool) -> String {
    let width = COMMANDS.iter().map(|(n, _)| n.len()).max().unwrap_or(0);
    // Narrow windows break an aligned column: give each name its own line.
    let mut out = String::from("Commands  (up/down navigate, Enter picks, Tab completes)\n");
    for (name, desc) in COMMANDS {
        if narrow {
            out.push_str(&format!("  {name}\n      {desc}\n"));
        } else {
            out.push_str(&format!("  {name:<width$}  {desc}\n"));
        }
    }
    out.push_str("  /?  alias of /help, /quit alias of /exit");
    out
}

/// Names starting with `prefix` (case-insensitive), for Tab-completion.
pub fn candidates(prefix: &str) -> Vec<(&'static str, &'static str)> {
    let p = prefix.to_ascii_lowercase();
    COMMANDS
        .iter()
        .filter(|(n, _)| n.to_ascii_lowercase().starts_with(&p))
        .copied()
        .collect()
}

/// Closest command to a mistyped name, for "did you mean" hints.
pub fn suggest(name: &str) -> Option<&'static str> {
    let mut best: Option<(&'static str, usize)> = None;
    for (cmd, _) in COMMANDS {
        let d = edit_distance(&name.to_ascii_lowercase(), &cmd.to_ascii_lowercase());
        if d <= 3 && best.is_none_or(|(_, b)| d < b) {
            best = Some((cmd, d));
        }
    }
    best.map(|(c, _)| c)
}

// ponytail: O(n*m) scan, fine for sub-20-char command names
fn edit_distance(a: &str, b: &str) -> usize {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, &ca) in a.iter().enumerate() {
        let mut cur = vec![i + 1];
        for (j, &cb) in b.iter().enumerate() {
            cur.push((prev[j] + usize::from(ca != cb)).min(prev[j + 1] + 1).min(cur[j] + 1));
        }
        prev = cur;
    }
    prev[b.len()]
}

pub fn handle(line: &str, cfg: &mut Config, store: &crate::store::Store) -> Action {
    let line = line.trim();
    if !line.starts_with('/') {
        return Action::NotCommand;
    }
    // Capabilities may have landed since the last prompt. Resolve them before
    // commands inspect or change the level.
    cfg.refresh_thinking(store);
    let (name, arg) = match line.split_once(char::is_whitespace) {
        Some((n, a)) => (n, a.trim()),
        None => (line, ""),
    };
    match name {
        // Bare "/" lists commands instead of erroring, so typing `/` discovers.
        "/" => Action::Message(help()),
        "/help" | "/?" => Action::Message(help()),
        "/exit" | "/quit" => Action::Exit,
        "/reset" => Action::Reset("history cleared".into()),
        "/status" => Action::Message(status(cfg)),
        "/provider" => provider(arg, cfg, store),
        "/model" | "/models" => model(arg, cfg, store),
        "/thinking" => Action::Thinking(arg.to_string()),
        "/compact" => Action::Compact(arg.to_string()),
        "/mcp" => Action::Mcp,
        "/resume" => {
            if arg.is_empty() {
                Action::Resume(None)
            } else {
                Action::Resume(Some(arg.to_string()))
            }
        }
        "/rewind" => Action::Rewind(if arg.is_empty() { None } else { Some(arg.to_string()) }),
        "/stats" => Action::Stats,
        "/skills" => Action::Skills,
        "/url" => Action::Message("gone: set it inline with /provider <name> url=<url>".into()),
        "/key" => Action::Message("gone: set it inline with /provider <name> key=<key>".into()),
        "/cwd" => Action::Message("gone: start rift with -C <dir> to set the working directory".into()),
        _ => {
            let mut msg = format!("unknown command: {name}");
            if let Some(s) = suggest(name) {
                msg.push_str(&format!(", did you mean {s}?"));
            } else {
                msg.push_str(" (try /help)");
            }
            Action::Problem(msg)
        }
    }
}

/// Persist the session so restart restores provider, model, URL and key.
fn persist(store: &crate::store::Store, cfg: &Config) {
    // A file-backed provider is configured by its JSON entry, not by a row
    // copied into the database: writing one would go stale the moment the
    // file changes. A local runtime gets only its chosen model remembered;
    // its endpoint stays the one discovery found.
    if cfg.provider.is_local() {
        store.remember_model(cfg.provider.name(), &cfg.model);
    } else if !crate::provider::is_file_backed(cfg.provider.name()) {
        // An environment key stays ambient: copying it into the row would make
        // it outrank a changed or removed env var on the next run. Only a key
        // entered in the app (or a CLI flag) is stored.
        let env_key = cfg.provider.env_key();
        let key = if !cfg.key_flagged && env_key.as_deref() == Some(cfg.api_key.as_str()) {
            crate::provider::state(cfg.provider.name(), store)
                .map(|s| s.api_key)
                .filter(|k| !k.is_empty())
                .unwrap_or_default()
        } else {
            cfg.api_key.clone()
        };
        store.save_provider(cfg.provider.name(), &cfg.model, &cfg.base_url, &key);
    }
    store.set_active(cfg.provider.name());
}

fn mask(key: &str) -> String {
    if key.is_empty() {
        return "(not set)".into();
    }
    let tail: String = key.chars().rev().take(4).collect::<Vec<_>>().into_iter().rev().collect();
    format!("••••{tail}")
}

/// The current thinking level described against what the model actually
/// documents. `requested` is the level that was asked for (stored or typed);
/// when the two differ, the clamp is named instead of happening silently.
/// Shared by `/status` and the messages that report a provider/model change.
pub fn thinking_note(cfg: &Config, requested: &str) -> String {
    if !cfg.thinking_support.known {
        return format!(
            "{} (model not in the catalog: no reasoning parameters are sent)",
            cfg.thinking
        );
    }
    if !cfg.thinking_support.reasoning {
        return format!("{} (this model does not reason)", cfg.thinking);
    }
    let levels = cfg.thinking_levels().join("|");
    if cfg.thinking != requested {
        format!("{} (clamped from {requested}: this model supports {levels})", cfg.thinking)
    } else {
        format!("{} (this model supports {levels})", cfg.thinking)
    }
}

pub fn status(cfg: &Config) -> String {
    // A level wider than the model is never silently kept: the line shows the
    // level a request will actually use, plus the levels this model accepts.
    let thinking = thinking_note(cfg, &cfg.thinking);
    let key = if cfg.api_key.is_empty() {
        if cfg.provider.requires_key() { "(not set)" } else { "(not needed)" }.to_string()
    } else {
        mask(&cfg.api_key)
    };
    let custom = if cfg.provider.is_custom() {
        format!("  (custom, {} API)", cfg.provider.wire().as_str())
    } else {
        String::new()
    };
    format!(
        "  provider  {}{}\n  model     {}\n  thinking  {}\n  url       {}\n  key       {}\n  cwd       {}",
        cfg.provider.name(),
        custom,
        if cfg.model.is_empty() { "(not set)" } else { &cfg.model },
        thinking,
        cfg.base_url,
        key,
        cfg.cwd.display()
    )
}

#[derive(Default)]
struct Options {
    key: Option<String>,
    url: Option<String>,
    model: Option<String>,
    thinking: Option<String>,
}

/// Parse the `key=` / `url=` / `model=` / `thinking=` tokens accepted by
/// `/provider <name>`. A bare `http...` token is a URL, any other bare token is
/// a model id.
fn parse_options(rest: &str) -> Result<Options, String> {
    let mut o = Options::default();
    for tok in rest.split_whitespace() {
        if let Some((k, v)) = tok.split_once('=') {
            match k {
                "key" | "api_key" | "apikey" => o.key = Some(v.to_string()),
                "url" | "base_url" | "base-url" => o.url = Some(v.to_string()),
                "model" => o.model = Some(v.to_string()),
                "thinking" => o.thinking = Some(v.to_string()),
                _ => {
                    return Err(format!(
                        "unknown option: {k} (key=, url=, model=, thinking=)"
                    ))
                }
            }
        } else if tok.starts_with("http://") || tok.starts_with("https://") {
            o.url = Some(tok.to_string());
        } else {
            o.model = Some(tok.to_string());
        }
    }
    if let Some(t) = &o.thinking {
        if !crate::config::THINKING_LEVELS.contains(&t.as_str()) {
            return Err(format!(
                "unknown thinking level: {t} ({})",
                crate::config::THINKING_LEVELS.join("|")
            ));
        }
    }
    Ok(o)
}

fn provider(arg: &str, cfg: &mut Config, store: &crate::store::Store) -> Action {
    let arg = arg.trim();
    if arg.is_empty() {
        return Action::Providers;
    }
    let (name, rest) = match arg.split_once(char::is_whitespace) {
        Some((n, r)) => (n, r.trim()),
        None => (arg, ""),
    };
    if name.eq_ignore_ascii_case("add") {
        return Action::Problem(format!(
            "/provider add is gone: declare providers in {} (see the README for the format)",
            crate::config::providers_path().display()
        ));
    }
    if name.eq_ignore_ascii_case("remove") {
        return provider_remove(rest, cfg, store);
    }
    let provider = match Provider::resolve(name, store) {
        Some(p) => p,
        None => {
            return Action::Problem(format!(
                "unknown provider: {name} (built-in: {}; local: {}; custom: {})",
                PROVIDERS.join(" | "),
                LOCAL_PROVIDERS.join(" | "),
                crate::config::providers_path().display()
            ))
        }
    };
    let opts = match parse_options(rest) {
        Ok(o) => o,
        Err(e) => return Action::Problem(e),
    };
    // The model the switch will land on: an inline `model=`, else the one
    // stored for this provider, else its default. The thinking level belongs
    // to that model, so an inline `thinking=` is judged against it before
    // anything changes. A model with no metadata yet accepts any canonical
    // level; the level is kept and no reasoning parameter is sent until the
    // catalog arrives.
    let target_model = opts
        .model
        .clone()
        .or_else(|| {
            crate::provider::state(provider.name(), store)
                .map(|s| s.model)
                .filter(|m| !m.is_empty())
        })
        .unwrap_or_else(|| provider.default_model().to_string());
    let requested = opts
        .thinking
        .clone()
        .or_else(|| store.get_model_thinking(provider.name(), &target_model))
        .unwrap_or_else(|| "medium".to_string());
    if let Some(t) = &opts.thinking {
        let support = crate::provider::thinking_support(Some(store), &provider, &target_model);
        if support.known {
            let levels = crate::provider::thinking_levels(&support, &provider, &target_model);
            if !levels.contains(&t.as_str()) {
                return Action::Problem(format!(
                    "thinking level '{t}' is not supported by {target_model}. This model supports: {}",
                    levels.join(", ")
                ));
            }
        }
    }
    // An inline `key=` is the key for this switch. Otherwise a built-in always
    // asks, even when one is already stored: that is how the old key is
    // replaced instead of being kept forever.
    let key_given = opts.key.as_deref().is_some_and(|k| !k.is_empty());
    match cfg.switch_provider(&provider, store) {
        Ok(()) => {
            cfg.model = target_model;
            if let Some(u) = opts.url {
                cfg.base_url = normalize_url(&u);
            }
            if let Some(k) = opts.key {
                cfg.api_key = k;
            }
            // Restore what this target model had stored, or apply the inline
            // level. `requested` is kept before clamping so the reply can say
            // when the model changed it.
            cfg.thinking = requested.clone();
            if let Some(t) = &opts.thinking {
                store.set_model_thinking(provider.name(), &cfg.model, t);
            }
            cfg.refresh_thinking(store);
            persist(store, cfg);
            let custom = if provider.is_custom() {
                format!(" (custom, {} API)", provider.wire().as_str())
            } else {
                String::new()
            };
            let msg = format!(
                "switched to {}{custom}, model {}, url {}; thinking {}",
                provider.name(),
                if cfg.model.is_empty() { "(not set: /model)" } else { &cfg.model },
                cfg.base_url,
                thinking_note(cfg, &requested)
            );
            if provider.requires_key() && !key_given {
                return Action::NeedKey { msg, env: provider.key_env() };
            }
            Action::Changed(msg)
        }
        Err(e) => Action::Problem(format!("{e:#}")),
    }
}

/// `/provider remove <name>`: forget a custom endpoint written by an older
/// build. New custom providers live in `providers.json`, so removing one means
/// editing that file. Built-ins and local runtimes stay.
fn provider_remove(rest: &str, cfg: &mut Config, store: &crate::store::Store) -> Action {
    let name = match rest.split_whitespace().next() {
        Some(n) => n.to_ascii_lowercase(),
        None => return Action::Problem("usage: /provider remove <name>".into()),
    };
    if Provider::builtin(&name).is_some() {
        return Action::Problem(format!("cannot remove built-in provider {name}"));
    }
    if crate::provider::is_file_backed(&name) {
        return Action::Problem(format!(
            "{name} is defined in {}; remove it from that file",
            crate::config::providers_path().display()
        ));
    }
    if store.provider_state(&name).is_none() {
        return Action::Problem(format!("no custom provider named {name}"));
    }
    store.remove_provider(&name);
    if cfg.provider.name() == name {
        if let Some(fallback) = Provider::builtin("anthropic") {
            let _ = cfg.switch_provider(&fallback, store);
            persist(store, cfg);
            return Action::Changed(format!("removed {name}; switched to anthropic"));
        }
    }
    Action::Changed(format!("removed custom provider {name}"))
}

fn model(arg: &str, cfg: &mut Config, store: &crate::store::Store) -> Action {
    if arg.is_empty() {
        return Action::Models;
    }
    // Picking another connected provider's model switches to it. When two
    // providers list the same id, a provider file entry is the intended source
    // (it is the one the user wrote down), then the current provider.
    let catalog = connected_catalog(store);
    let owners: Vec<&String> = catalog
        .iter()
        .filter(|(_, id)| id == arg)
        .map(|(owner, _)| owner)
        .collect();
    let owner = owners
        .iter()
        .find(|p| crate::provider::is_file_backed(p.as_str()))
        .or_else(|| owners.iter().find(|p| p.as_str() == cfg.provider.name()))
        .or_else(|| owners.first())
        .map(|p| p.as_str());
    if let Some(owner) = owner {
        if owner != cfg.provider.name() {
            if let Some(p) = Provider::resolve(owner, store) {
                if cfg.switch_provider(&p, store).is_ok() {
                    let requested = cfg.select_model(arg.to_string(), store);
                    persist(store, cfg);
                    return Action::Changed(format!(
                        "switched to {}, model {}; thinking {}",
                        p.name(),
                        cfg.model,
                        thinking_note(cfg, &requested)
                    ));
                }
            }
        }
    }
    // The level belongs to the model being set: restore its own choice, then
    // resolve and clamp against its capabilities.
    let requested = cfg.select_model(arg.to_string(), store);
    persist(store, cfg);
    Action::Changed(format!(
        "model set to {}; thinking {}",
        cfg.model,
        thinking_note(cfg, &requested)
    ))
}

/// `/rewind [n]`: how many user turns back to truncate to. Pure, tested.
pub fn parse_rewind(arg: Option<&str>) -> Result<usize, String> {
    match arg {
        None => Ok(1),
        Some(s) => s
            .trim()
            .parse::<usize>()
            .map_err(|_| format!("usage: /rewind [n], got {s:?}"))
            .and_then(|n| if n == 0 { Err("usage: /rewind [n], n must be >= 1".into()) } else { Ok(n) }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Wire;
    use crate::store::Store;

    const ENV_VARS: &[&str] = &[
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_AUTH_TOKEN",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        "TOKENHARBOR_API_KEY",
        "OPENROUTER_API_KEY",
        "OLLAMA_API_KEY",
        "LMSTUDIO_API_KEY",
        "VLLM_API_KEY",
    ];

    static ENV_GUARD: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Run `f` with all provider env vars cleared (restored after).
    /// Tests share a process, so env mutation is serialized through a mutex.
    fn with_clean_env(f: impl FnOnce()) {
        let _g = ENV_GUARD.lock().unwrap_or_else(|e| e.into_inner());
        let saved: Vec<(&str, Option<String>)> =
            ENV_VARS.iter().map(|v| (*v, std::env::var(v).ok())).collect();
        for v in ENV_VARS {
            std::env::remove_var(v);
        }
        f();
        for (v, val) in saved {
            match val {
                Some(x) => std::env::set_var(v, x),
                None => std::env::remove_var(v),
            }
        }
    }

    #[test]
    fn bare_slash_lists_commands() {
        let store = Store::memory();
        let mut cfg = test_cfg();
        assert!(matches!(handle("/", &mut cfg, &store), Action::Message(_)));
    }

    #[test]
    fn unknown_command_suggests_closest() {
        assert_eq!(suggest("/hepl"), Some("/help"));
        assert_eq!(suggest("/modell"), Some("/model"));
        assert_eq!(suggest("/zzzzzz"), None);
    }

    #[test]
    fn candidates_match_prefix() {
        // `/mcp` joined `/model` in the /m prefix class when it was added.
        let names: Vec<_> = candidates("/m").iter().map(|(n, _)| *n).collect();
        assert_eq!(names, vec!["/model", "/mcp"]);
        assert_eq!(candidates("/").len(), COMMANDS.len());
    }

    #[test]
    fn switches_to_openrouter() {
        with_clean_env(|| {
            let store = Store::memory();
            let mut cfg = test_cfg();
            match handle("/provider openrouter", &mut cfg, &store) {
                Action::NeedKey { msg, env } => {
                    assert!(msg.contains("openrouter"));
                    assert_eq!(env, "OPENROUTER_API_KEY");
                }
                _ => panic!("expected NeedKey"),
            }
            assert_eq!(cfg.provider, Provider::builtin("openrouter").unwrap());
            assert_eq!(cfg.base_url, "https://openrouter.ai/api/v1");
            assert_eq!(store.active_provider().as_deref(), Some("openrouter"));
        });
    }

    #[test]
    fn provider_switch_with_a_key_still_prompts_for_a_replacement() {
        let _g = ENV_GUARD.lock().unwrap_or_else(|e| e.into_inner());
        let saved = std::env::var("OPENROUTER_API_KEY").ok();
        std::env::set_var("OPENROUTER_API_KEY", "test-key");
        let store = Store::memory();
        let mut cfg = test_cfg();
        // An existing key does not silently win: the switch asks for a new
        // one, and Esc leaves the old one in place.
        assert!(matches!(handle("/provider openrouter", &mut cfg, &store), Action::NeedKey { .. }));
        assert_eq!(cfg.api_key, "test-key");
        assert_eq!(
            store.provider_state("openrouter").unwrap().api_key,
            "",
            "an environment key must stay ambient, not be frozen into the row"
        );
        assert_eq!(
            provider_key(&Provider::builtin("openrouter").unwrap(), &store).as_deref(),
            Some("test-key")
        );
        match saved {
            Some(v) => std::env::set_var("OPENROUTER_API_KEY", v),
            None => std::env::remove_var("OPENROUTER_API_KEY"),
        }
    }

    #[test]
    fn a_stored_key_beats_the_environment_key() {
        use clap::Parser;
        let _g = ENV_GUARD.lock().unwrap_or_else(|e| e.into_inner());
        let saved = std::env::var("OPENROUTER_API_KEY").ok();
        std::env::set_var("OPENROUTER_API_KEY", "env-key");
        let store = Store::memory();
        store.save_provider("openrouter", "openai/gpt-4.1", "https://openrouter.ai/api/v1", "stored-key");
        let cfg = Config::from_cli(
            crate::config::Cli::parse_from(["rift", "--provider", "openrouter"]),
            &store,
            true,
        )
        .unwrap();
        assert_eq!(cfg.api_key, "stored-key", "the key entered in the app must win");
        let mut switched = Config::for_test();
        switched
            .switch_provider(&Provider::builtin("openrouter").unwrap(), &store)
            .unwrap();
        assert_eq!(switched.api_key, "stored-key", "a provider switch must use the stored key");
        match saved {
            Some(v) => std::env::set_var("OPENROUTER_API_KEY", v),
            None => std::env::remove_var("OPENROUTER_API_KEY"),
        }
    }

    #[test]
    fn an_openai_key_scoped_to_another_endpoint_is_ignored() {
        with_clean_env(|| {
            std::env::set_var("OPENAI_API_KEY", "gateway-key");
            std::env::set_var("OPENAI_BASE_URL", "https://gateway.example/v1");
            let openai = Provider::builtin("openai").unwrap();
            assert!(
                openai.env_key().is_none(),
                "a key scoped to another gateway is not an OpenAI key"
            );
            let rows = provider_candidates(&Store::memory(), "anthropic");
            let desc = rows.into_iter().find(|(n, _)| n == "openai").unwrap().1;
            assert!(desc.contains("no key"), "{desc}");

            // The same key is used when it is scoped to OpenAI itself.
            std::env::set_var("OPENAI_BASE_URL", "https://api.openai.com/v1");
            assert_eq!(openai.env_key().as_deref(), Some("gateway-key"));
            std::env::remove_var("OPENAI_BASE_URL");
            assert_eq!(openai.env_key().as_deref(), Some("gateway-key"));
        });
    }

    #[test]
    fn the_provider_picker_says_where_a_key_came_from() {
        with_clean_env(|| {
            let store = Store::memory();
            let desc = |rows: Vec<(String, String)>| {
                rows.into_iter().find(|(n, _)| n == "openai").unwrap().1
            };
            // Nothing configured anywhere.
            assert!(desc(provider_candidates(&store, "anthropic")).contains("no key"));
            // An env key is labelled as ambient, not as a saved choice.
            std::env::set_var("OPENAI_API_KEY", "env-key");
            assert!(desc(provider_candidates(&store, "anthropic")).contains("from env"));
            // A stored key wins and is labelled as the stored one.
            store.save_provider("openai", "gpt-4.1", "https://api.openai.com/v1", "stored-key");
            let d = desc(provider_candidates(&store, "anthropic"));
            assert!(d.contains("key set") && !d.contains("from env"), "{d}");
        });
    }

    #[test]
    fn model_bare_defers_to_refresh() {
        let store = Store::memory();
        let mut cfg = test_cfg();
        assert!(matches!(handle("/model", &mut cfg, &store), Action::Models));
    }

    #[test]
    fn model_freeform_id_still_sets_and_persists() {
        with_clean_env(|| {
            let store = Store::memory();
            let mut cfg = test_cfg();
            assert!(matches!(
                handle("/model foo/bar-custom", &mut cfg, &store),
                Action::Changed(_)
            ));
            assert_eq!(cfg.model, "foo/bar-custom");
            assert_eq!(store.provider_state("anthropic").unwrap().model, "foo/bar-custom");
        });
    }

    #[test]
    fn model_pick_from_other_provider_switches() {
        with_clean_env(|| {
            let store = Store::memory();
            store.save_provider(
                "openrouter",
                "openai/gpt-4.1",
                "https://openrouter.ai/api/v1",
                "k-or",
            );
            // A fetched catalog: a provider shows only models it really has,
            // so the pick has to come from the cache, not a static list.
            store.save_models("openrouter", &["openai/gpt-4.1".to_string(), "x-ai/grok-4".to_string()]);
            let mut cfg = test_cfg();
            match handle("/model x-ai/grok-4", &mut cfg, &store) {
                Action::Changed(m) => assert!(m.contains("switched to openrouter"), "{m}"),
                _ => panic!("expected provider switch"),
            }
            assert_eq!(cfg.provider, Provider::builtin("openrouter").unwrap());
            assert_eq!(cfg.model, "x-ai/grok-4");
            assert_eq!(store.active_provider().as_deref(), Some("openrouter"));
        });
    }

    #[test]
    fn catalog_groups_connected_and_searches() {
        with_clean_env(|| {
            let store = Store::memory();
            assert!(connected_catalog(&store).is_empty());
            store.save_provider("anthropic", "claude-sonnet-4-5", "https://api.anthropic.com", "k-a");
            store.save_provider(
                "openrouter",
                "openai/gpt-4.1",
                "https://openrouter.ai/api/v1",
                "k-or",
            );
            let catalog = connected_catalog(&store);
            // Anthropic shows its static catalog; openrouter has never fetched
            // a list here, so it contributes nothing rather than guesses.
            assert_eq!(catalog.len(), 3, "{catalog:?}");
            let names: Vec<_> =
                model_candidates(&catalog, "anthropic", "claude-sonnet-4-5", "SONNET")
                    .iter()
                    .map(|(n, _)| n.clone())
                    .collect();
            assert_eq!(names, vec!["claude-sonnet-4-5".to_string()]);
        });
    }

    #[test]
    fn large_catalogs_are_fully_searchable() {
        with_clean_env(|| {
            let store = Store::memory();
            let big: Vec<String> = (0..250).map(|i| format!("some-org/model-{i}")).collect();
            store.save_models("openrouter", &big);
            store.save_provider(
                "openrouter",
                "some-org/model-0",
                "https://openrouter.ai/api/v1",
                "k-or",
            );
            let catalog = connected_catalog(&store);
            assert_eq!(catalog.len(), 250);
            // The tail end is searchable, not silently dropped.
            let names: Vec<_> =
                model_candidates(&catalog, "openrouter", "", "model-249")
                    .iter()
                    .map(|(n, _)| n.clone())
                    .collect();
            assert_eq!(names, vec!["some-org/model-249".to_string()]);
        });
    }

    fn test_cfg() -> Config {
        Config {
            provider: Provider::builtin("anthropic").unwrap(),
            model: "x".into(),
            base_url: "http://localhost".into(),
            api_key: String::new(),
            thinking: "medium".into(),
            thinking_support: Default::default(),
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
    fn help_reflows_when_narrow() {
        // The column is as wide as the longest name, so assert against the
        // table itself rather than a padding count that any new command breaks.
        let width = COMMANDS.iter().map(|(n, _)| n.len()).max().unwrap_or(0);
        let wide = help_at(false);
        let want = format!("  /status{}show provider", " ".repeat(width + 2 - "/status".len()));
        assert!(wide.contains(&want), "{wide}");
        let narrow = help_at(true);
        assert!(narrow.contains("  /status\n      show provider"), "{narrow}");
        assert!(!narrow.contains("  /status    "), "{narrow}");
    }

    #[test]
    fn removed_commands_point_to_replacements() {
        let store = Store::memory();
        let mut cfg = test_cfg();
        for cmd in ["/key x", "/url http://x", "/cwd /tmp"] {
            match handle(cmd, &mut cfg, &store) {
                Action::Message(m) => assert!(m.starts_with("gone:"), "{cmd} -> {m}"),
                _ => panic!("{cmd} should explain itself"),
            }
        }
    }

    #[test]
    fn provider_inline_options_set_key_url_model() {
        with_clean_env(|| {
            let store = Store::memory();
            let mut cfg = test_cfg();
            match handle(
                "/provider openrouter key=k1 url=https://example.com/ model=m1 thinking=low",
                &mut cfg,
                &store,
            ) {
                Action::Changed(_) => {}
                _ => panic!("expected Changed"),
            }
            assert_eq!(cfg.provider, Provider::builtin("openrouter").unwrap());
            assert_eq!(cfg.api_key, "k1");
            assert_eq!(cfg.base_url, "https://example.com");
            assert_eq!(cfg.model, "m1");
            assert_eq!(cfg.thinking, "low");
            assert_eq!(
                store.get_model_thinking("openrouter", "m1").as_deref(),
                Some("low"),
                "the level is remembered for the model it was chosen on"
            );
        });
    }

    #[test]
    fn provider_thinking_is_judged_against_the_target_models_capabilities() {
        with_clean_env(|| {
            let store = Store::memory();
            // models.dev says this model reasons only in low/medium/high.
            store.save_provider_thinking(
                "openrouter",
                &[(
                    "vendor/m1".to_string(),
                    crate::provider::ThinkingSupport {
                        known: true,
                        reasoning: true,
                        effort: vec!["low".into(), "medium".into(), "high".into()],
                        ..Default::default()
                    },
                )],
            );
            let mut cfg = test_cfg();
            match handle(
                "/provider openrouter model=vendor/m1 thinking=max",
                &mut cfg,
                &store,
            ) {
                Action::Problem(m) => {
                    assert!(m.contains("not supported"), "{m}");
                    assert!(m.contains("low, medium, high"), "the model's levels are named: {m}");
                }
                _ => panic!("expected rejection"),
            }
            assert_eq!(
                cfg.provider,
                Provider::builtin("anthropic").unwrap(),
                "a rejected level must not half-apply the switch"
            );
            // A level the model does document is applied and stored per model.
            match handle(
                "/provider openrouter key=k model=vendor/m1 thinking=high",
                &mut cfg,
                &store,
            ) {
                Action::Changed(m) => assert!(m.contains("thinking high"), "{m}"),
                _ => panic!("expected a provider switch"),
            }
            assert_eq!(cfg.thinking, "high");
            assert_eq!(
                store.get_model_thinking("openrouter", "vendor/m1").as_deref(),
                Some("high")
            );
        });
    }

    #[test]
    fn a_stored_level_is_clamped_when_the_model_supports_only_toggle() {
        with_clean_env(|| {
            let store = Store::memory();
            store.set_model_thinking("openrouter", "vendor/switch", "medium");
            store.save_provider_thinking(
                "openrouter",
                &[(
                    "vendor/switch".to_string(),
                    crate::provider::ThinkingSupport {
                        known: true,
                        reasoning: true,
                        toggle: true,
                        ..Default::default()
                    },
                )],
            );
            let mut cfg = test_cfg();
            match handle(
                "/provider openrouter key=k model=vendor/switch",
                &mut cfg,
                &store,
            ) {
                Action::Changed(m) => {
                    assert!(m.contains("thinking high"), "{m}");
                    assert!(m.contains("clamped from medium"), "the clamp is named: {m}");
                }
                _ => panic!("expected the model to be set"),
            }
            assert_eq!(cfg.thinking, "high");
        });
    }

    #[test]
    fn switching_models_restores_each_models_own_level() {
        with_clean_env(|| {
            let store = Store::memory();
            let support = |levels: &[&str]| crate::provider::ThinkingSupport {
                known: true,
                reasoning: true,
                effort: levels.iter().map(|s| s.to_string()).collect(),
                ..Default::default()
            };
            store.save_provider_thinking(
                "openrouter",
                &[
                    ("vendor/a".to_string(), support(&["low", "medium", "high"])),
                    ("vendor/b".to_string(), support(&["low", "medium", "high"])),
                ],
            );
            store.set_model_thinking("openrouter", "vendor/a", "high");
            store.set_model_thinking("openrouter", "vendor/b", "low");

            let mut cfg = test_cfg();
            // The inline model owns the level, not whatever the provider's
            // previously stored model was using.
            match handle(
                "/provider openrouter key=k model=vendor/a",
                &mut cfg,
                &store,
            ) {
                Action::Changed(m) => assert!(m.contains("thinking high"), "{m}"),
                _ => panic!("expected a provider switch"),
            }
            assert_eq!(cfg.thinking, "high");

            match handle("/model vendor/b", &mut cfg, &store) {
                Action::Changed(m) => assert!(m.contains("thinking low"), "{m}"),
                _ => panic!("expected the model to be set"),
            }
            assert_eq!(cfg.model, "vendor/b");
            assert_eq!(cfg.thinking, "low", "the other model's level must not leak over");
        });
    }

    #[test]
    fn provider_rejects_bad_thinking_level() {
        with_clean_env(|| {
            let store = Store::memory();
            let mut cfg = test_cfg();
            match handle("/provider openrouter thinking=ultra", &mut cfg, &store) {
                Action::Problem(m) => assert!(m.contains("unknown thinking level"), "{m}"),
                _ => panic!("expected rejection"),
            }
            assert_eq!(cfg.provider, Provider::builtin("anthropic").unwrap());
        });
    }

    #[test]
    fn bare_provider_opens_the_picker_without_switching() {
        let store = Store::memory();
        let mut cfg = test_cfg();
        assert!(matches!(handle("/provider", &mut cfg, &store), Action::Providers));
        assert_eq!(cfg.provider, Provider::builtin("anthropic").unwrap());
    }

    #[test]
    fn provider_add_is_gone_and_points_at_the_file() {
        with_clean_env(|| {
            let store = Store::memory();
            let mut cfg = test_cfg();
            match handle(
                "/provider add ollama url=http://localhost:11434/v1 model=llama3.2",
                &mut cfg,
                &store,
            ) {
                Action::Problem(m) => {
                    assert!(m.contains("/provider add is gone"), "{m}");
                    assert!(m.contains("providers.json"), "{m}");
                }
                _ => panic!("expected /provider add to be rejected"),
            }
            assert!(store.provider_rows().is_empty());
            assert_eq!(cfg.provider, Provider::builtin("anthropic").unwrap());
        });
    }

    #[test]
    fn a_discovered_local_provider_joins_the_picker_and_catalog() {
        with_clean_env(|| {
            let store = Store::memory();
            let mut cfg = test_cfg();
            // Before discovery the runtime is resolvable but not offered.
            assert!(Provider::resolve("ollama", &store).is_some());
            assert!(!known_providers(&store).iter().any(|p| p.name() == "ollama"));

            store.save_models("ollama", &["qwen3:8b".to_string()]);
            store.save_contexts("ollama", &[("qwen3:8b".to_string(), 32_768)]);

            let rows = provider_candidates(&store, "ollama");
            assert!(
                rows.iter().any(|(n, d)| n == "ollama" && d.contains("local") && d.contains("current")),
                "{rows:?}"
            );
            assert!(connected_catalog(&store)
                .iter()
                .any(|(p, id)| p == "ollama" && id == "qwen3:8b"));
            assert_eq!(
                crate::provider::context_window_for(Some(&store), "ollama", "qwen3:8b"),
                32_768
            );

            match handle("/provider ollama", &mut cfg, &store) {
                Action::Changed(m) => assert!(m.contains("ollama"), "{m}"),
                _ => panic!("expected a provider switch"),
            }
            assert!(cfg.provider.is_local());
            assert_eq!(cfg.model, "qwen3:8b", "discovery supplies the first model");
            assert_eq!(cfg.base_url, "http://127.0.0.1:11434/v1");
            assert!(!cfg.provider.requires_key(), "local servers are keyless");
            assert_eq!(store.active_provider().as_deref(), Some("ollama"));
            assert_eq!(
                store.remembered_model("ollama").as_deref(),
                Some("qwen3:8b"),
                "the chosen local model should survive a restart"
            );
            assert!(
                store.provider_rows().is_empty(),
                "a local runtime must not be frozen into the providers table"
            );
        });
    }

    #[test]
    fn an_empty_discovery_hides_a_local_provider_again() {
        let store = Store::memory();
        store.save_models("ollama", &["qwen3:8b".to_string()]);
        assert!(known_providers(&store).iter().any(|p| p.name() == "ollama"));
        store.clear_models("ollama");
        store.clear_contexts("ollama");
        assert!(!known_providers(&store).iter().any(|p| p.name() == "ollama"));
    }

    #[test]
    fn a_provider_row_from_before_kinds_is_recovered_as_openai() {
        with_clean_env(|| {
            let store = Store::memory();
            // An older build wrote the row without a kind; its cached models
            // must not disappear from /model after the upgrade.
            store.save_provider("legacy", "local-2", "http://127.0.0.1:8787/v1", "");
            store.save_models("legacy", &["local-2".to_string(), "local-2-instruct".to_string()]);
            let p = Provider::resolve("legacy", &store).expect("legacy row resolved");
            assert_eq!(p.wire(), Wire::OpenAi);
            assert!(p.is_custom());
            assert!(known_providers(&store).iter().any(|p| p.name() == "legacy"));
            let catalog = connected_catalog(&store);
            assert!(catalog.iter().any(|(n, id)| n == "legacy" && id == "local-2"), "{catalog:?}");
        });
    }

    #[test]
    fn provider_option_aliases_are_accepted() {
        let opts = parse_options(
            "base_url=https://proxy.example/v1 api_key=k1 model=m1 thinking=high",
        )
        .unwrap();
        assert_eq!(opts.url.as_deref(), Some("https://proxy.example/v1"));
        assert_eq!(opts.key.as_deref(), Some("k1"));
        assert_eq!(opts.model.as_deref(), Some("m1"));
        assert_eq!(opts.thinking.as_deref(), Some("high"));
        assert!(parse_options("type=anthropic").is_err());
        assert!(parse_options("models=a,b").is_err());
    }

    #[test]
    fn every_local_runtime_has_a_default_endpoint() {
        for name in LOCAL_PROVIDERS {
            let provider = Provider::local(name).unwrap();
            assert!(provider.is_local());
            assert!(!provider.is_custom());
            assert!(!provider.requires_key(), "{name} must be keyless");
            assert!(provider.default_url().starts_with("http://127.0.0.1:"), "{name}");
            assert_eq!(provider.wire(), Wire::OpenAi);
        }
    }

    #[test]
    fn parse_rewind_defaults_and_rejects() {
        assert_eq!(parse_rewind(None), Ok(1));
        assert_eq!(parse_rewind(Some("2")), Ok(2));
        assert!(parse_rewind(Some("0")).is_err());
        assert!(parse_rewind(Some("x")).is_err());
    }

    #[test]
    fn stats_is_dispatched_and_listed() {
        let store = Store::memory();
        let mut cfg = test_cfg();
        assert!(matches!(handle("/stats", &mut cfg, &store), Action::Stats));
        // The completer and /help read the same table, so it must be there.
        assert!(COMMANDS.iter().any(|(n, _)| *n == "/stats"));
        assert!(candidates("/sta").iter().any(|(n, _)| *n == "/stats"));
    }
}
