//! Slash commands typed at the `>` prompt. They change the session, not the files on disk.
use crate::config::{
    local_default_url, normalize_url, valid_custom_name, Config, Provider, Wire,
};

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
    /// Autocompact: `Some(arg)` sets or clears (`off`) the threshold, already
    /// validated; `None` reports the current setting and pressure.
    Autocompact(Option<String>),
}

/// Single source of truth for every slash command: name + one-line description.
/// The input completer in `ui` reads this table, so adding a row here
/// automatically teaches Tab-completion, hints and `/help`.
pub const COMMANDS: &[(&str, &str)] = &[
    ("/help", "show this list"),
    ("/status", "show provider, model, thinking and key status"),
    ("/provider", "pick or configure a provider (/provider add for a custom one)"),
    ("/model", "pick from every configured provider, or set one id"),
    ("/thinking", "show or set thinking effort (off|minimal|low|medium|high|xhigh|max)"),
    ("/compact", "summarize history to free context"),
    ("/autocompact", "compact automatically at N% of the context window (/autocompact 80, off)"),
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

/// Every provider the user can select: the four built-ins plus every custom
/// endpoint saved on disk. Custom rows carry their wire shape, so this is the
/// one list every provider command and the picker share.
pub fn known_providers(store: &crate::store::Store) -> Vec<Provider> {
    let mut out: Vec<Provider> = PROVIDERS.iter().filter_map(|n| Provider::parse(n)).collect();
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

/// The key that would be used for this provider right now: environment first,
/// then the stored row. `None` means no key is set (which is fine for custom
/// endpoints and fatal for built-ins).
pub fn provider_key(p: &Provider, store: &crate::store::Store) -> Option<String> {
    p.env_key()
        .or_else(|| {
            crate::provider::state(p.name(), store)
                .map(|s| s.api_key)
                .filter(|k| !k.is_empty())
        })
}

/// Picker rows for `/provider`: every known provider, with whether it is
/// current, custom, and ready to use.
pub fn provider_candidates(store: &crate::store::Store, active: &str) -> Vec<(String, String)> {
    known_providers(store)
        .into_iter()
        .map(|p| {
            let keyed = provider_key(&p, store).is_some();
            let from_file = crate::provider::file_provider(p.name()).is_some();
            let mut desc = if from_file {
                format!("from providers.json, {} API", p.wire().as_str())
            } else if p.is_custom() {
                format!("custom {} API", p.wire().as_str())
            } else {
                "built-in".to_string()
            };
            if p.name() == active {
                desc = format!("current, {desc}");
            }
            if keyed {
                desc.push_str(", key set");
            } else if p.is_custom() {
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
        // a custom one may carry a hand-written list from `/provider add`.
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
        "/autocompact" => {
            if arg.is_empty() {
                Action::Autocompact(None)
            } else {
                let good = arg.eq_ignore_ascii_case("off")
                    || arg.parse::<u8>().map(|p| (1..=99).contains(&p)).unwrap_or(false);
                if good {
                    Action::Autocompact(Some(arg.to_string()))
                } else {
                    Action::Problem(format!(
                        "usage: /autocompact <1-99|off>, got {arg:?}"
                    ))
                }
            }
        }
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
    // file changes. The active name is still stored so a restart lands here.
    if !crate::provider::is_file_backed(cfg.provider.name()) {
        store.save_provider(cfg.provider.name(), &cfg.model, &cfg.base_url, &cfg.api_key);
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

pub fn status(cfg: &Config) -> String {
    // A level wider than the API is reported here as well as where it is set,
    // because the status line is where someone looks to find out what is sent.
    let thinking = match crate::provider::effort_note(&cfg.thinking) {
        Some(note) if cfg.provider.wire() != crate::config::Wire::Anthropic => {
            format!("{} ({note})", cfg.thinking)
        }
        _ => cfg.thinking.clone(),
    };
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
    wire: Option<Wire>,
    models: Vec<String>,
}

/// Parse the `key=` / `url=` / `model=` / `thinking=` / `type=` / `models=`
/// tokens shared by `/provider` and `/provider add`. A bare `http...` token is
/// a URL, any other bare token is a model id.
fn parse_options(rest: &str) -> Result<Options, String> {
    let mut o = Options::default();
    for tok in rest.split_whitespace() {
        if let Some((k, v)) = tok.split_once('=') {
            match k {
                "key" | "api_key" | "apikey" => o.key = Some(v.to_string()),
                "url" | "base_url" | "base-url" => o.url = Some(v.to_string()),
                "model" => o.model = Some(v.to_string()),
                "thinking" => o.thinking = Some(v.to_string()),
                "type" | "wire" => match Wire::parse(v) {
                    Some(w) => o.wire = Some(w),
                    None => {
                        return Err(format!(
                            "unknown provider type {v:?} (use openai or anthropic)"
                        ))
                    }
                },
                "models" => o
                    .models
                    .extend(v.split(',').map(str::trim).filter(|s| !s.is_empty()).map(str::to_string)),
                _ => {
                    return Err(format!(
                        "unknown option: {k} (key=, url=, model=, thinking=, type=, models=)"
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
        return provider_add(rest, cfg, store);
    }
    if name.eq_ignore_ascii_case("remove") {
        return provider_remove(rest, cfg, store);
    }
    let provider = match Provider::resolve(name, store) {
        Some(p) => p,
        None => {
            return Action::Problem(format!(
                "unknown provider: {name} (built-in: {}; add a custom one with /provider add {name} url=<url>)",
                PROVIDERS.join(" | ")
            ))
        }
    };
    let opts = match parse_options(rest) {
        Ok(o) => o,
        Err(e) => return Action::Problem(e),
    };
    match cfg.switch_provider(&provider, store) {
        Ok(()) => {
            if let Some(m) = opts.model {
                cfg.model = m;
            }
            if let Some(u) = opts.url {
                cfg.base_url = normalize_url(&u);
            }
            if let Some(k) = opts.key {
                cfg.api_key = k;
            }
            if let Some(t) = opts.thinking {
                cfg.thinking = t.clone();
                store.set_thinking(provider.name(), &t);
            }
            persist(store, cfg);
            let custom = if provider.is_custom() {
                format!(" (custom, {} API)", provider.wire().as_str())
            } else {
                String::new()
            };
            let msg = format!(
                "switched to {}{custom}, model {}, url {}",
                provider.name(),
                if cfg.model.is_empty() { "(not set: /model)" } else { &cfg.model },
                cfg.base_url
            );
            if cfg.api_key.is_empty() && provider.requires_key() {
                return Action::NeedKey { msg, env: provider.key_env() };
            }
            Action::Changed(msg)
        }
        Err(e) => Action::Problem(format!("{e:#}")),
    }
}

/// `/provider add <name> url=<url> [type=openai|anthropic] [key=] [model=]
/// [models=a,b] [thinking=]`: register (or update) a custom endpoint. A key is
/// optional, so local servers work with no secret at all.
fn provider_add(rest: &str, cfg: &mut Config, store: &crate::store::Store) -> Action {
    let mut name: Option<String> = None;
    let mut extra: Option<String> = None;
    let mut options: Vec<String> = Vec::new();
    for tok in rest.split_whitespace() {
        if tok.contains('=') {
            options.push(tok.to_string());
        } else if tok.starts_with("http://") || tok.starts_with("https://") {
            // A bare URL is still a URL: `/provider add ollama http://...`.
            options.push(format!("url={tok}"));
        } else if name.is_none() {
            name = Some(tok.to_ascii_lowercase());
        } else {
            extra = Some(tok.to_string());
        }
    }
    let Some(name) = name else {
        return Action::Problem(
            "usage: /provider add <name> [url=<url>] [type=openai|anthropic] [key=] [model=] [models=a,b]"
                .into(),
        );
    };
    if let Some(word) = extra {
        return Action::Problem(format!(
            "unexpected word {word:?} in /provider add (only the name is positional)"
        ));
    }
    if Provider::builtin(&name).is_some() {
        return Action::Problem(format!(
            "{name} is built in; set its key with /provider {name} key=<key>"
        ));
    }
    if name == "add" || name == "remove" {
        return Action::Problem(format!("{name} is not a usable provider name"));
    }
    if !valid_custom_name(&name) {
        return Action::Problem(
            "provider names use letters, digits, '-' and '_' (max 32 chars), e.g. ollama"
                .into(),
        );
    }
    if crate::provider::is_file_backed(&name) {
        return Action::Problem(format!(
            "{name} is defined in {}; edit that file to change it",
            crate::config::providers_path().display()
        ));
    }
    let opts = match parse_options(&options.join(" ")) {
        Ok(o) => o,
        Err(e) => return Action::Problem(e),
    };
    let url = match opts.url {
        Some(u) => u,
        None => match local_default_url(&name) {
            Some(u) => u.to_string(),
            None => {
                return Action::Problem(
                    "a custom provider needs a base URL: /provider add <name> url=<url>".into(),
                )
            }
        },
    };
    let url = normalize_url(&url);
    if !url.starts_with("http://") && !url.starts_with("https://") {
        return Action::Problem(format!("the base URL must start with http:// or https:// (got {url:?})"));
    }
    // Re-adding an existing endpoint updates it: fields that were not given
    // keep what the provider already has, so rotating a URL cannot erase a key
    // or a chosen model by accident.
    let existing = store.provider_state(&name);
    let wire = opts
        .wire
        .or_else(|| existing.as_ref().and_then(|s| Wire::parse(&s.kind)))
        .unwrap_or(Wire::OpenAi);
    let provider = Provider::custom(&name, wire);
    let key = opts
        .key
        .or_else(|| provider.env_key())
        .or_else(|| existing.as_ref().map(|s| s.api_key.clone()).filter(|k| !k.is_empty()))
        .unwrap_or_default();
    // One model in `models=` is unambiguous; take it as the default so the
    // endpoint is usable immediately.
    let model = opts
        .model
        .or_else(|| (opts.models.len() == 1).then(|| opts.models[0].clone()))
        .or_else(|| existing.as_ref().map(|s| s.model.clone()).filter(|m| !m.is_empty()))
        .unwrap_or_default();
    store.save_custom_provider(&name, wire.as_str(), &model, &url, &key);
    if !opts.models.is_empty() {
        store.save_models(&name, &opts.models);
    }
    if let Some(t) = &opts.thinking {
        store.set_thinking(&name, t);
    }
    let _ = cfg.switch_provider(&provider, store);
    persist(store, cfg);
    let key_note = if key.is_empty() { ", no key" } else { ", key saved" };
    Action::Changed(format!(
        "added custom provider {name} ({} API) at {url}{key_note}. Its models join /model",
        wire.as_str()
    ))
}

/// `/provider remove <name>`: forget a custom endpoint. Built-ins stay.
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
                    cfg.model = arg.to_string();
                    persist(store, cfg);
                    return Action::Changed(format!(
                        "switched to {}, model {}",
                        p.name(),
                        cfg.model
                    ));
                }
            }
        }
    }
    cfg.model = arg.to_string();
    persist(store, cfg);
    Action::Changed(format!("model set to {}", cfg.model))
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
    use crate::store::Store;

    const KEYS: &[&str] = &[
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "TOKENHARBOR_API_KEY",
        "OPENROUTER_API_KEY",
    ];

    static ENV_GUARD: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Run `f` with all provider key env vars cleared (restored after).
    /// Tests share a process, so env mutation is serialized through a mutex.
    fn with_clean_env(f: impl FnOnce()) {
        let _g = ENV_GUARD.lock().unwrap_or_else(|e| e.into_inner());
        let saved: Vec<(&str, Option<String>)> =
            KEYS.iter().map(|v| (*v, std::env::var(v).ok())).collect();
        for v in KEYS {
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
    fn provider_switch_with_key_skips_prompt() {
        let _g = ENV_GUARD.lock().unwrap_or_else(|e| e.into_inner());
        let saved = std::env::var("OPENROUTER_API_KEY").ok();
        std::env::set_var("OPENROUTER_API_KEY", "test-key");
        let store = Store::memory();
        let mut cfg = test_cfg();
        assert!(matches!(handle("/provider openrouter", &mut cfg, &store), Action::Changed(_)));
        assert_eq!(cfg.api_key, "test-key");
        assert_eq!(store.provider_state("openrouter").unwrap().api_key, "test-key");
        match saved {
            Some(v) => std::env::set_var("OPENROUTER_API_KEY", v),
            None => std::env::remove_var("OPENROUTER_API_KEY"),
        }
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
            assert_eq!(store.get_thinking("openrouter").as_deref(), Some("low"));
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
    fn custom_providers_are_added_usable_and_removable() {
        with_clean_env(|| {
            let store = Store::memory();
            let mut cfg = test_cfg();
            match handle(
                "/provider add ollama url=http://localhost:11434/v1 type=openai model=llama3.2",
                &mut cfg,
                &store,
            ) {
                Action::Changed(m) => assert!(m.contains("added custom provider ollama"), "{m}"),
                other => panic!("expected add, got {:?}", std::mem::discriminant(&other)),
            }
            assert_eq!(cfg.provider, Provider::custom("ollama", Wire::OpenAi));
            assert_eq!(cfg.base_url, "http://localhost:11434/v1");
            assert_eq!(cfg.model, "llama3.2");
            assert!(cfg.api_key.is_empty(), "a local endpoint needs no key");
            // The connection survives a restart, including its wire shape.
            let st = store.provider_state("ollama").unwrap();
            assert_eq!(st.kind, "openai");
            assert_eq!(store.active_provider().as_deref(), Some("ollama"));
            // It joins the provider picker and the shared model catalog.
            let rows = provider_candidates(&store, "ollama");
            assert!(rows.iter().any(|(n, d)| n == "ollama" && d.contains("current")), "{rows:?}");
            store.save_models("ollama", &["llama3.2".to_string()]);
            assert!(connected_catalog(&store)
                .iter()
                .any(|(p, id)| p == "ollama" && id == "llama3.2"));
            // And it can be removed; the session falls back to a built-in.
            match handle("/provider remove ollama", &mut cfg, &store) {
                Action::Changed(m) => assert!(m.contains("switched to anthropic"), "{m}"),
                _ => panic!("expected removal"),
            }
            assert!(store.provider_state("ollama").is_none());
            assert_eq!(cfg.provider, Provider::builtin("anthropic").unwrap());
        });
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
    fn custom_provider_names_and_urls_are_validated() {
        with_clean_env(|| {
            let store = Store::memory();
            let mut cfg = test_cfg();
            for cmd in [
                "/provider add badly named url=http://x",
                "/provider add ok name url=http://x",
                "/provider add no-url",
                "/provider add nokey url=ftp://x",
                "/provider add anthropic url=http://x",
                "/provider add weird url=http://x type=carrier-pigeon",
            ] {
                assert!(
                    matches!(handle(cmd, &mut cfg, &store), Action::Problem(_)),
                    "{cmd} should be rejected"
                );
            }
            assert!(store.provider_rows().is_empty());
        });
    }

    #[test]
    fn re_adding_a_custom_provider_keeps_its_key_and_model() {
        with_clean_env(|| {
            let store = Store::memory();
            let mut cfg = test_cfg();
            assert!(matches!(
                handle(
                    "/provider add proxy url=https://one.example type=anthropic key=secret model=claude-x",
                    &mut cfg,
                    &store
                ),
                Action::Changed(_)
            ));
            assert!(matches!(
                handle("/provider add proxy url=https://two.example", &mut cfg, &store),
                Action::Changed(_)
            ));
            let st = store.provider_state("proxy").unwrap();
            assert_eq!(st.base_url, "https://two.example");
            assert_eq!(st.api_key, "secret", "a URL update erased the key");
            assert_eq!(st.model, "claude-x", "a URL update erased the model");
            assert_eq!(st.kind, "anthropic", "a URL update changed the wire shape");
        });
    }

    #[test]
    fn well_known_local_names_need_no_url() {
        with_clean_env(|| {
            let store = Store::memory();
            let mut cfg = test_cfg();
            assert!(matches!(
                handle("/provider add ollama model=llama3.2", &mut cfg, &store),
                Action::Changed(_)
            ));
            assert_eq!(cfg.base_url, "http://localhost:11434/v1");
            assert_eq!(cfg.model, "llama3.2");
            assert!(cfg.api_key.is_empty(), "local endpoints stay keyless");
        });
    }

    #[test]
    fn provider_option_aliases_are_accepted() {
        with_clean_env(|| {
            let store = Store::memory();
            let mut cfg = test_cfg();
            assert!(matches!(
                handle(
                    "/provider add proxy base_url=https://proxy.example/v1 api_key=k1 model=m1",
                    &mut cfg,
                    &store
                ),
                Action::Changed(_)
            ));
            let st = store.provider_state("proxy").unwrap();
            assert_eq!(st.base_url, "https://proxy.example/v1");
            assert_eq!(st.api_key, "k1");
            assert_eq!(st.model, "m1");
        });
    }

    #[test]
    fn a_custom_provider_with_one_model_is_ready_immediately() {
        with_clean_env(|| {
            let store = Store::memory();
            let mut cfg = test_cfg();
            assert!(matches!(
                handle(
                    "/provider add lmstudio url=http://localhost:1234/v1 models=qwen2.5-coder",
                    &mut cfg,
                    &store
                ),
                Action::Changed(_)
            ));
            assert_eq!(cfg.model, "qwen2.5-coder");
            assert_eq!(store.cached_models("lmstudio").unwrap().0, vec!["qwen2.5-coder"]);
            assert!(known_providers(&store)
                .iter()
                .any(|p| p.is_custom() && p.name() == "lmstudio"));
        });
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
