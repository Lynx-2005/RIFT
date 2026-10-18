//! Slash commands typed at the `>` prompt. They change the session, not the files on disk.
use crate::config::{normalize_url, Config, Provider};

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
    NeedKey { msg: String, env: &'static str },
    /// Leave the REPL.
    Exit,
    /// Bare `/model`: the loop refreshes catalogs, then opens the live picker.
    Models,
    /// Show or set the thinking effort level (already validated by agent).
    Thinking(String),
    /// Summarize history into one line of context. Arg = extra instructions.
    Compact(String),
    /// Side question: same context, no history mutation. Arg = question.
    Btw(String),
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
    ("/provider", "show or switch provider (key=/url= inline)"),
    ("/model", "show or set the model for this session"),
    ("/thinking", "show or set thinking effort (off|minimal|low|medium|high|xhigh|max)"),
    ("/compact", "summarize history to free context"),
    ("/autocompact", "compact automatically at N% of the context window (/autocompact 80, off)"),
    ("/btw", "side question: same context, history untouched"),
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

/// Every provider with a key (env wins, else stored), each with its cached
/// catalog or the static fallback. This is what `/model` lists and searches.
pub fn connected_catalog(store: &crate::store::Store) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for name in PROVIDERS {
        let Some(p) = Provider::parse(name) else { continue };
        let key = std::env::var(p.key_env())
            .ok()
            .filter(|k| !k.is_empty())
            .or_else(|| store.provider_state(name).map(|s| s.api_key))
            .unwrap_or_default();
        if key.is_empty() {
            continue;
        }
        // The live catalog or nothing: a provider whose list has never been
        // fetched (or whose fetch keeps failing) shows no guesses, so a short
        // menu is always a fact about the connection, never a fake default.
        // Anthropic is the exception: it has no list API, so its static
        // catalog is the real one.
        if let Some((ids, _)) = store.cached_models(name) {
            for id in ids {
                if !id.is_empty() {
                    out.push((name.to_string(), id));
                }
            }
        } else if *name == "anthropic" {
            for id in ANTHROPIC_MODELS {
                out.push((name.to_string(), id.to_string()));
            }
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
        "/model" => model(arg, cfg, store),
        "/thinking" => Action::Thinking(arg.to_string()),
        "/compact" => Action::Compact(arg.to_string()),
        "/btw" => {
            if arg.is_empty() {
                Action::Problem("usage: /btw <question> (side question, history untouched)".into())
            } else {
                Action::Btw(arg.to_string())
            }
        }
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
    store.save_provider(cfg.provider.name(), &cfg.model, &cfg.base_url, &cfg.api_key);
    store.set_active(cfg.provider.name());
}

fn mask(key: &str) -> String {
    if key.is_empty() {
        return "(not set)".into();
    }
    let tail: String = key.chars().rev().take(4).collect::<Vec<_>>().into_iter().rev().collect();
    format!("****{tail}")
}

pub fn status(cfg: &Config) -> String {
    // A level wider than the API is reported here as well as where it is set,
    // because the status line is where someone looks to find out what is sent.
    let thinking = match crate::provider::effort_note(&cfg.thinking) {
        Some(note) if cfg.provider != crate::config::Provider::Anthropic => {
            format!("{} ({note})", cfg.thinking)
        }
        _ => cfg.thinking.clone(),
    };
    format!(
        "  provider  {}\n  model     {}\n  thinking  {}\n  url       {}\n  key       {}\n  cwd       {}",
        cfg.provider.name(),
        cfg.model,
        thinking,
        cfg.base_url,
        mask(&cfg.api_key),
        cfg.cwd.display()
    )
}

fn provider(arg: &str, cfg: &mut Config, store: &crate::store::Store) -> Action {
    let mut parts = arg.split_whitespace();
    let name = match parts.next() {
        Some(n) => n,
        None => {
            return Action::Message(format!(
                "provider: {}  (available: {})",
                cfg.provider.name(),
                PROVIDERS.join(", ")
            ))
        }
    };
    let provider = match Provider::parse(name) {
        Some(p) => p,
        None => {
            return Action::Problem(format!(
                "unknown provider: {name} ({})",
                PROVIDERS.join(" | ")
            ))
        }
    };
    // Inline overrides so /key and /url need no separate commands:
    // `/provider openrouter key=sk-.. url=https://.. model=... thinking=high`.
    // A bare `http...` token is a URL, any other bare token is a model id.
    let mut key_ov: Option<String> = None;
    let mut url_ov: Option<String> = None;
    let mut model_ov: Option<String> = None;
    let mut think_ov: Option<String> = None;
    for tok in parts {
        if let Some((k, v)) = tok.split_once('=') {
            match k {
                "key" => key_ov = Some(v.to_string()),
                "url" => url_ov = Some(v.to_string()),
                "model" => model_ov = Some(v.to_string()),
                "thinking" => think_ov = Some(v.to_string()),
                _ => return Action::Problem(format!("unknown option: {k} (key=, url=, model=, thinking=)")),
            }
        } else if tok.starts_with("http://") || tok.starts_with("https://") {
            url_ov = Some(tok.to_string());
        } else {
            model_ov = Some(tok.to_string());
        }
    }
    if let Some(t) = &think_ov {
        if !crate::config::THINKING_LEVELS.contains(&t.as_str()) {
            return Action::Problem(format!(
                "unknown thinking level: {t} ({})",
                crate::config::THINKING_LEVELS.join("|")
            ));
        }
    }
    match cfg.switch_provider(provider, store) {
        Ok(()) => {
            if let Some(m) = model_ov {
                cfg.model = m;
            }
            if let Some(u) = url_ov {
                cfg.base_url = normalize_url(&u);
            }
            if let Some(k) = key_ov {
                cfg.api_key = k;
            }
            if let Some(t) = think_ov {
                cfg.thinking = t.clone();
                store.set_thinking(provider.name(), &t);
            }
            persist(store, cfg);
            let msg = format!(
                "switched to {}, model {}, url {}",
                provider.name(),
                cfg.model,
                cfg.base_url
            );
            if cfg.api_key.is_empty() {
                return Action::NeedKey { msg, env: provider.key_env() };
            }
            Action::Changed(msg)
        }
        Err(e) => Action::Problem(format!("{e:#}")),
    }
}

fn model(arg: &str, cfg: &mut Config, store: &crate::store::Store) -> Action {
    if arg.is_empty() {
        return Action::Models;
    }
    // Picking another connected provider's model switches to it.
    let catalog = connected_catalog(store);
    if let Some((owner, _)) = catalog.iter().find(|(_, id)| id == arg) {
        if owner != cfg.provider.name() {
            if let Some(p) = Provider::parse(owner) {
                if cfg.switch_provider(p, store).is_ok() {
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
            assert_eq!(cfg.provider, Provider::Openrouter);
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
            assert_eq!(cfg.provider, Provider::Openrouter);
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
            provider: Provider::Anthropic,
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
            assert_eq!(cfg.provider, Provider::Openrouter);
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
            assert_eq!(cfg.provider, Provider::Anthropic);
        });
    }

    #[test]
    fn bare_provider_lists_without_switching() {
        let store = Store::memory();
        let mut cfg = test_cfg();
        match handle("/provider", &mut cfg, &store) {
            Action::Message(m) => assert!(m.contains("provider:"), "{m}"),
            _ => panic!("expected status message"),
        }
        assert_eq!(cfg.provider, Provider::Anthropic);
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

    #[test]
    fn btw_without_question_shows_usage() {
        let store = Store::memory();
        let mut cfg = test_cfg();
        match handle("/btw", &mut cfg, &store) {
            Action::Problem(m) => assert!(m.contains("usage"), "{m}"),
            _ => panic!("expected usage"),
        }
        assert!(matches!(handle("/btw what is this?", &mut cfg, &store), Action::Btw(_)));
    }
}
