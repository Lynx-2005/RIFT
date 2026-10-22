//! The agent loop: user turn -> model -> shell tool calls -> results -> ... until done.
use crate::commands::{self, Action};
use crate::config::{self, Config};
use crate::provider::{Chunk, Client, Msg, ToolCall, Turn};
use crate::shell;
use crate::ui;
use anyhow::Result;
use std::fs::OpenOptions;
use std::io::Write;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc, Arc, Mutex,
};
use std::time::Duration;

pub fn run(mut cfg: Config, store: crate::store::Store) -> Result<()> {
    ui::quiet_motion(!cfg.spinner);
    if cfg.print_system {
        println!("{}", crate::provider::system(&cfg, &crate::skills::catalog(&cfg.cwd)));
        return Ok(());
    }
    if let Some(problem) = crate::provider::file_problem() {
        ui::warn(&problem);
    }
    let client = Client::new()?;
    // A local runtime has no default model: its catalog is the default. When
    // this is the first run (or the cache was cleared), discover once before
    // the first request so `rift --provider ollama` is usable immediately.
    if cfg.provider.is_local() && cfg.model.is_empty() {
        crate::local::discover_all(&store, &client);
        if let Some(model) = store
            .cached_models(cfg.provider.name())
            .and_then(|(ids, _)| ids.into_iter().next())
        {
            cfg.model = model;
        }
    }
    // Both of these are optional extras and neither may stop a session from
    // opening: a skill directory that is unreadable and a server that will not
    // start are both reported, then left behind.
    let tools = Tools::connect(&cfg.cwd);
    for note in tools.take_notes() {
        ui::info(&note);
    }
    // A one-shot run has no later turn to pick up a server that was still
    // connecting, so give the initial connections a bounded head start.
    // Interactive sessions wait inside their first turn instead.
    if cfg.prompt.is_some() {
        tools.wait_ready(Duration::from_secs(5));
        for note in tools.take_notes() {
            ui::info(&note);
        }
    }
    let mut session = Session::new(cfg.clone(), client, tools);
    persist_session(&store, &session.cfg);
    // Every launch starts a new conversation. Continuing one behind your back
    // was wrong: `/resume` is how you go back to an earlier session, and a
    // session is still tagged with the directory it happened in so the list
    // says where each one belongs.
    let one_shot = cfg.prompt.is_some();
    let where_from = session.cfg.cwd.display().to_string();
    let mut session_id = store.create_session(
        session.cfg.provider.name(),
        &session.cfg.model,
        "",
        if one_shot { "" } else { &where_from },
    );

    // Warm the model catalog without blocking the first prompt. The loop
    // rebuilds the menu from the cache on every prompt, so fresh rows appear
    // as soon as the fetch lands.
    spawn_catalog_refresh();
    // Local servers change independently of rift (a model is pulled, a server
    // starts or stops), so they are re-probed for the life of the session.
    spawn_local_discovery();

    if let Some(prompt) = cfg.prompt.clone() {
        // One-shot: nothing can be typed mid-turn, so the queues stay empty.
        session.turn(&prompt, &store, session_id, &WhileWorking::new())?;
        return Ok(());
    }

    ui::banner(&session.cfg);
    crate::stats::spawn_sampler();
    crate::term::watch_resize();
    let mut rl = ui::Reader::new();
    // Lines typed while the agent works land here; they run oldest first.
    let work = WhileWorking::new();
    let mut prefill: Option<String> = None;
    loop {
        // Servers that landed while the prompt sat idle are announced here;
        // the connection itself never blocked the reader.
        for note in session.tools.take_notes() {
            ui::info(&note);
        }
        let next = match work.typed.take() {
            Some(line) => Some(line),
            None => {
                // Text still being typed when the last turn ended belongs in
                // the prompt, not in the transcript: open with it so the user
                // can finish it and press Enter when they mean to.
                let mut start = prefill.take();
                if let Some(draft) = work.typed.take_draft() {
                    match start.as_mut() {
                        Some(text) => text.push_str(&draft),
                        None => start = Some(draft),
                    }
                }
                let providers =
                    commands::provider_candidates(&store, session.cfg.provider.name());
                rl.read_line(
                    &session.cfg,
                    &commands::connected_catalog(&store),
                    &providers,
                    start.as_deref(),
                )?
            }
        };
        let Some(raw) = next else { break };
        prefill = None;
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        match commands::handle(line, &mut session.cfg, &store) {
            Action::Exit => break,
            Action::Reset(msg) => {
                session.history.clear();
                session_id = store.create_session(
                    session.cfg.provider.name(),
                    &session.cfg.model,
                    "",
                    &session.cfg.cwd.display().to_string(),
                );
                ui::ok(&msg);
            }
            Action::Changed(msg) => {
                store.touch_session(session_id, session.cfg.provider.name(), &session.cfg.model);
                ui::ok(&msg);
                // A just-configured provider has no cached catalog yet; fetch
                // it in the background so /model works without a restart.
                spawn_catalog_refresh();
            }
            Action::Models => {
                // Open from cache when there is one: the network refresh runs
                // in the background, so a slow models.dev or a dead local
                // server never stalls the picker.
                let mut catalog = commands::connected_catalog(&store);
                if catalog.is_empty() {
                    // Nothing cached: fetch the provider lists once. This skips
                    // models.dev (context windows, not menu rows), which is the
                    // slow download that used to make this command feel hung.
                    refresh_model_lists(&store, &session.client);
                    catalog = commands::connected_catalog(&store);
                }
                if catalog.is_empty() {
                    let configured = commands::provider_candidates(&store, session.cfg.provider.name());
                    ui::info(&format!(
                        "no models yet. Configured providers:\n  {}\nSet a key with /provider <name>, start a local server (Ollama, LM Studio, vLLM), or declare a custom endpoint in {}",
                        configured
                            .iter()
                            .map(|(n, d)| format!("{n} ({d})"))
                            .collect::<Vec<_>>()
                            .join("\n  "),
                        config::providers_path().display()
                    ));
                } else {
                    // Refresh stale lists without holding the prompt.
                    spawn_catalog_refresh();
                    // Open the live picker: type to filter, up/down to pick, Enter sets.
                    prefill = Some("/model ".to_string());
                }
            }
            Action::Providers => {
                if crate::term::is_tty() {
                    // The picker itself is the completion menu: prefill the
                    // command and let up/down + Enter choose. No refresh
                    // needed: the rows describe configuration, not a catalog.
                    prefill = Some("/provider ".to_string());
                } else {
                    // No menu to draw: print the same rows as text.
                    let rows = commands::provider_candidates(&store, session.cfg.provider.name());
                    ui::info(
                        &rows
                            .iter()
                            .map(|(n, d)| format!("  {n:<12} {d}"))
                            .collect::<Vec<_>>()
                            .join("\n"),
                    );
                }
            }
            Action::NeedKey { msg, env } => {
                ui::ok(&msg);
                match ui::prompt_secret(&format!("Paste {env} (Enter saves, Esc skips): ")) {
                    Some(k) => {
                        session.cfg.api_key = k;
                        persist_session(&store, &session.cfg);
                        store.touch_session(
                            session_id,
                            session.cfg.provider.name(),
                            &session.cfg.model,
                        );
                        // The key just made this provider usable: fetch its
                        // model list now so /model reflects it without a
                        // restart or a second /provider trip.
                        spawn_catalog_refresh();
                        ui::ok(&format!(
                            "key saved for {}. Its models join /model once the list lands",
                            session.cfg.provider.name()
                        ));
                    }
                    None => {
                        ui::info("no key entered. Set it later with /provider <name> key=<key>")
                    }
                }
            }
            Action::Thinking(arg) => {
                if arg.is_empty() {
                    ui::info(&format!(
                        "thinking: {} ({})",
                        session.cfg.thinking,
                        config::THINKING_LEVELS.join("|")
                    ));
                } else if config::THINKING_LEVELS.contains(&arg.as_str()) {
                    session.cfg.thinking = arg.clone();
                    store.set_thinking(session.cfg.provider.name(), &arg);
                    match crate::provider::effort_note(&arg) {
                        Some(note) if session.cfg.provider.wire() != config::Wire::Anthropic => {
                            ui::ok(&format!("thinking set to {arg} ({note})"))
                        }
                        _ => ui::ok(&format!("thinking set to {arg}")),
                    }
                } else {
                    ui::error(&format!(
                        "unknown thinking level: {arg} ({})",
                        config::THINKING_LEVELS.join("|")
                    ));
                }
            }
            Action::Compact(extra) => session.compact(&extra, &store, session_id),
            Action::Resume(opt) => match opt {
                None => resume_list(&store, Some(&where_from)),
                Some(id) => {
                    if id.eq_ignore_ascii_case("all") {
                        resume_list(&store, None);
                    } else if let Some(new_id) = session.resume(&id, &store, &where_from) {
                        session_id = new_id;
                    }
                }
            },
            Action::Rewind(opt) => match commands::parse_rewind(opt.as_deref()) {
                Ok(n) => {
                    let turns = session
                        .history
                        .iter()
                        .filter(|m| matches!(m, Msg::User(_)))
                        .count();
                    let left = turns.saturating_sub(n);
                    let removed = rewind(&mut session.history, n);
                    if removed == 0 {
                        ui::info("nothing to rewind");
                    } else {
                        store.save_messages(session_id, &session.history);
                        let mut files = String::new();
                        if let Ok(dir) = std::env::current_dir() {
                            files = match crate::checkpoint::put_back_all(
                                session_id,
                                left + 1,
                                turns,
                                &dir,
                            ) {
                                Some(Ok(r)) if r.unattributed => {
                                    ", no record of what that turn changed, so the files were left alone".to_string()
                                }
                                Some(Ok(r)) => {
                                    let mut said = format!(
                                        ", {} file(s) restored, {} removed",
                                        r.restored, r.removed
                                    );
                                    if r.changed_since > 0 {
                                        said.push_str(&format!(
                                            ", {} changed since by something else and left alone",
                                            r.changed_since
                                        ));
                                    }
                                    if r.partial {
                                        said.push_str("; that snapshot ran out of room, so files made since were kept");
                                    }
                                    said
                                }
                                Some(Err(e)) => format!(", files not restored: {e}"),
                                None => ", no snapshot for that turn: files untouched".to_string(),
                            };
                            crate::checkpoint::drop_after(session_id, left);
                        }
                        ui::ok(&format!("rewound {removed} message(s){files}"));
                        // What the model holds now is what the screen should show.
                        ui::transcript(&session.history);
                    }
                }
                Err(e) => ui::error(&e),
            },
            Action::Skills => {
                let c = crate::skills::catalog(&session.cfg.cwd);
                if c.is_empty() {
                    ui::info("no skills. Put one at .rift/skills/<name>/SKILL.md");
                } else {
                    ui::info(&c);
                }
            }
            Action::Stats => {
                // The same two numbers autocompact compares, so a compaction
                // can be checked against what the last request actually used.
                let ctx = session
                    .last
                    .lock()
                    .ok()
                    .and_then(|l| *l)
                    .map(|u| {
                        (
                            u.prompt(),
                            crate::provider::context_window_for(
                                Some(&store),
                                session.cfg.provider.name(),
                                &session.cfg.model,
                            ),
                        )
                    });
                ui::stats_report(ctx);
            }
            Action::Mcp => {
                let m = session.tools.mcp.lock();
                match m {
                    Ok(mut m) => println!("{}", m.status()),
                    Err(_) => ui::error("the MCP connection was lost"),
                }
            }
            Action::Autocompact(arg) => {
                session.autocompact(arg.as_deref(), &store);
            }
            Action::Message(msg) => ui::info(&msg),
            Action::Problem(msg) => ui::error(&msg),
            Action::NotCommand => {
                if let Err(e) = session.turn(line, &store, session_id, &work) {
                    ui::error(&format!("{e:#}"));
                }
                for note in session.tools.take_notes() {
                    ui::info(&note);
                }
            }
        }
    }
    Ok(())
}

fn persist_session(store: &crate::store::Store, cfg: &Config) {
    // A flagged value is this run's override, so whatever is already stored
    // wins for that field. With nothing stored yet the flag is also the only
    // thing that can configure the provider, so then it is kept.
    let saved = crate::provider::state(cfg.provider.name(), store);
    let model = match (&saved, cfg.model_flagged) {
        (Some(s), true) => s.model.clone(),
        _ => cfg.model.clone(),
    };
    let url = match (&saved, cfg.url_flagged) {
        (Some(s), true) => s.base_url.clone(),
        _ => cfg.base_url.clone(),
    };
    let mut key = match (&saved, cfg.key_flagged) {
        (Some(s), true) => s.api_key.clone(),
        _ => cfg.api_key.clone(),
    };
    // Keep an environment key out of the row: it should stop applying when the
    // variable is changed or removed, rather than being frozen here.
    if !cfg.key_flagged && cfg.provider.env_key().as_deref() == Some(cfg.api_key.as_str()) {
        key = saved
            .as_ref()
            .map(|s| s.api_key.clone())
            .filter(|k| !k.is_empty())
            .unwrap_or_default();
    }
    if cfg.provider.is_local() {
        store.remember_model(cfg.provider.name(), &model);
    } else if !crate::provider::is_file_backed(cfg.provider.name()) {
        store.save_provider(cfg.provider.name(), &model, &url, &key);
    }
    store.set_active(cfg.provider.name());
}

/// Refresh the catalog on a background thread: no network in the typing path.
/// A separate connection, since Store is not shared between threads. Used at
/// startup and after a provider change, whose catalog is never cached yet.
pub(crate) fn spawn_catalog_refresh() {
    std::thread::spawn(|| {
        let Ok(store) = crate::store::Store::open() else { return };
        let Ok(client) = Client::new() else { return };
        refresh_catalogs(&store, &client);
    });
}

/// Re-probe the local runtimes for the life of the session. This is a single
/// long-lived thread: `/provider` changes must not start another loop every
/// time. The first round comes from `spawn_catalog_refresh`, so this one waits
/// before its first pass.
pub(crate) fn spawn_local_discovery() {
    if cfg!(test) {
        return;
    }
    std::thread::spawn(|| {
        let Ok(store) = crate::store::Store::open() else { return };
        let Ok(client) = Client::new() else { return };
        loop {
            std::thread::sleep(Duration::from_secs(crate::local::REFRESH_SECS));
            crate::local::discover_all(&store, &client);
        }
    });
}

/// Refresh stale model catalogs for connected providers (Anthropic has no
/// list API, so it always uses the static list). Safe to run on a background
/// thread while the user is typing: every write goes through the row lock,
/// and a failed fetch is reported rather than silently shrinking the menu.
pub(crate) fn refresh_catalogs(store: &crate::store::Store, client: &Client) {
    // Local runtimes first: a server that was just started should show up in
    // /model before the slower models.dev download finishes.
    refresh_model_lists(store, client);
    // Context windows from models.dev land even when no provider list needs a
    // refresh: the map changes on its own schedule, covers every provider at
    // once, and costs one keyless request.
    let dev = client.fetch_model_contexts();
    if !dev.is_empty() {
        store.save_contexts("models.dev", &dev);
    }
}

/// The provider `/models` half of a refresh, without the models.dev download.
/// The interactive `/model` path calls this when it has nothing cached: the
/// menu needs model ids, not context windows.
pub(crate) fn refresh_model_lists(store: &crate::store::Store, client: &Client) {
    // Local runtimes answer through their own native metadata APIs (and are
    // refreshed by their own thread), so they do not use the generic `/models`
    // path below.
    crate::local::discover_all(store, client);
    // Built-ins with a key, plus every custom endpoint: a local server with no
    // key is still worth asking for its model list.
    let need: Vec<(String, String, String)> = crate::commands::known_providers(store)
        .into_iter()
        .filter_map(|p| {
            let name = p.name();
            if name == "anthropic" || p.is_local() || store.models_fresh(name) {
                return None;
            }
            // A file entry that lists its own models is already complete: no
            // request to a server that may not be running, and no warning.
            if crate::provider::file_provider(name)
                .is_some_and(|f| !f.models.is_empty())
            {
                return None;
            }
            let key = crate::commands::provider_key(&p, store);
            if p.requires_key() && key.is_none() {
                return None;
            }
            let url = crate::provider::state(name, store)
                .map(|s| s.base_url)
                .filter(|u| !u.is_empty())
                .unwrap_or_else(|| p.default_url().to_string());
            if url.is_empty() {
                return None;
            }
            Some((name.to_string(), url, key.unwrap_or_default()))
        })
        .collect();
    if need.is_empty() {
        return;
    }
    // ponytail: scoped threads, bounded parallelism, no new deps
    std::thread::scope(|s| {
        let mut handles = Vec::new();
        for (name, url, key) in &need {
            handles.push(s.spawn(|| (name.clone(), client.fetch_models(url, key))));
        }
        for h in handles {
            if let Ok((name, res)) = h.join() {
                match res {
                    Ok(list) => {
                        store.save_models(&name, &list.ids);
                        // The provider's own list is the better source when it
                        // carries context lengths (OpenRouter does): exact ids.
                        if !list.contexts.is_empty() {
                            store.save_contexts(&name, &list.contexts);
                        }
                    }
                    Err(e) => crate::ui::warn(&format!(
                        "{name}: could not refresh the model list ({e:#}); showing what is cached"
                    )),
                }
            }
        }
    });
}

fn resume_list(store: &crate::store::Store, cwd: Option<&str>) {
    let rows = store.list_sessions(20, cwd);
    if rows.is_empty() {
        ui::info(if cwd.is_some() {
            "no past sessions in this directory. /resume all searches every directory"
        } else {
            "no past sessions"
        });
        return;
    }
    // Narrow windows break a one-line record: split it into id/age, then detail.
    let narrow = crate::term::cols() < 60;
    let mut out = String::from(if cwd.is_some() {
        "sessions here (newest first). /resume <id> loads one, /resume all widens the search:"
    } else {
        "sessions everywhere (newest first). /resume <id> loads one:"
    });
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    for r in rows {
        let age = age(now.saturating_sub(r.created_at));
        let title = if r.title.is_empty() { "".to_string() } else { format!(" - {}", r.title) };
        if narrow {
            out.push_str(&format!(
                "\n  {}  {}\n      {}, {}, {} msgs{}",
                r.id, age, r.provider, r.model, r.count, title
            ));
        } else {
            out.push_str(&format!(
                "\n  {}  {}, {}, {}, {} msgs{}",
                r.id, age, r.provider, r.model, r.count, title
            ));
        }
        // `/resume all` shows what came from where; the scoped list needs no
        // such label, every row is already from where you stand.
        if cwd.is_none() && !r.cwd.is_empty() {
            out.push_str(&format!(" @ {}", short_dir(&r.cwd)));
        }
    }
    println!("{out}");
}

/// Last two path components, so a `/resume all` list stays one line per row.
fn short_dir(path: &str) -> String {
    let p = std::path::Path::new(path);
    let name = p.file_name().and_then(|s| s.to_str());
    let parent = p.parent().and_then(|s| s.file_name()).and_then(|s| s.to_str());
    match (parent, name) {
        (Some(parent), Some(name)) => {
            format!("{parent}{}{name}", std::path::MAIN_SEPARATOR)
        }
        _ => path.to_string(),
    }
}

// ponytail: coarse buckets, resume list only
fn age(secs: u64) -> String {
    if secs < 60 {
        "now".into()
    } else if secs < 3600 {
        format!("{}m", secs / 60)
    } else if secs < 86400 {
        format!("{}h", secs / 3600)
    } else {
        format!("{}d", secs / 86400)
    }
}

pub struct Session {
    pub cfg: Config,
    client: Client,
    pub history: Vec<Msg>,
    /// The toolset for this session: the harness's own tools plus any MCP
    /// server's.
    tools: Tools,
    /// What rides along with the system prompt. Recomputed per turn, because a
    /// skill written while rift is open should be usable in the next turn.
    context: String,
    /// The most recent request's usage, for autocompact.
    last: LastUsage,
}

impl Session {
    fn new(cfg: Config, client: Client, tools: Tools) -> Self {
        let context = crate::skills::catalog(&cfg.cwd);
        Self {
            cfg,
            client,
            history: Vec::new(),
            tools,
            context,
            last: Arc::new(std::sync::Mutex::new(None)),
        }
    }

    /// Runs one user request until the model stops calling tools (no step cap, like pi).
    /// Esc/Ctrl-C aborts: the turn ends and the `>` prompt returns.
    fn turn(
        &mut self,
        input: &str,
        store: &crate::store::Store,
        session_id: i64,
        work: &WhileWorking,
    ) -> Result<()> {
        let first = self.history.is_empty();
        if first {
            // Fast MCP servers get a moment to land so the first prompt sees
            // their tools; slower ones join a later round instead of holding
            // the session back.
            self.tools.wait_first_turn(Duration::from_millis(1500));
            for note in self.tools.take_notes() {
                ui::info(&note);
            }
        }
        self.history.push(Msg::User(input.to_string()));
        if first {
            let t: String = input.chars().take(60).collect();
            store.set_title(session_id, t.trim());
        }
        self.log(&serde_json::json!({"role": "user", "content": input}));
        store.save_messages(session_id, &self.history);
        // Snapshot before any work runs, so `/rewind` can put the tree back.
        // A one-shot run has no rewind to serve, so it pays nothing for one.
        let turn = self
            .history
            .iter()
            .filter(|m| matches!(m, Msg::User(_)))
            .count();
        if self.cfg.prompt.is_none() {
            if let Ok(dir) = std::env::current_dir() {
                if let Err(e) = crate::checkpoint::take(session_id, turn, &dir) {
                    ui::warn(&format!("no snapshot for this turn: {e}"));
                }
            }
        }
        // From here to the end of the turn the prompt row is ours: it stays on
        // screen and typable while the model thinks, writes, and runs things.
        ui::turn_begin();
        // A skill added while the session was open is usable now.
        self.context = crate::skills::catalog(&self.cfg.cwd);
        let outcome = run_loop(
            &self.cfg,
            &self.client,
            &mut self.history,
            self.cfg.log.clone(),
            work,
            &self.tools,
            &self.context,
            &self.last,
        );
        ui::turn_close();
        self.maybe_autocompact(store, session_id);
        // Now that the turn is over, work out what it changed. A rewind then
        // undoes this session's own edits, and leaves alone anything that was
        // already different for other reasons.
        if self.cfg.prompt.is_none() {
            if let Ok(dir) = std::env::current_dir() {
                let _ = crate::checkpoint::mark(session_id, turn, &dir);
            }
        }
        match outcome {
            Ok(LoopEnd::Done) => {}
            Ok(LoopEnd::Cancelled) => ui::info("stopped (Esc)"),
            Err(e) => {
                store.save_messages(session_id, &self.history);
                return Err(e);
            }
        }
        store.save_messages(session_id, &self.history);
        Ok(())
    }

    /// Compacts on its own when the stored threshold says so. Called after a
    /// turn finishes: the last request's prompt usage is the context as the
    /// model just saw it, and the output it is about to write only grows it.
    /// Compaction runs here, not mid-turn, so a turn is never torn apart under
    /// itself; the next request then starts from the summary.
    fn maybe_autocompact(&mut self, store: &crate::store::Store, session_id: i64) {
        // A one-shot run exits after its one turn: there is no next request to
        // protect, so spending a model call on a summary would be pure waste.
        if self.cfg.prompt.is_some() {
            return;
        }
        let Some(percent) = store.get_autocompact() else { return };
        let Some(u) = self.last.lock().ok().and_then(|l| *l) else { return };
        let prompt = u.prompt();
        let window = crate::provider::context_window_for(
            Some(store),
            self.cfg.provider.name(),
            &self.cfg.model,
        );
        if prompt * 100 < window * percent as u64 {
            return;
        }
        ui::warn(&format!(
            "context at {} of ~{} tokens ({percent}% of the window) - compacting",
            crate::stats::num(prompt),
            crate::stats::num(window)
        ));
        self.compact("", store, session_id);
    }

    /// Set, clear or report the autocompact threshold.
    fn autocompact(&mut self, arg: Option<&str>, store: &crate::store::Store) {
        match arg {
            Some("off") => {
                store.set_autocompact(None);
                ui::ok("autocompact off");
            }
            Some(p) => {
                let percent: u8 = p.parse().unwrap_or(0);
                store.set_autocompact(Some(percent));
                ui::ok(&format!(
                    "autocompact set: history compacts on its own at {percent}% of the context window"
                ));
            }
            None => match store.get_autocompact() {
                Some(p) => {
                    let pressure = self
                        .last
                        .lock()
                        .ok()
                        .and_then(|l| *l)
                        .map(|u| u.prompt())
                        .unwrap_or(0);
                    let window = crate::provider::context_window_for(
                        Some(store),
                        self.cfg.provider.name(),
                        &self.cfg.model,
                    );
                    let at = if pressure > 0 {
                        format!(
                            ", last request read {} of ~{} tokens ({}%)",
                            crate::stats::num(pressure),
                            crate::stats::num(window),
                            pressure * 100 / window
                        )
                    } else {
                        ", nothing sent to the model yet".to_string()
                    };
                    ui::ok(&format!("autocompact at {p}% of the context window{at}"));
                }
                None => ui::info(
                    "autocompact off. Set a threshold with /autocompact <1-99>, \
                     e.g. /autocompact 80",
                ),
            },
        }
    }

    /// Replace the transcript with a model-written summary. Tool calls in the
    /// summary request are ignored (text only), so compaction can't recurse.
    fn compact(&mut self, extra: &str, store: &crate::store::Store, session_id: i64) {
        if self.history.is_empty() {
            ui::info("nothing to compact");
            return;
        }
        let n = self.history.len();
        let mut t = String::new();
        for m in &self.history {
            let (role, body) = match m {
                Msg::User(s) => ("USER", s.clone()),
                Msg::Assistant { text, calls } => {
                    let mut b = text.clone();
                    for c in calls {
                        b.push_str(&format!("\n[tool {}: {}]", c.name, c.label()));
                    }
                    ("ASSISTANT", b)
                }
                Msg::ToolResult { output, .. } => ("TOOL", output.clone()),
            };
            t.push_str(&format!("### {role}\n{}\n", shell::truncate(body, 2000)));
        }
        let prompt = format!(
            "Summarize this coding-agent conversation so work can continue from the summary alone. Keep: the goal, key decisions, files touched (paths), command results and errors, current state, next steps.{extra}\n\nTRANSCRIPT:\n{t}",
            extra = if extra.is_empty() { String::new() } else { format!(" Extra instructions: {extra}") },
        );
        let spin = ui::Spinner::start("compacting");
        // Mechanical task: no reasoning budget.
        let mut cfg = self.cfg.clone();
        cfg.thinking = "off".to_string();
        let req = vec![Msg::User(prompt)];
        let mut summary = String::new();
        // Compaction is a text task: no tools, and no skill catalog to read.
        let res = self.client.complete_streaming(&cfg, &req, &[], "", |c| {
            if let Chunk::Text(d) = c {
                summary.push_str(d);
            }
        });
        spin.stop();
        match res {
            Ok(_) if !summary.trim().is_empty() => {
                self.history = vec![Msg::User(format!(
                    "Earlier conversation, compacted from {n} messages:\n{summary}"
                ))];
                // The old context is gone: the pressure figure from before the
                // compact says nothing about what is on screen now, and keeping
                // it would make the next status read like it is still full.
                if let Ok(mut l) = self.last.lock() {
                    *l = None;
                }
                self.log(&serde_json::json!({"role": "compact", "messages": n}));
                store.save_messages(session_id, &self.history);
                ui::ok(&format!("compacted {n} messages into one summary"));
            }
            _ => ui::error("compaction failed, history unchanged"),
        }
    }

    /// Load a past session by id. Returns the id on success.
    fn resume(&mut self, id_str: &str, store: &crate::store::Store, here: &str) -> Option<i64> {
        let id: i64 = match id_str.trim().parse() {
            Ok(i) => i,
            Err(_) => {
                ui::error(&format!("usage: /resume <id>, got {id_str:?}"));
                return None;
            }
        };
        let row = match store.session(id) {
            Some(r) => r,
            None => {
                ui::error(&format!("no session #{id}"));
                return None;
            }
        };
        // An id found by `/resume all` may have happened elsewhere: say so
        // before loading it, so the source of the unfamiliar history is known.
        if !row.cwd.is_empty() && row.cwd != here {
            ui::info(&format!(
                "that session ran in {} - its history is about that project",
                row.cwd
            ));
        }
        let provider = match config::Provider::resolve(&row.provider, store) {
            Some(p) => p,
            None => {
                ui::error(&format!(
                    "session #{id} used provider {}, which is no longer configured",
                    row.provider
                ));
                return None;
            }
        };
        if self.cfg.switch_provider(&provider, store).is_ok() {
            self.cfg.model = row.model.clone();
            persist_session(store, &self.cfg);
        }
        self.history = store.load_messages(id);
        ui::ok(&format!(
            "resumed session #{id} ({} - {} - {} msgs)",
            row.provider,
            self.cfg.model,
            self.history.len()
        ));
        // The history is live either way; without painting it back, a resume
        // looks like it did nothing.
        ui::transcript(&self.history);
        Some(id)
    }

    fn log(&self, v: &serde_json::Value) {
        log_to(&self.cfg.log, v);
    }
}

/// Drop the last `n` user turns (their replies and tool traffic go with
/// them). Returns the number of messages removed. Pure, unit-tested.
fn rewind(history: &mut Vec<Msg>, n: usize) -> usize {
    // ponytail: linear scan, histories are short
    let users: Vec<usize> = history
        .iter()
        .enumerate()
        .filter(|(_, m)| matches!(m, Msg::User(_)))
        .map(|(i, _)| i)
        .collect();
    if users.is_empty() {
        return 0;
    }
    let at = users[users.len() - n.min(users.len())];
    let removed = history.len() - at;
    history.truncate(at);
    removed
}

#[derive(Debug, PartialEq)]
enum LoopEnd {
    Done,
    Cancelled,
}

/// The session's toolset. `Manager` is behind a mutex because a call writes to
/// a server's stdin and reads its reply: two calls to one server at once would
/// interleave their bytes. The tools themselves are therefore called one at a
/// time, whatever the batch's parallelism, while shell commands still overlap.
#[derive(Clone)]
pub struct Tools {
    mcp: Arc<Mutex<crate::mcp::Manager>>,
    /// Cleared on the first turn: the one-time grace that lets fast MCP
    /// servers land before the first request is built.
    first_turn: Arc<AtomicBool>,
}

impl Tools {
    /// Start connecting every configured MCP server. Returns immediately:
    /// servers finish in the background and join `specs()` as they land.
    pub fn connect(cwd: &std::path::Path) -> Self {
        let mcp = crate::mcp::Manager::connect(cwd);
        Tools {
            mcp: Arc::new(Mutex::new(mcp)),
            first_turn: Arc::new(AtomicBool::new(true)),
        }
    }

    /// Just the harness's own tools, without MCP: what tests use.
    #[cfg(test)]
    pub fn builtin_only() -> Self {
        Tools {
            mcp: Arc::new(Mutex::new(crate::mcp::Manager::none())),
            first_turn: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Connection notes added since the last call, including servers that just
    /// landed. The caller prints them at a quiet point.
    pub fn take_notes(&self) -> Vec<String> {
        match self.mcp.lock() {
            Ok(mut m) => m.take_notes(),
            Err(_) => Vec::new(),
        }
    }

    /// The current tool list: the harness's own plus every server connected so
    /// far. Rebuilt per model round, so a server that lands mid-turn joins the
    /// next one.
    pub fn specs(&self) -> Vec<crate::provider::ToolSpec> {
        match self.mcp.lock() {
            Ok(mut m) => m.tool_specs(),
            Err(_) => crate::provider::builtin_tools(),
        }
    }

    /// Wait for every pending server, bounded by `timeout`.
    pub fn wait_ready(&self, timeout: std::time::Duration) {
        if let Ok(mut m) = self.mcp.lock() {
            m.wait_ready(timeout);
        }
    }

    /// The first turn waits briefly for the initial connections; later turns
    /// never do.
    pub fn wait_first_turn(&self, timeout: std::time::Duration) {
        if self.first_turn.swap(false, Ordering::Relaxed) {
            self.wait_ready(timeout);
        }
    }

    /// Run one non-shell call. Returns the text to hand back to the model.
    fn call_mcp(&self, call: &ToolCall) -> String {
        let args = if call.input.is_null() { serde_json::json!({}) } else { call.input.clone() };
        match self.mcp.lock() {
            Ok(mut m) => m.call(&call.name, &args),
            Err(_) => "[tool error] the MCP connection was lost".to_string(),
        }
    }
}

/// Lines typed while a turn is running, waiting for the turn to finish.
pub struct WhileWorking {
    pub typed: Arc<crate::term::Typed>,
}

impl WhileWorking {
    fn new() -> Self {
        Self { typed: crate::term::Typed::new() }
    }
}

/// One model→tools loop over `hist`. Streaming runs on a worker thread and
/// forwards owned chunks over a channel, so the main thread stays responsive
/// to Esc and to typing (the watcher edits a live line, tools just read flags).
type LastUsage = Arc<std::sync::Mutex<Option<crate::stats::Usage>>>;

#[allow(clippy::too_many_arguments)]
fn run_loop(
    cfg: &Config,
    client: &Client,
    hist: &mut Vec<Msg>,
    log: Option<std::path::PathBuf>,
    work: &WhileWorking,
    tools: &Tools,
    context: &str,
    last: &LastUsage,
) -> Result<LoopEnd> {
    let cancel = Arc::new(AtomicBool::new(false));
    let finished = Arc::new(AtomicBool::new(false));
    // Raw stdin for the whole turn: Esc/Ctrl-C and typing arrive as bytes.
    let _raw = crate::term::RawGuard::new();
    let watcher = {
        let (c, d) = (cancel.clone(), finished.clone());
        let t = work.typed.clone();
        std::thread::spawn(move || crate::term::input_watcher(c, d, t))
    };
    let end = drive(cfg, client, hist, &log, &cancel, tools, context, last);
    finished.store(true, Ordering::Relaxed);
    // A console read can still be waiting for a key; wake it so the next
    // prompt opens now instead of on the user's first keystroke.
    crate::term::unblock_stdin(&watcher);
    let _ = watcher.join();
    end
}

enum StreamEvent {
    Text(String),
    Think(String),
    Done(Turn),
    Failed(String),
}

#[allow(clippy::too_many_arguments)]
fn drive(
    cfg: &Config,
    client: &Client,
    hist: &mut Vec<Msg>,
    log: &Option<std::path::PathBuf>,
    cancel: &Arc<AtomicBool>,
    tools: &Tools,
    context: &str,
    last: &LastUsage,
) -> Result<LoopEnd> {
    let mut round = 0usize;
    loop {
        let (tx, rx) = mpsc::channel::<StreamEvent>();
        let cfg2 = cfg.clone();
        let client2 = client.clone();
        let hist2 = hist.clone();
        let worker_cancel = cancel.clone();
        let specs = tools.specs();
        let context = context.to_string();
        std::thread::spawn(move || {
            let res = client2.complete_streaming(&cfg2, &hist2, &specs, &context, |c| {
                if worker_cancel.load(Ordering::Relaxed) {
                    return;
                }
                let e = match c {
                    Chunk::Text(d) => StreamEvent::Text(d.to_string()),
                    Chunk::Think(d) => StreamEvent::Think(d.to_string()),
                };
                let _ = tx.send(e);
            });
            match res {
                Ok(t) => {
                    let _ = tx.send(StreamEvent::Done(t));
                }
                Err(e) => {
                    let _ = tx.send(StreamEvent::Failed(format!("{e:#}")));
                }
            }
        });

        let label = if round == 0 {
            format!("waiting on {}", cfg.model)
        } else {
            "reading tool output".to_string()
        };
        round += 1;
        let mut spin = Some(ui::Spinner::start(&label));
        let mut md = ui::Md::new();
        let mut saw_delta = false;
        let mut streamed = false;
        let mut think_open = false;
        let turn: Turn = loop {
            if cancel.load(Ordering::Relaxed) {
                if let Some(s) = spin.take() {
                    s.stop();
                }
                ui::stream_end();
                // The caller reports the cancel once, with the turn's words.
                return Ok(LoopEnd::Cancelled);
            }
            match rx.recv_timeout(Duration::from_millis(20)) {
                Ok(StreamEvent::Text(d)) => {
                    // The indicator stays; only its word changes. Stopping it
                    // here is what used to take the prompt row off the screen.
                    if let Some(s) = spin.as_mut() {
                        s.relabel("writing");
                    }
                    if think_open {
                        think_open = false;
                        ui::stream_end();
                    }
                    saw_delta = true;
                    let r = md.push(&d);
                    if !r.is_empty() {
                        streamed = true;
                        ui::stream_text(&r);
                    }
                }
                Ok(StreamEvent::Think(d)) => {
                    if let Some(s) = spin.as_mut() {
                        s.relabel("thinking");
                    }
                    if !think_open {
                        think_open = true;
                        ui::think_start();
                    }
                    ui::stream_think(&d);
                }
                Ok(StreamEvent::Done(t)) => {
                    if let Some(s) = spin.take() {
                        s.stop();
                    }
                    break t;
                }
                Ok(StreamEvent::Failed(e)) => {
                    if let Some(s) = spin.take() {
                        s.stop();
                    }
                    // Drop the unanswered user message so history stays valid.
                    if matches!(hist.last(), Some(Msg::User(_))) {
                        hist.pop();
                    }
                    return Err(anyhow::anyhow!("{e}"));
                }
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    if let Some(s) = spin.take() {
                        s.stop();
                    }
                    return Err(anyhow::anyhow!("stream thread died"));
                }
            }
        };

        if !saw_delta && !turn.text.is_empty() {
            // Provider ignored streaming: render the whole body at once.
            // (When deltas arrived, `md` already holds them: no re-push.)
            let r = md.push(&turn.text);
            if !r.is_empty() {
                streamed = true;
                ui::stream_text(&r);
            }
        }
        let tail = md.flush();
        if !tail.is_empty() {
            streamed = true;
            ui::stream_text(&tail);
        }
        if streamed || think_open {
            ui::stream_end();
        }
        let Turn { text, thinking, calls, usage } = turn;
        // What this request was charged to read is how full the context now
        // is: the newest fact about the transcript, kept for autocompact.
        if let Ok(mut l) = last.lock() {
            *l = (usage.prompt() > 0).then_some(usage);
        }
        log_to(
            log,
            &serde_json::json!({"role": "assistant", "text": text, "thinking": thinking, "calls": calls.iter().map(|c| c.label()).collect::<Vec<_>>() }),
        );
        hist.push(Msg::Assistant { text, calls: calls.clone() });

        if calls.is_empty() {
            return Ok(LoopEnd::Done);
        }

        // Every call in the round gets a result, in order, even a round cut
        // short by Esc: a history with a call and no result is one the next
        // request would be rejected for.
        let outputs = execute_batch(cfg, &calls, cancel, tools);
        let cancelled = cancel.load(Ordering::Relaxed);
        for (call, output) in calls.into_iter().zip(outputs) {
            log_to(
                log,
                &serde_json::json!({"role": "tool", "tool": call.name, "command": call.label(), "output": output}),
            );
            hist.push(Msg::ToolResult { id: call.id, output });
        }
        if cancelled {
            return Ok(LoopEnd::Cancelled);
        }
    }
}

/// Run one reply's commands. A reply is the model's own grouping, so what sits
/// in it is meant to be independent: every command in it starts at once, however
/// many that is, because how much to run in parallel is the model's call. A
/// `--jobs` throttle can hold some back, for a machine that needs it.
/// Results are keyed back to their call, printed in call order as soon as each
/// is known, so a fast command never lands under a slow one's heading.
fn execute_batch(cfg: &Config, calls: &[ToolCall], cancel: &AtomicBool, tools: &Tools) -> Vec<String> {
    if calls.len() == 1 {
        return vec![execute(cfg, &calls[0], cancel, tools)];
    }
    // The echo waits for the result: the dot is the outcome, so it cannot be
    // drawn before there is one. The spinner carries the wait's identity.
    let label = format!("running {} commands", calls.len());
    let spin = ui::Spinner::start(&label);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let (tx, rx) = std::sync::mpsc::channel::<(usize, String)>();
    let mut got: Vec<Option<String>> = calls.iter().map(|_| None).collect();
    let workers = if cfg.jobs == 0 {
        calls.len()
    } else {
        cfg.jobs.min(calls.len())
    };
    std::thread::scope(|scope| {
        for _ in 0..workers.max(1) {
            let tx = tx.clone();
            let next = &next;
            scope.spawn(move || loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= calls.len() {
                    return;
                }
                if cancel.load(Ordering::Relaxed) {
                    let _ = tx.send((i, "[stopped by user]".to_string()));
                    continue;
                }
                // Only shell commands run here: an MCP call would need the
                // manager's lock, and holding it across a worker would put the
                // batch back in single file. Those are run after the scope.
                if calls[i].name != "shell" {
                    continue;
                }
                let cmd = calls[i].command().unwrap_or_default();
                let out = shell::run(cmd, &cfg.cwd, cfg.timeout, cfg.max_output, || {
                    cancel.load(Ordering::Relaxed)
                });
                let _ = tx.send((i, out.text));
            });
        }
        drop(tx);
        let mut printed = 0usize;
        for (i, text) in rx.iter() {
            got[i] = Some(text);
            while printed < got.len() {
                // Print what is ready, but keep it: these strings are also what
                // the model is told the command did.
                match got[printed].as_ref() {
                    Some(t) => {
                        ui::command(&calls[printed].label(), crate::shell::exit_ok(t));
                        ui::output(t);
                        printed += 1;
                    }
                    None => break,
                }
            }
        }
    });
    // Server tools, in call order, one at a time: each takes the manager lock.
    for (i, call) in calls.iter().enumerate() {
        if call.name == "shell" {
            continue;
        }
        got[i] = Some(execute(cfg, call, cancel, tools));
    }
    spin.stop();
    // A batch that mixed the two kinds has printed its shell output already, so
    // the server results are echoed here in the order they ran.
    for (i, text) in got.iter().enumerate() {
        if calls[i].name == "shell" {
            continue;
        }
        let text = text.as_deref().unwrap_or_default();
        ui::command(&calls[i].label(), !text.starts_with("[tool error]"));
        ui::output(text);
    }
    got.into_iter().map(|o| o.unwrap_or_default()).collect()
}

/// Run one call: a shell command through the runner, or a server tool through
/// its MCP connection. Either way the echo is printed with the outcome, and the
/// text that comes back is exactly what the model is handed.
fn execute(cfg: &Config, call: &ToolCall, cancel: &AtomicBool, tools: &Tools) -> String {
    // A call can outlast everything else here; the wait gets a spinner, so a
    // slow one never looks like a hang and the echo is not printed blind.
    let spin = ui::Spinner::start("running");
    let (text, ok) = match call.command() {
        Some(command) => {
            let out = shell::run(command, &cfg.cwd, cfg.timeout, cfg.max_output, || {
                cancel.load(Ordering::Relaxed)
            });
            let ok = shell::exit_ok(&out.text);
            (out.text, ok)
        }
        None => {
            let text = match call.name.as_str() {
                // The skill tool is the harness's own: it reads a file and
                // hands the model its instructions, which is exactly what a
                // deferred skill body is for.
                "skill" => {
                    let name = call.input.get("name").and_then(|n| n.as_str()).unwrap_or("");
                    match crate::skills::load(&cfg.cwd, name) {
                        Ok(body) => body,
                        Err(e) => format!("[tool error] {e:#}"),
                    }
                }
                _ => tools.call_mcp(call),
            };
            let ok = !text.starts_with("[tool error]");
            (text, ok)
        }
    };
    spin.stop();
    // Echo now, with the outcome in the dot, then the output under it.
    ui::command(&call.label(), ok);
    // A cancel is reported once, by the turn: the runner's `[stopped by user]`
    // marker in the output already says the command itself was cut short.
    ui::output(&text);
    text
}

fn log_to(path: &Option<std::path::PathBuf>, v: &serde_json::Value) {
    let Some(path) = path else { return };
    let line = serde_json::json!({"ts": ts(), "data": v}).to_string();
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(f, "{line}");
    }
}

fn ts() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msgs(n_user: usize) -> Vec<Msg> {
        let mut h = Vec::new();
        for i in 0..n_user {
            h.push(Msg::User(format!("q{i}")));
            h.push(Msg::Assistant { text: format!("a{i}"), calls: vec![] });
        }
        h
    }

    #[test]
    fn short_dir_keeps_a_resume_all_row_on_one_line() {
        let sep = std::path::MAIN_SEPARATOR;
        assert_eq!(
            short_dir(&format!("{sep}media{sep}atlas1{sep}RIFT")),
            format!("atlas1{sep}RIFT")
        );
        assert_eq!(short_dir(&format!("{sep}alpha")), format!("{sep}alpha"));
        assert_eq!(short_dir(""), "");
        #[cfg(windows)]
        assert_eq!(short_dir(r"C:\Users\rift\proj"), r"rift\proj");
    }

    #[test]
    fn a_flag_overrides_a_run_without_rewriting_what_is_stored() {
        use clap::Parser;
        let store = crate::store::Store::memory();
        store.save_provider("openai", "real-model", "https://real.example/v1", "sk-real");

        let cli = crate::config::Cli::parse_from([
            "rift",
            "--provider",
            "openai",
            "--base-url",
            "http://127.0.0.1:9/v1",
            "--api-key",
            "throwaway",
        ]);
        let cfg = Config::from_cli(cli, &store, true).unwrap();
        // The run uses what it was told.
        assert_eq!(cfg.base_url, "http://127.0.0.1:9/v1");
        assert_eq!(cfg.api_key, "throwaway");
        persist_session(&store, &cfg);
        // The store keeps what it had.
        let saved = store.provider_state("openai").unwrap();
        assert_eq!(saved.base_url, "https://real.example/v1", "a test run repointed the provider");
        assert_eq!(saved.api_key, "sk-real");
        assert_eq!(saved.model, "real-model");

        // A provider with nothing stored still gets configured by its flag.
        let cli = crate::config::Cli::parse_from(["rift", "--provider", "openrouter", "--api-key", "fresh"]);
        let cfg = Config::from_cli(cli, &store, true).unwrap();
        persist_session(&store, &cfg);
        assert_eq!(store.provider_state("openrouter").unwrap().api_key, "fresh");
    }

    /// A command that pauses then prints `n`, in the platform shell's syntax.
    #[cfg(unix)]
    fn pause_then_echo(n: &str) -> String {
        format!("sleep 0.4; echo {n}")
    }

    #[cfg(windows)]
    fn pause_then_echo(n: &str) -> String {
        format!("Start-Sleep -Milliseconds 400; Write-Output {n}")
    }

    /// The real shell this test needs. A bare wine prefix answers `powershell`
    /// with a stub that runs nothing, so ask it to actually print something.
    fn shell_available() -> bool {
        let mut cmd = std::process::Command::new(crate::shell::SHELL);
        #[cfg(windows)]
        cmd.args(["-NoProfile", "-Command", "Write-Output rift-shell-ok"]);
        #[cfg(unix)]
        cmd.args(["-c", "echo rift-shell-ok"]);
        cmd.stdin(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .output()
            .map(|o| {
                o.status.success() && String::from_utf8_lossy(&o.stdout).contains("rift-shell-ok")
            })
            .unwrap_or(false)
    }

    #[test]
    fn a_batch_shares_the_wait_and_keeps_call_order() {
        use crate::provider::ToolCall;
        use clap::Parser;
        if !shell_available() {
            return;
        }
        let store = crate::store::Store::memory();
        let cli = crate::config::Cli::parse_from(["rift", "--provider", "openai", "--api-key", "k"]);
        let cfg = Config::from_cli(cli, &store, true).unwrap();
        assert_eq!(cfg.jobs, 0, "nothing caps the batch unless the user asks");
        let calls: Vec<ToolCall> = ["a", "b", "c"]
            .iter()
            .map(|n| ToolCall {
                id: (*n).to_string(),
                name: "shell".to_string(),
                input: serde_json::json!({"command": pause_then_echo(n)}),
            })
            .collect();

        let started = std::time::Instant::now();
        let out = execute_batch(&cfg, &calls, &AtomicBool::new(false), &Tools::builtin_only());
        let took = started.elapsed();

        assert_eq!(out.len(), 3);
        for (i, name) in ["a", "b", "c"].iter().enumerate() {
            assert!(out[i].contains(name), "result {i} is not {name}: {:?}", out[i]);
        }
        // Three 0.4s commands one after another take 1.2s; together, about
        // 0.4s, plus a shell startup each on Windows.
        let ceiling = if cfg!(windows) { 1600 } else { 1100 };
        assert!(
            took < std::time::Duration::from_millis(ceiling),
            "batch did not overlap: {took:?}"
        );
    }

    #[test]
    fn rewind_drops_last_user_turn_by_default() {
        let mut h = msgs(3);
        assert_eq!(rewind(&mut h, 1), 2);
        assert_eq!(h.len(), 4);
        assert!(matches!(&h[3], Msg::Assistant { text, .. } if text == "a1"));
    }

    #[test]
    fn rewind_two_turns_and_clamp() {
        let mut h = msgs(3);
        assert_eq!(rewind(&mut h, 2), 4);
        assert_eq!(h.len(), 2);
        let mut h = msgs(2);
        assert_eq!(rewind(&mut h, 99), 4);
        assert!(h.is_empty());
    }

    #[test]
    fn rewind_empty_is_zero() {
        let mut h: Vec<Msg> = Vec::new();
        assert_eq!(rewind(&mut h, 1), 0);
    }
}
