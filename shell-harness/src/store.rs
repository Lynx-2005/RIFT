//! On-disk session store (SQLite): active provider, per-provider model/URL/key,
//! and a cached model catalog per provider. Everything degrades to in-memory
//! when the DB cannot be opened, so the app never fails to start.
use anyhow::{Context, Result};
use rusqlite::{params, Connection};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

pub const MODELS_TTL_SECS: u64 = 24 * 3600;

pub struct ProviderState {
    pub model: String,
    pub base_url: String,
    pub api_key: String,
}

pub struct Store {
    conn: Connection,
}

/// Unix keeps XDG's convention; Windows keeps application data where the
/// platform expects it.
#[cfg(not(windows))]
fn base_dir() -> PathBuf {
    if let Ok(x) = std::env::var("XDG_DATA_HOME") {
        if !x.is_empty() {
            return PathBuf::from(x);
        }
    }
    crate::config::home_dir()
        .map(|h| h.join(".local").join("share"))
        .unwrap_or_else(|| PathBuf::from("."))
}

#[cfg(windows)]
fn base_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("LOCALAPPDATA") {
        if !dir.is_empty() {
            return PathBuf::from(dir);
        }
    }
    crate::config::home_dir()
        .map(|h| h.join("AppData").join("Local"))
        .unwrap_or_else(|| PathBuf::from("."))
}

pub fn data_dir() -> PathBuf {
    base_dir().join("rift")
}

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

pub struct SessionRow {
    pub id: i64,
    pub created_at: u64,
    pub provider: String,
    pub model: String,
    pub title: String,
    /// The directory the session ran in: '' marks a one-shot run.
    pub cwd: String,
    pub count: i64,
}

impl Store {
    fn init(conn: &Connection) -> Result<()> {
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS providers(
               name TEXT PRIMARY KEY, model TEXT NOT NULL,
               base_url TEXT NOT NULL, api_key TEXT NOT NULL DEFAULT '');
             CREATE TABLE IF NOT EXISTS session(
               key TEXT PRIMARY KEY, value TEXT NOT NULL);
             CREATE TABLE IF NOT EXISTS models(
               provider TEXT PRIMARY KEY, ids TEXT NOT NULL, fetched_at INTEGER NOT NULL);
             CREATE TABLE IF NOT EXISTS sessions(
               id INTEGER PRIMARY KEY AUTOINCREMENT, created_at INTEGER NOT NULL,
               provider TEXT NOT NULL, model TEXT NOT NULL, title TEXT NOT NULL DEFAULT '',
               cwd TEXT NOT NULL DEFAULT '');
             CREATE TABLE IF NOT EXISTS messages(
               session_id INTEGER NOT NULL, idx INTEGER NOT NULL,
               json TEXT NOT NULL, PRIMARY KEY(session_id, idx));",
        )?;
        // Databases written before sessions remembered a directory: the column
        // is added in place, and older rows keep '' so they belong to no
        // directory and cannot be picked up as one.
        let _ = conn.execute("ALTER TABLE sessions ADD COLUMN cwd TEXT NOT NULL DEFAULT ''", []);
        Ok(())
    }

    pub fn open() -> Result<Self> {
        let dir = data_dir();
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("creating {}", dir.display()))?;
        let conn = Connection::open(dir.join("rift.db"))?;
        // A background refresh thread can write while the loop saves.
        conn.busy_timeout(std::time::Duration::from_secs(5))?;
        Self::init(&conn)?;
        Ok(Self { conn })
    }

    /// In-memory store: used when the DB cannot be opened, and in tests.
    pub fn memory() -> Self {
        let conn = Connection::open_in_memory().expect("memory db");
        Self::init(&conn).expect("memory schema");
        Self { conn }
    }

    pub fn provider_state(&self, name: &str) -> Option<ProviderState> {
        self.conn
            .query_row(
                "SELECT model, base_url, api_key FROM providers WHERE name = ?1",
                params![name],
                |r| {
                    Ok(ProviderState {
                        model: r.get(0)?,
                        base_url: r.get(1)?,
                        api_key: r.get(2)?,
                    })
                },
            )
            .ok()
    }

    pub fn save_provider(&self, name: &str, model: &str, url: &str, key: &str) {
        let _ = self.conn.execute(
            "INSERT INTO providers(name, model, base_url, api_key) VALUES(?1,?2,?3,?4)
             ON CONFLICT(name) DO UPDATE SET model=excluded.model, base_url=excluded.base_url, api_key=excluded.api_key",
            params![name, model, url, key],
        );
    }

    pub fn active_provider(&self) -> Option<String> {
        self.conn
            .query_row("SELECT value FROM session WHERE key = 'active'", [], |r| {
                r.get(0)
            })
            .ok()
    }

    pub fn set_active(&self, name: &str) {
        let _ = self.conn.execute(
            "INSERT INTO session(key, value) VALUES('active', ?1)
             ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            params![name],
        );
    }

    /// Cached catalog + age. `None` means never fetched.
    pub fn cached_models(&self, provider: &str) -> Option<(Vec<String>, u64)> {
        let (ids, at): (String, u64) = self
            .conn
            .query_row("SELECT ids, fetched_at FROM models WHERE provider = ?1", params![provider], |r| {
                Ok((r.get(0)?, r.get(1)?))
            })
            .ok()?;
        let list: Vec<String> = ids.split('\n').filter(|s| !s.is_empty()).map(str::to_string).collect();
        Some((list, at))
    }

    pub fn save_models(&self, provider: &str, ids: &[String]) {
        let joined = ids.join("\n");
        let _ = self.conn.execute(
            "INSERT INTO models(provider, ids, fetched_at) VALUES(?1,?2,?3)
             ON CONFLICT(provider) DO UPDATE SET ids=excluded.ids, fetched_at=excluded.fetched_at",
            params![provider, joined, now_secs() as i64],
        );
    }

    pub fn models_fresh(&self, provider: &str) -> bool {
        match self.cached_models(provider) {
            Some((ids, at)) => !ids.is_empty() && now_secs().saturating_sub(at) < MODELS_TTL_SECS,
            None => false,
        }
    }

    /// Autocompact threshold as a percent of the context window (1..=99),
    /// or `None` when autocompact is off. Same KV table as thinking: no
    /// schema migration on databases an older build wrote.
    pub fn get_autocompact(&self) -> Option<u8> {
        let v: String = self
            .conn
            .query_row("SELECT value FROM session WHERE key = 'autocompact'", [], |r| r.get(0))
            .ok()?;
        v.parse().ok()
    }

    pub fn set_autocompact(&self, percent: Option<u8>) {
        let _ = self.conn.execute(
            "INSERT INTO session(key, value) VALUES('autocompact', ?1)
             ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            params![percent.map(|p| p.to_string()).unwrap_or_else(|| "off".to_string())],
        );
    }

    /// Thinking level per provider (`off|minimal|low|medium|high|xhigh|max`).
    /// KV in the existing `session` table: no schema migration on old DBs.
    pub fn get_thinking(&self, provider: &str) -> Option<String> {
        self.conn
            .query_row(
                "SELECT value FROM session WHERE key = 'thinking:' || ?1",
                params![provider],
                |r| r.get(0),
            )
            .ok()
    }

    pub fn set_thinking(&self, provider: &str, level: &str) {
        let _ = self.conn.execute(
            "INSERT INTO session(key, value) VALUES('thinking:' || ?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            params![provider, level],
        );
    }
}

impl Store {
    /// A new conversation. `cwd` ties it to a directory, so opening the agent
    /// there again continues it. One-shot runs pass "" and stay unlisted.
    pub fn create_session(&self, provider: &str, model: &str, title: &str, cwd: &str) -> i64 {
        let _ = self.conn.execute(
            "INSERT INTO sessions(created_at, provider, model, title, cwd) VALUES(?1,?2,?3,?4,?5)",
            params![now_secs() as i64, provider, model, title, cwd],
        );
        self.conn.last_insert_rowid()
    }

    /// Sessions newest first. `cwd = Some(dir)` keeps only that directory's:
    /// a bare `/resume` asks about here, `/resume all` passes `None`. A
    /// directory with no history lists nothing rather than borrowing another
    /// project's conversations.
    pub fn list_sessions(&self, limit: usize, cwd: Option<&str>) -> Vec<SessionRow> {
        let mut stmt = match self.conn.prepare(
            "SELECT id, created_at, provider, model, title, cwd,
                    (SELECT COUNT(*) FROM messages WHERE session_id = sessions.id)
             FROM sessions WHERE (?1 IS NULL OR cwd = ?1)
             ORDER BY id DESC LIMIT ?2",
        ) {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };
        stmt.query_map(params![cwd, limit as i64], |r| {
            Ok(SessionRow {
                id: r.get(0)?,
                created_at: r.get(1).unwrap_or(0),
                provider: r.get(2)?,
                model: r.get(3)?,
                title: r.get(4)?,
                cwd: r.get(5).unwrap_or_default(),
                count: r.get(6).unwrap_or(0),
            })
        })
        .map(|rows| rows.filter_map(|r| r.ok()).collect())
        .unwrap_or_default()
    }

    pub fn session(&self, id: i64) -> Option<SessionRow> {
        self.conn
            .query_row(
                "SELECT id, created_at, provider, model, title, cwd,
                        (SELECT COUNT(*) FROM messages WHERE session_id = ?1)
                 FROM sessions WHERE id = ?1",
                params![id],
                |r| {
                    Ok(SessionRow {
                        id: r.get(0)?,
                        created_at: r.get(1).unwrap_or(0),
                        provider: r.get(2)?,
                        model: r.get(3)?,
                        title: r.get(4)?,
                        cwd: r.get(5).unwrap_or_default(),
                        count: r.get(6).unwrap_or(0),
                    })
                },
            )
            .ok()
    }

    pub fn touch_session(&self, id: i64, provider: &str, model: &str) {
        let _ = self.conn.execute(
            "UPDATE sessions SET provider=?1, model=?2 WHERE id=?3",
            params![provider, model, id],
        );
    }

    pub fn set_title(&self, id: i64, title: &str) {
        let _ = self.conn.execute(
            "UPDATE sessions SET title=?1 WHERE id=?2 AND title=''",
            params![title, id],
        );
    }

    /// Replace the stored transcript with `history` (small histories: simplest sync).
    pub fn save_messages(&self, session_id: i64, history: &[crate::provider::Msg]) {
        let _ = self.conn.execute(
            "DELETE FROM messages WHERE session_id = ?1",
            params![session_id],
        );
        for (i, m) in history.iter().enumerate() {
            let _ = self.conn.execute(
                "INSERT INTO messages(session_id, idx, json) VALUES(?1,?2,?3)",
                params![session_id, i as i64, crate::provider::msg_json(m).to_string()],
            );
        }
    }

    pub fn load_messages(&self, session_id: i64) -> Vec<crate::provider::Msg> {
        let mut stmt = match self
            .conn
            .prepare("SELECT json FROM messages WHERE session_id = ?1 ORDER BY idx")
        {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };
        stmt.query_map(params![session_id], |r| r.get::<_, String>(0))
            .map(|rows| {
                rows.filter_map(|r| r.ok())
                    .filter_map(|s| {
                        serde_json::from_str::<serde_json::Value>(&s)
                            .ok()
                            .and_then(|v| crate::provider::msg_from_json(&v))
                    })
                    .collect()
            })
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_roundtrip() {
        let s = Store::memory();
        assert!(s.provider_state("openrouter").is_none());
        assert!(s.active_provider().is_none());
        s.save_provider("openrouter", "x-ai/grok-4", "https://openrouter.ai/api/v1", "k1");
        s.set_active("openrouter");
        let st = s.provider_state("openrouter").unwrap();
        assert_eq!((st.model, st.base_url, st.api_key), ("x-ai/grok-4".into(), "https://openrouter.ai/api/v1".into(), "k1".into()));
        assert_eq!(s.active_provider().as_deref(), Some("openrouter"));
        s.save_provider("openrouter", "openai/gpt-4.1", "https://openrouter.ai/api/v1", "k2");
        assert_eq!(s.provider_state("openrouter").unwrap().api_key, "k2");
    }

    #[test]
    fn models_cache_roundtrip_and_freshness() {
        let s = Store::memory();
        assert!(!s.models_fresh("openai"));
        assert!(s.cached_models("openai").is_none());
        s.save_models("openai", &["gpt-4.1".to_string(), "gpt-4o".to_string()]);
        assert!(s.models_fresh("openai"));
        assert_eq!(s.cached_models("openai").unwrap().0, vec!["gpt-4.1", "gpt-4o"]);
    }

    #[test]
    fn thinking_level_roundtrip() {
        let s = Store::memory();
        assert!(s.get_thinking("openrouter").is_none());
        s.set_thinking("openrouter", "high");
        assert_eq!(s.get_thinking("openrouter").as_deref(), Some("high"));
        s.set_thinking("openrouter", "off");
        assert_eq!(s.get_thinking("openrouter").as_deref(), Some("off"));
    }

    #[test]
    fn autocompact_threshold_roundtrips_and_turns_off() {
        let s = Store::memory();
        assert!(s.get_autocompact().is_none());
        s.set_autocompact(Some(80));
        assert_eq!(s.get_autocompact(), Some(80));
        s.set_autocompact(Some(45));
        assert_eq!(s.get_autocompact(), Some(45));
        s.set_autocompact(None);
        assert!(s.get_autocompact().is_none(), "off must clear the threshold");
    }

    #[test]
    fn a_directory_remembers_only_its_own_conversation() {
        let s = Store::memory();
        let alpha = s.create_session("anthropic", "m", "", "/work/alpha");
        let beta = s.create_session("anthropic", "m", "", "/work/beta");
        // A directory's list holds its own newest conversation and nothing
        // from anywhere else.
        assert_eq!(s.list_sessions(1, Some("/work/alpha"))[0].id, alpha);
        assert_eq!(s.list_sessions(1, Some("/work/beta"))[0].id, beta);
        // A directory nothing has happened in lists nothing rather than
        // borrowing another project's chat.
        assert!(s.list_sessions(1, Some("/work/gamma")).is_empty());

        // A newer conversation in the same directory is the one to continue.
        let again = s.create_session("anthropic", "m", "", "/work/alpha");
        assert_eq!(s.list_sessions(1, Some("/work/alpha"))[0].id, again);

        // One-shot runs are unlisted, and cannot shadow what is here.
        s.create_session("anthropic", "m", "", "");
        assert_eq!(s.list_sessions(1, Some("/work/beta"))[0].id, beta);
        let _ = beta;
    }

    #[test]
    fn the_session_list_scopes_to_a_directory_when_asked() {
        let s = Store::memory();
        let alpha = s.create_session("anthropic", "m", "", "/work/alpha");
        let beta = s.create_session("anthropic", "m", "", "/work/beta");
        // A bare /resume sees only the directory it stands in...
        assert_eq!(s.list_sessions(10, Some("/work/alpha")).len(), 1);
        let scoped = s.list_sessions(10, Some("/work/beta"));
        assert_eq!(scoped[0].id, beta);
        // ...an empty directory lists nothing rather than borrowing...
        assert!(s.list_sessions(10, Some("/work/other")).is_empty());
        // ...and /resume all crosses the fences, with each row's directory.
        let all = s.list_sessions(10, None);
        assert_eq!(all.len(), 2);
        assert_eq!(all.iter().map(|r| r.cwd.as_str()).collect::<Vec<_>>(),
                   vec!["/work/beta", "/work/alpha"]);
        let _ = alpha;
    }

    #[test]
    fn a_database_from_before_directories_gains_the_column() {
        // The schema an older build wrote, with this build's init on top of it:
        // existing rows belong to no directory, so none of them get inherited.
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE sessions(
               id INTEGER PRIMARY KEY AUTOINCREMENT, created_at INTEGER NOT NULL,
               provider TEXT NOT NULL, model TEXT NOT NULL, title TEXT NOT NULL DEFAULT '');",
        )
        .unwrap();
        conn.execute(
            "INSERT INTO sessions(created_at, provider, model, title) VALUES(1,'openai','m','old')",
            [],
        )
        .unwrap();
        Store::init(&conn).unwrap();
        let s = Store { conn };
        // Rows from before directories belong to none, so they surface only
        // under /resume all, never as this directory's conversation.
        assert!(s.list_sessions(10, Some("/work/alpha")).is_empty());
        let id = s.create_session("openai", "m", "", "/work/alpha");
        assert_eq!(s.list_sessions(1, Some("/work/alpha"))[0].id, id);
        assert_eq!(s.list_sessions(10, None).len(), 2, "the old row went missing");
    }

    #[test]
    fn sessions_list_load_and_messages() {
        use crate::provider::{Msg, ToolCall};
        let s = Store::memory();
        assert!(s.list_sessions(10, None).is_empty());
        let id = s.create_session("openrouter", "x-ai/grok-4", "", "/work/one");
        s.set_title(id, "first title");
        s.set_title(id, "ignored second"); // first title wins
        let hist = vec![
            Msg::User("hi".into()),
            Msg::Assistant {
                text: "yo".into(),
                calls: vec![ToolCall::shell("c1", "ls")],
            },
            Msg::ToolResult { id: "c1".into(), output: "ok".into() },
        ];
        s.save_messages(id, &hist);
        let rows = s.list_sessions(10, None);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].count, 3);
        assert_eq!(rows[0].title, "first title");
        let back = s.load_messages(id);
        assert_eq!(back.len(), 3);
        assert!(matches!(&back[0], Msg::User(t) if t == "hi"));
        assert!(matches!(&back[1], Msg::Assistant { text, calls } if text == "yo" && calls[0].command() == Some("ls")));
        assert!(matches!(&back[2], Msg::ToolResult { output, .. } if output == "ok"));
        // Resync replaces, it does not append.
        s.save_messages(id, &hist[..1]);
        assert_eq!(s.load_messages(id).len(), 1);
    }
}
