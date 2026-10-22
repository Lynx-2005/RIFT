//! On-disk session store (SQLite): active provider, per-provider model/URL/key,
//! and a cached model catalog per provider. Everything degrades to in-memory
//! when the DB cannot be opened, so the app never fails to start.
use anyhow::{Context, Result};
use rusqlite::{params, Connection};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

pub const MODELS_TTL_SECS: u64 = 24 * 3600;

#[derive(Clone)]
pub struct ProviderState {
    pub model: String,
    pub base_url: String,
    pub api_key: String,
    /// The wire shape of a custom provider (`openai` or `anthropic`). Empty
    /// for rows written by builds that only knew the four built-ins.
    pub kind: String,
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
               base_url TEXT NOT NULL, api_key TEXT NOT NULL DEFAULT '',
               kind TEXT NOT NULL DEFAULT '');
             CREATE TABLE IF NOT EXISTS session(
               key TEXT PRIMARY KEY, value TEXT NOT NULL);
             CREATE TABLE IF NOT EXISTS models(
               provider TEXT PRIMARY KEY, ids TEXT NOT NULL, fetched_at INTEGER NOT NULL);
             CREATE TABLE IF NOT EXISTS contexts(
               provider TEXT NOT NULL, model TEXT NOT NULL, context INTEGER NOT NULL,
               PRIMARY KEY(provider, model));
             CREATE TABLE IF NOT EXISTS model_thinking(
               provider TEXT NOT NULL, model TEXT NOT NULL, spec TEXT NOT NULL,
               PRIMARY KEY(provider, model));
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
        // Same story for custom providers: an older `providers` table has no
        // kind, and its rows are all built-ins, so '' is the right value.
        let _ = conn.execute("ALTER TABLE providers ADD COLUMN kind TEXT NOT NULL DEFAULT ''", []);
        Ok(())
    }

    pub fn open() -> Result<Self> {
        let dir = data_dir();
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("creating {}", dir.display()))?;
        let conn = Connection::open(dir.join("rift.db"))?;
        // A background refresh thread can write while the loop saves. WAL lets
        // the two coexist; the timeout covers the brief writer handoff.
        conn.busy_timeout(std::time::Duration::from_secs(10))?;
        let _ = conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;");
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
                "SELECT model, base_url, api_key, kind FROM providers WHERE name = ?1",
                params![name],
                |r| {
                    Ok(ProviderState {
                        model: r.get(0)?,
                        base_url: r.get(1)?,
                        api_key: r.get(2)?,
                        kind: r.get(3).unwrap_or_default(),
                    })
                },
            )
            .ok()
    }

    /// Every provider row, custom ones included, sorted by name.
    pub fn provider_rows(&self) -> Vec<(String, ProviderState)> {
        let Ok(mut stmt) = self
            .conn
            .prepare("SELECT name, model, base_url, api_key, kind FROM providers ORDER BY name")
        else {
            return Vec::new();
        };
        stmt.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                ProviderState {
                    model: r.get(1)?,
                    base_url: r.get(2)?,
                    api_key: r.get(3)?,
                    kind: r.get(4).unwrap_or_default(),
                },
            ))
        })
        .map(|rows| rows.filter_map(|r| r.ok()).collect())
        .unwrap_or_default()
    }

    /// Save a built-in's connection details. `kind` is deliberately not part
    /// of the update: a custom row saved here keeps its wire shape.
    pub fn save_provider(&self, name: &str, model: &str, url: &str, key: &str) {
        let _ = self.conn.execute(
            "INSERT INTO providers(name, model, base_url, api_key) VALUES(?1,?2,?3,?4)
             ON CONFLICT(name) DO UPDATE SET model=excluded.model, base_url=excluded.base_url, api_key=excluded.api_key",
            params![name, model, url, key],
        );
    }

    /// Create or update a custom provider, storing the wire shape it speaks.
    /// Custom providers now come from `providers.json`; this remains for tests
    /// that need to simulate a database row written by an older build.
    #[cfg(test)]
    pub fn save_custom_provider(
        &self,
        name: &str,
        kind: &str,
        model: &str,
        url: &str,
        key: &str,
    ) {
        let _ = self.conn.execute(
            "INSERT INTO providers(name, model, base_url, api_key, kind) VALUES(?1,?2,?3,?4,?5)
             ON CONFLICT(name) DO UPDATE SET model=excluded.model, base_url=excluded.base_url,
                                            api_key=excluded.api_key, kind=excluded.kind",
            params![name, model, url, key, kind],
        );
    }

    /// Forget a provider: its row, catalog, context windows, remembered model
    /// and level.
    pub fn remove_provider(&self, name: &str) {
        for sql in [
            "DELETE FROM providers WHERE name = ?1",
            "DELETE FROM models WHERE provider = ?1",
            "DELETE FROM contexts WHERE provider = ?1",
            "DELETE FROM model_thinking WHERE provider = ?1",
            "DELETE FROM session WHERE key = 'thinking:' || ?1",
            "DELETE FROM session WHERE key LIKE 'thinking:' || ?1 || ':%'",
            "DELETE FROM session WHERE key = 'model:' || ?1",
        ] {
            let _ = self.conn.execute(sql, params![name]);
        }
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

    /// Forget a provider's cached catalog. A list fetched from an endpoint the
    /// provider no longer points at must not keep showing its models.
    pub fn clear_models(&self, provider: &str) {
        let _ = self.conn.execute("DELETE FROM models WHERE provider = ?1", params![provider]);
    }

    /// Context windows learned for a provider's models, keyed by model id.
    pub fn save_contexts(&self, provider: &str, map: &[(String, u64)]) {
        if map.is_empty() {
            return;
        }
        // One transaction: thousands of autocommitted rows used to hold the
        // write lock long enough for the foreground save to fail with "database
        // is locked". A failed cache update is silent: it is not worth a line
        // on the user's prompt.
        let Ok(tx) = self.conn.unchecked_transaction() else { return };
        if tx
            .execute("DELETE FROM contexts WHERE provider = ?1", params![provider])
            .is_err()
        {
            return;
        }
        {
            let Ok(mut stmt) = tx.prepare(
                "INSERT OR REPLACE INTO contexts(provider, model, context) VALUES(?1,?2,?3)",
            ) else {
                return;
            };
            for (id, ctx) in map {
                if stmt
                    .execute(params![provider, id, *ctx as i64])
                    .is_err()
                {
                    return;
                }
            }
        }
        let _ = tx.commit();
    }

    /// Forget every context window learned for a provider. Discovery calls
    /// this when a server now reports an empty catalog, so numbers from a
    /// previous model set cannot leak into the next one.
    pub fn clear_contexts(&self, provider: &str) {
        let _ = self
            .conn
            .execute("DELETE FROM contexts WHERE provider = ?1", params![provider]);
    }

    /// The context window learned for one model, exact id first, then bare id
    /// (OpenRouter ids carry a `vendor/` prefix models.dev's do not).
    pub fn context_of(&self, provider: &str, model: &str) -> Option<u64> {
        let q = |m: &str| -> Option<u64> {
            self.conn
                .query_row(
                    "SELECT context FROM contexts WHERE provider = ?1 AND model = ?2",
                    params![provider, m],
                    |r| r.get::<_, i64>(0),
                )
                .ok()
                .map(|v| v.max(0) as u64)
        };
        q(model).or_else(|| model.split_once('/').and_then(|(_, bare)| q(bare)))
    }

    /// Replace the models.dev reasoning catalog. Local runtimes keep their
    /// own rows: they describe the model actually running on this machine,
    /// which the community catalog cannot know.
    pub fn save_model_thinking(
        &self,
        rows: &[(String, String, crate::provider::ThinkingSupport)],
    ) {
        let Ok(tx) = self.conn.unchecked_transaction() else { return };
        let local: Vec<String> = crate::config::LOCAL_PROVIDERS
            .iter()
            .map(|name| format!("'{}'", name.replace('\'', "''")))
            .collect();
        let sql = format!(
            "DELETE FROM model_thinking WHERE provider NOT IN ({})",
            local.join(",")
        );
        if tx.execute(&sql, []).is_err() {
            return;
        }
        {
            let Ok(mut stmt) = tx.prepare(
                "INSERT OR REPLACE INTO model_thinking(provider, model, spec) VALUES(?1,?2,?3)",
            ) else {
                return;
            };
            for (provider, model, support) in rows {
                if crate::config::LOCAL_PROVIDERS.contains(&provider.as_str()) {
                    continue;
                }
                if stmt
                    .execute(params![provider, model, support.to_json().to_string()])
                    .is_err()
                {
                    return;
                }
            }
        }
        let _ = tx.commit();
    }

    /// Replace one provider's reasoning rows. Used by local discovery, whose
    /// server is authoritative for the models it is actually serving.
    pub fn save_provider_thinking(
        &self,
        provider: &str,
        map: &[(String, crate::provider::ThinkingSupport)],
    ) {
        let Ok(tx) = self.conn.unchecked_transaction() else { return };
        if tx
            .execute("DELETE FROM model_thinking WHERE provider = ?1", params![provider])
            .is_err()
        {
            return;
        }
        {
            let Ok(mut stmt) = tx.prepare(
                "INSERT OR REPLACE INTO model_thinking(provider, model, spec) VALUES(?1,?2,?3)",
            ) else {
                return;
            };
            for (model, support) in map {
                if stmt
                    .execute(params![provider, model, support.to_json().to_string()])
                    .is_err()
                {
                    return;
                }
            }
        }
        let _ = tx.commit();
    }

    /// Cached reasoning support for one provider+model, if models.dev or a
    /// local discovery round described it.
    pub fn thinking_spec(
        &self,
        provider: &str,
        model: &str,
    ) -> Option<crate::provider::ThinkingSupport> {
        let spec: String = self
            .conn
            .query_row(
                "SELECT spec FROM model_thinking WHERE provider = ?1 AND model = ?2",
                params![provider, model],
                |r| r.get(0),
            )
            .ok()?;
        let value: serde_json::Value = serde_json::from_str(&spec).ok()?;
        crate::provider::ThinkingSupport::from_json(&value)
    }

    /// Every cached row for an exact model id, across providers. The fallback
    /// for a gateway whose own name is not in models.dev.
    pub fn thinking_specs_for(&self, model: &str) -> Vec<crate::provider::ThinkingSupport> {
        let Ok(mut stmt) = self
            .conn
            .prepare("SELECT spec FROM model_thinking WHERE model = ?1")
        else {
            return Vec::new();
        };
        stmt.query_map(params![model], |r| r.get::<_, String>(0))
            .map(|rows| {
                rows.filter_map(|r| r.ok())
                    .filter_map(|spec| serde_json::from_str::<serde_json::Value>(&spec).ok())
                    .filter_map(|value| crate::provider::ThinkingSupport::from_json(&value))
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn models_fresh(&self, provider: &str) -> bool {
        match self.cached_models(provider) {
            Some((ids, at)) => !ids.is_empty() && now_secs().saturating_sub(at) < MODELS_TTL_SECS,
            None => false,
        }
    }

    /// Thinking level per provider. This is the legacy row written before
    /// levels were remembered per model; `model_thinking` below is preferred.
    pub fn get_thinking(&self, provider: &str) -> Option<String> {
        self.conn
            .query_row(
                "SELECT value FROM session WHERE key = 'thinking:' || ?1",
                params![provider],
                |r| r.get(0),
            )
            .ok()
    }

    /// Writes the legacy per-provider row. Only tests need this: a build that
    /// remembers levels per model never writes it again, and it lingers only
    /// so an upgrade still finds the old choice.
    #[cfg(test)]
    pub fn set_thinking(&self, provider: &str, level: &str) {
        let _ = self.conn.execute(
            "INSERT INTO session(key, value) VALUES('thinking:' || ?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            params![provider, level],
        );
    }

    /// The level last chosen for this exact provider+model, falling back to
    /// the old per-provider row so an upgrade keeps the user's choice.
    pub fn get_model_thinking(&self, provider: &str, model: &str) -> Option<String> {
        self.conn
            .query_row(
                "SELECT value FROM session WHERE key = 'thinking:' || ?1 || ':' || ?2",
                params![provider, model],
                |r| r.get(0),
            )
            .ok()
            .or_else(|| self.get_thinking(provider))
    }

    pub fn set_model_thinking(&self, provider: &str, model: &str, level: &str) {
        if model.is_empty() {
            return;
        }
        let _ = self.conn.execute(
            "INSERT INTO session(key, value) VALUES('thinking:' || ?1 || ':' || ?2, ?3)
             ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            params![provider, model, level],
        );
    }

    /// The model last chosen for an auto-discovered runtime. Local providers
    /// do not get a `providers` row (that would freeze their URL), but the
    /// choice should still survive a restart if discovery still serves it.
    pub fn remembered_model(&self, provider: &str) -> Option<String> {
        self.conn
            .query_row(
                "SELECT value FROM session WHERE key = 'model:' || ?1",
                params![provider],
                |r| r.get(0),
            )
            .ok()
    }

    pub fn remember_model(&self, provider: &str, model: &str) {
        if model.is_empty() {
            return;
        }
        let _ = self.conn.execute(
            "INSERT INTO session(key, value) VALUES('model:' || ?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            params![provider, model],
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
        // Replace in one transaction: a half-written transcript (or a write
        // that loses the lock to a background refresh) is worse than a failed
        // save, and the next turn retries anyway.
        let Ok(tx) = self.conn.unchecked_transaction() else { return };
        if tx
            .execute("DELETE FROM messages WHERE session_id = ?1", params![session_id])
            .is_err()
        {
            return;
        }
        {
            let Ok(mut stmt) = tx.prepare(
                "INSERT INTO messages(session_id, idx, json) VALUES(?1,?2,?3)",
            ) else {
                return;
            };
            for (i, m) in history.iter().enumerate() {
                if stmt
                    .execute(params![session_id, i as i64, crate::provider::msg_json(m).to_string()])
                    .is_err()
                {
                    return;
                }
            }
        }
        let _ = tx.commit();
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
    fn a_custom_provider_keeps_its_kind_and_can_be_removed() {
        let s = Store::memory();
        s.save_custom_provider("localbox", "openai", "llama3.2", "http://127.0.0.1:8787/v1", "");
        let st = s.provider_state("localbox").unwrap();
        assert_eq!(st.kind, "openai");
        assert_eq!(st.api_key, "", "a local endpoint needs no key");
        // Saving the connection details again must not erase the wire shape.
        s.save_provider("localbox", "qwen2.5", "http://127.0.0.1:8787/v1", "");
        let st = s.provider_state("localbox").unwrap();
        assert_eq!(st.model, "qwen2.5");
        assert_eq!(st.kind, "openai");
        // Removal takes the row and everything cached about it.
        s.save_models("localbox", &["qwen2.5".to_string()]);
        s.save_contexts("localbox", &[("qwen2.5".to_string(), 32_000)]);
        s.save_provider_thinking("localbox", &[("qwen2.5".into(), support(&["low", "high"]))]);
        s.set_thinking("localbox", "low");
        s.set_model_thinking("localbox", "qwen2.5", "high");
        s.remember_model("localbox", "qwen2.5");
        s.remove_provider("localbox");
        assert!(s.provider_state("localbox").is_none());
        assert!(s.cached_models("localbox").is_none());
        assert!(s.context_of("localbox", "qwen2.5").is_none());
        assert!(s.get_thinking("localbox").is_none());
        assert!(s.thinking_spec("localbox", "qwen2.5").is_none());
        assert!(s.get_model_thinking("localbox", "qwen2.5").is_none());
        assert!(s.remembered_model("localbox").is_none());
        assert!(s.provider_rows().is_empty());
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

    fn support(effort: &[&str]) -> crate::provider::ThinkingSupport {
        crate::provider::ThinkingSupport {
            known: true,
            reasoning: true,
            effort: effort.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        }
    }

    #[test]
    fn a_thinking_level_belongs_to_its_provider_and_model() {
        let s = Store::memory();
        // An upgrade finds the old per-provider row until a model has its own.
        s.set_thinking("openai", "low");
        assert_eq!(s.get_model_thinking("openai", "gpt-5.2").as_deref(), Some("low"));
        s.set_model_thinking("openai", "gpt-5.2", "high");
        assert_eq!(s.get_model_thinking("openai", "gpt-5.2").as_deref(), Some("high"));
        // Another model keeps reading the legacy fallback, not this model's.
        assert_eq!(s.get_model_thinking("openai", "gpt-4.1").as_deref(), Some("low"));
        assert_eq!(s.get_thinking("openai").as_deref(), Some("low"), "legacy row untouched");
        // A model id with a colon (Ollama tags) still roundtrips.
        s.set_model_thinking("ollama", "qwen3:8b", "off");
        assert_eq!(s.get_model_thinking("ollama", "qwen3:8b").as_deref(), Some("off"));
        // An empty model cannot own a level.
        s.set_model_thinking("openai", "", "max");
        assert_eq!(s.get_model_thinking("openai", "").as_deref(), Some("low"));
    }

    #[test]
    fn reasoning_specs_are_replaced_per_source() {
        let s = Store::memory();
        s.save_model_thinking(&[
            ("openai".into(), "gpt-5.2".into(), support(&["low", "medium", "high"])),
            // Local runtimes are authoritative for the models they serve.
            ("ollama".into(), "qwen3:8b".into(), support(&["high"])),
        ]);
        assert_eq!(
            s.thinking_spec("openai", "gpt-5.2").unwrap().effort,
            vec!["low", "medium", "high"]
        );
        assert!(
            s.thinking_spec("ollama", "qwen3:8b").is_none(),
            "models.dev rows must not overwrite local discovery"
        );

        // Discovery writes its own rows and a later models.dev refresh keeps
        // them, while replacing the community rows wholesale.
        s.save_provider_thinking("ollama", &[("qwen3:8b".into(), support(&["high", "none"]))]);
        assert_eq!(
            s.thinking_spec("ollama", "qwen3:8b").unwrap().effort,
            vec!["high", "none"]
        );
        s.save_model_thinking(&[("openai".into(), "gpt-4.1".into(), support(&[]))]);
        assert!(s.thinking_spec("openai", "gpt-5.2").is_none(), "stale rows are dropped");
        assert!(s.thinking_spec("ollama", "qwen3:8b").is_some(), "local rows survive");

        // A stopped server clears its rows.
        s.save_provider_thinking("ollama", &[]);
        assert!(s.thinking_spec("ollama", "qwen3:8b").is_none());
    }

    #[test]
    fn a_spec_can_be_found_across_providers_for_a_gateway() {
        let s = Store::memory();
        s.save_provider_thinking(
            "openrouter",
            &[("anthropic/claude-sonnet-4.5".into(), support(&["high"]))],
        );
        assert_eq!(
            s.thinking_specs_for("anthropic/claude-sonnet-4.5")
                .first()
                .unwrap()
                .effort,
            vec!["high"]
        );
        assert!(s.thinking_specs_for("nothing").is_empty());
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

#[cfg(test)]
mod context_tests {
    use super::*;

    #[test]
    fn context_windows_are_learned_and_read_back_with_vendor_prefixes() {
        let s = Store::memory();
        // OpenRouter annotates its own ids (`vendor/model`); models.dev keys
        // bare ids. A lookup by either form must find the same window.
        s.save_contexts(
            "openrouter",
            &[("anthropic/claude-sonnet-4.5".to_string(), 1_000_000)],
        );
        assert_eq!(s.context_of("openrouter", "anthropic/claude-sonnet-4.5"), Some(1_000_000));
        s.save_contexts("models.dev", &[("claude-sonnet-4-5".to_string(), 200_000)]);
        assert_eq!(s.context_of("models.dev", "claude-sonnet-4-5"), Some(200_000));
        // Unknown model: None, so the heuristic guess still decides.
        assert_eq!(s.context_of("models.dev", "who-knows"), None);
        // An empty map must not wipe what is stored.
        s.save_contexts("openrouter", &[]);
        assert_eq!(s.context_of("openrouter", "anthropic/claude-sonnet-4.5"), Some(1_000_000));
    }
}
