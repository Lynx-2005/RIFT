use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;

/// The user's home directory. `HOME` covers Unix; Windows uses `USERPROFILE`,
/// or the two variables that make it up when that is missing.
pub fn home_dir() -> Option<PathBuf> {
    for key in ["HOME", "USERPROFILE"] {
        if let Ok(h) = std::env::var(key) {
            if !h.is_empty() {
                return Some(PathBuf::from(h));
            }
        }
    }
    let drive = std::env::var("HOMEDRIVE").ok()?;
    let path = std::env::var("HOMEPATH").ok()?;
    if drive.is_empty() && path.is_empty() {
        return None;
    }
    Some(PathBuf::from(format!("{drive}{path}")))
}

/// Where per-user configuration lives: `~/.config/rift` on Unix,
/// `%APPDATA%\rift` on Windows. `RIFT_CONFIG_DIR` overrides it.
pub fn config_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("RIFT_CONFIG_DIR") {
        if !dir.is_empty() {
            return PathBuf::from(dir);
        }
    }
    #[cfg(windows)]
    if let Ok(dir) = std::env::var("APPDATA") {
        if !dir.is_empty() {
            return PathBuf::from(dir).join("rift");
        }
    }
    if let Ok(dir) = std::env::var("XDG_CONFIG_HOME") {
        if !dir.is_empty() {
            return PathBuf::from(dir).join("rift");
        }
    }
    home_dir()
        .map(|h| h.join(".config").join("rift"))
        .unwrap_or_else(|| PathBuf::from(".rift"))
}

/// The global RIFT directory, `~/.rift`. The project's `.rift` is its
/// per-project twin; this one holds settings that apply everywhere.
pub fn dot_dir() -> Option<PathBuf> {
    home_dir().map(|h| h.join(".rift"))
}

/// The provider file: a small JSON map read on every lookup, so editing it
/// takes effect without a restart. `RIFT_PROVIDERS` points somewhere else.
/// `~/.rift/providers.json` wins when it exists; the XDG location stays as a
/// fallback for existing installs.
pub fn providers_path() -> PathBuf {
    if let Ok(path) = std::env::var("RIFT_PROVIDERS") {
        if !path.is_empty() {
            return PathBuf::from(path);
        }
    }
    if let Ok(dir) = std::env::var("RIFT_CONFIG_DIR") {
        if !dir.is_empty() {
            return PathBuf::from(dir).join("providers.json");
        }
    }
    if let Some(dot) = dot_dir() {
        let path = dot.join("providers.json");
        if path.exists() {
            return path;
        }
    }
    config_dir().join("providers.json")
}

/// How a provider's HTTP API is shaped. Both are normalized to one `Turn`, so
/// this is the only thing the request builders branch on.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Wire {
    Anthropic,
    OpenAi,
}

impl Wire {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "anthropic" => Some(Wire::Anthropic),
            "openai" | "openai-compatible" | "openai_compatible" => Some(Wire::OpenAi),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Wire::Anthropic => "anthropic",
            Wire::OpenAi => "openai",
        }
    }
}

/// Model servers that run on this machine and are discovered automatically.
/// They are not "custom" providers: no registration is needed, and their model
/// catalogs come from the server itself (see `local.rs`).
pub const LOCAL_PROVIDERS: &[&str] = &["ollama", "lmstudio", "vllm"];

/// A provider is an id plus the wire shape behind it. The four built-ins carry
/// their own defaults; the local runtimes are probed on this machine; and
/// everything else is a custom endpoint declared in `providers.json`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Provider {
    name: String,
    wire: Wire,
    builtin: bool,
    local: bool,
}

impl Provider {
    /// One of the four hosted providers with defaults baked in.
    pub fn builtin(name: &str) -> Option<Self> {
        let name = name.trim().to_ascii_lowercase();
        let wire = match name.as_str() {
            "anthropic" => Wire::Anthropic,
            "openai" | "tokenharbor" | "openrouter" => Wire::OpenAi,
            _ => return None,
        };
        Some(Self { name, wire, builtin: true, local: false })
    }

    /// A local model server with a conventional endpoint. These are always
    /// resolvable, but only appear in the picker once discovery has found
    /// models (or `providers.json` configures them explicitly).
    pub fn local(name: &str) -> Option<Self> {
        let name = name.trim().to_ascii_lowercase();
        if !LOCAL_PROVIDERS.contains(&name.as_str()) {
            return None;
        }
        Some(Self { name, wire: Wire::OpenAi, builtin: false, local: true })
    }

    /// A user-defined endpoint. The name must already be a valid slug.
    pub fn custom(name: &str, wire: Wire) -> Self {
        Self { name: name.to_ascii_lowercase(), wire, builtin: false, local: false }
    }

    /// The provider a stored row describes, a local runtime, or a built-in.
    pub fn from_state(name: &str, kind: &str) -> Option<Self> {
        if let Some(p) = Self::builtin(name) {
            return Some(p);
        }
        // Rows written before custom providers stored a wire kind can only
        // have come from an OpenAI-compatible endpoint. Recover them instead
        // of hiding a provider (and a cached model list) the user already has;
        // editing `providers.json` fixes the rare miss.
        let wire = if kind.trim().is_empty() {
            Wire::OpenAi
        } else {
            Wire::parse(kind)?
        };
        if let Some(p) = Self::local(name) {
            return Some(Self { wire, ..p });
        }
        Some(Self::custom(name, wire))
    }

    /// Resolve a name against the built-ins, the local runtimes, the provider
    /// file, and custom providers saved by an older build.
    pub fn resolve(name: &str, store: &crate::store::Store) -> Option<Self> {
        let name = name.trim().to_ascii_lowercase();
        if let Some(p) = Self::builtin(&name) {
            return Some(p);
        }
        if let Some(p) = Self::local(&name) {
            // A file entry (or a legacy row) may pin a non-default endpoint or
            // wire shape for this runtime.
            let kind = crate::provider::file_provider(&name)
                .map(|p| p.state.kind)
                .or_else(|| store.provider_state(&name).map(|s| s.kind));
            let wire = kind.as_deref().and_then(Wire::parse).unwrap_or(Wire::OpenAi);
            return Some(Self { wire, ..p });
        }
        let kind = crate::provider::file_provider(&name)
            .map(|p| p.state.kind)
            .or_else(|| store.provider_state(&name).map(|s| s.kind))?;
        Self::from_state(&name, &kind)
    }

    /// Aliases kept for call sites that mean "the built-in".
    pub fn parse(name: &str) -> Option<Self> {
        Self::builtin(name)
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn wire(&self) -> Wire {
        self.wire
    }

    pub fn is_custom(&self) -> bool {
        !self.builtin && !self.local
    }

    pub fn is_local(&self) -> bool {
        self.local
    }

    pub fn is_builtin(&self) -> bool {
        self.builtin
    }

    pub fn default_model(&self) -> &'static str {
        match self.name.as_str() {
            "anthropic" => "claude-sonnet-4-5",
            "openai" => "gpt-4.1",
            "tokenharbor" => "claude-sonnet-4.6",
            // ponytail: fixed default, use /model for the full catalog
            "openrouter" => "openai/gpt-4.1",
            // A local or custom endpoint names its own model; discovery or
            // `providers.json` supplies it. There is no honest guess.
            _ => "",
        }
    }

    pub fn default_url(&self) -> &'static str {
        match self.name.as_str() {
            "anthropic" => "https://api.anthropic.com",
            "openai" => "https://api.openai.com/v1",
            "tokenharbor" => "https://tokenharbor.ai/v1",
            "openrouter" => "https://openrouter.ai/api/v1",
            "ollama" => "http://127.0.0.1:11434/v1",
            "lmstudio" => "http://127.0.0.1:1234/v1",
            "vllm" => "http://127.0.0.1:8000/v1",
            _ => "",
        }
    }

    /// The environment variable that can supply this provider's key. Custom
    /// names derive one (`RIFT_MY_SERVER_API_KEY`), so a local endpoint can be
    /// configured without storing a secret at all.
    pub fn key_env(&self) -> String {
        match self.name.as_str() {
            "anthropic" => "ANTHROPIC_API_KEY".to_string(),
            "openai" => "OPENAI_API_KEY".to_string(),
            "tokenharbor" => "TOKENHARBOR_API_KEY".to_string(),
            "openrouter" => "OPENROUTER_API_KEY".to_string(),
            // The conventional variables the local runtimes use. They are
            // optional: these endpoints normally run without authentication.
            "ollama" => "OLLAMA_API_KEY".to_string(),
            "lmstudio" => "LMSTUDIO_API_KEY".to_string(),
            "vllm" => "VLLM_API_KEY".to_string(),
            other => format!(
                "RIFT_{}_API_KEY",
                other.to_ascii_uppercase().replace(['-', '.'], "_")
            ),
        }
    }

    /// Reads this provider's key from the environment, if set.
    ///
    /// An OpenAI-compatible gateway often exports its key as `OPENAI_API_KEY`
    /// and scopes it with `OPENAI_BASE_URL`/`OPENAI_API_BASE`. Rift's OpenAI
    /// provider talks to api.openai.com, so a key scoped to another endpoint
    /// would only be sent to the wrong host and fail. Leave it unset and let
    /// the picker ask for the key that belongs to this provider.
    pub fn env_key(&self) -> Option<String> {
        let key = std::env::var(self.key_env()).ok().filter(|k| !k.is_empty())?;
        if self.name == "openai" && openai_key_scoped_elsewhere() {
            return None;
        }
        Some(key)
    }

    /// Local and third-party endpoints often take no key. Built-ins always do.
    pub fn requires_key(&self) -> bool {
        self.builtin
    }
}

/// Names allowed for a custom provider: a short slug, so it is safe to type,
/// safe as an env var, and safe as a database key.
pub fn valid_custom_name(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else { return false };
    first.is_ascii_alphanumeric()
        && name.len() <= 32
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

/// The conventional OpenAI-compatible URL for well-known local servers, used
/// when a `providers.json` entry names one without spelling the URL out.
pub fn local_default_url(name: &str) -> Option<&'static str> {
    match name {
        "ollama" => Some("http://127.0.0.1:11434/v1"),
        "lmstudio" | "lm-studio" => Some("http://127.0.0.1:1234/v1"),
        "llamacpp" | "llama-cpp" => Some("http://127.0.0.1:8080/v1"),
        "vllm" => Some("http://127.0.0.1:8000/v1"),
        _ => None,
    }
}

/// Pi-style effort ladder. Levels pass through to the provider verbatim
/// (`reasoning_effort`) or via a budget map (Anthropic); servers clamp.
pub const THINKING_LEVELS: &[&str] = &["off", "minimal", "low", "medium", "high", "xhigh", "max"];

#[derive(Parser, Debug)]
#[command(name = "rift", version, about = "One tool to rule them all: the shell")]
pub struct Cli {
    /// Provider id: anthropic, openai, tokenharbor, openrouter, an
    /// auto-discovered local runtime (ollama, lmstudio, vllm), or a custom one
    /// declared in providers.json.
    #[arg(long, env = "SH_PROVIDER", default_value = "anthropic")]
    pub provider: String,

    /// Model name. Defaults depend on provider.
    #[arg(long, short, env = "SH_MODEL")]
    pub model: Option<String>,

    /// Base URL override (for OpenAI-compatible endpoints: Ollama, vLLM, OpenRouter, ...).
    #[arg(long, env = "SH_BASE_URL")]
    pub base_url: Option<String>,

    /// API key. Prefer the env vars ANTHROPIC_API_KEY / OPENAI_API_KEY.
    #[arg(long, env = "SH_API_KEY", hide_env_values = true)]
    pub api_key: Option<String>,

    /// Thinking level for this run, overriding the stored one:
    /// off, minimal, low, medium, high, xhigh, max.
    #[arg(long, env = "SH_THINKING")]
    pub thinking: Option<String>,

    /// Per-command timeout in seconds.
    #[arg(long, default_value_t = 120)]
    pub timeout: u64,

    /// Replace the animated wait spinner and startup reveal with static text.
    #[arg(long)]
    pub no_spinner: bool,

    /// Max bytes of tool output sent back to the model.
    #[arg(long, default_value_t = 50_000)]
    pub max_output: usize,

    /// Optional throttle: most commands from one reply that may run at once.
    /// The default is 0, no limit, because how much to run at once is the
    /// model's judgement, not this harness's.
    #[arg(long, default_value_t = 0)]
    pub jobs: usize,

    /// Working directory for commands.
    #[arg(long, short = 'C')]
    pub cwd: Option<PathBuf>,

    /// Append a JSONL transcript to this file.
    #[arg(long)]
    pub log: Option<PathBuf>,

    /// Run one prompt non-interactively and exit.
    #[arg(long, short = 'p')]
    pub prompt: Option<String>,

    /// Print the system prompt and exit.
    #[arg(long)]
    pub print_system: bool,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub provider: Provider,
    pub model: String,
    pub base_url: String,
    pub api_key: String,
    pub thinking: String,
    pub timeout: std::time::Duration,
    pub max_output: usize,
    /// Up to this many commands from one reply at once. 0 means all of them.
    pub jobs: usize,
    pub cwd: PathBuf,
    pub log: Option<PathBuf>,
    pub prompt: Option<String>,
    pub print_system: bool,
    /// Animate the wait spinner. Off means one static line per wait.
    pub spinner: bool,
    /// Values that arrived as command-line flags. A flag belongs to the run it
    /// was typed on, so it is never written back over what is stored: one test
    /// against a local server must not repoint the real provider.
    pub model_flagged: bool,
    pub url_flagged: bool,
    pub key_flagged: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    fn build(args: &[&str], store: &crate::store::Store) -> anyhow::Result<Config> {
        Config::from_cli(Cli::parse_from(args), store, true)
    }

    #[test]
    fn the_stored_thinking_level_is_used_when_no_level_is_given() {
        let store = crate::store::Store::memory();
        store.set_thinking("openai", "high");
        let cfg = build(&["rift", "--provider", "openai", "--api-key", "k"], &store).unwrap();
        assert_eq!(cfg.thinking, "high");
    }

    #[test]
    fn a_thinking_level_wins_for_one_run_without_rewriting_the_store() {
        let store = crate::store::Store::memory();
        store.set_thinking("openai", "low");
        let cfg = build(
            &["rift", "--provider", "openai", "--api-key", "k", "--thinking", "max"],
            &store,
        )
        .unwrap();
        assert_eq!(cfg.thinking, "max");
        assert_eq!(
            store.get_thinking("openai").unwrap(),
            "low",
            "the run's level was written over the stored one"
        );
    }

    #[test]
    fn a_saved_custom_provider_is_a_first_class_cli_target() {
        let store = crate::store::Store::memory();
        store.save_custom_provider(
            "localbox",
            "openai",
            "llama3.2",
            "http://localhost:8787/v1",
            "",
        );
        let cfg = build(&["rift", "--provider", "localbox"], &store).unwrap();
        assert_eq!(cfg.provider, Provider::custom("localbox", Wire::OpenAi));
        assert_eq!(cfg.model, "llama3.2");
        assert_eq!(cfg.base_url, "http://localhost:8787/v1");
        assert!(cfg.api_key.is_empty(), "custom endpoints may be keyless");
        assert!(!cfg.provider.requires_key());
        // A name nobody saved is a startup error that says how to fix it.
        let err = build(&["rift", "--provider", "nope"], &store).unwrap_err().to_string();
        assert!(err.contains("unknown provider 'nope'"), "{err}");
        assert!(err.contains("providers.json"), "{err}");
    }

    #[test]
    fn a_local_runtime_is_a_cli_target_without_registration() {
        let store = crate::store::Store::memory();
        store.save_models(
            "ollama",
            &["alpha:latest".to_string(), "qwen3:8b".to_string()],
        );
        store.remember_model("ollama", "qwen3:8b");
        let cfg = build(&["rift", "--provider", "ollama"], &store).unwrap();
        assert!(cfg.provider.is_local());
        assert_eq!(cfg.provider.name(), "ollama");
        assert_eq!(
            cfg.model, "qwen3:8b",
            "a remembered choice beats the first discovered model"
        );
        assert_eq!(cfg.base_url, "http://127.0.0.1:11434/v1");
        assert!(cfg.api_key.is_empty());
        assert!(!cfg.provider.requires_key());
    }

    #[test]
    fn a_custom_provider_restores_its_stored_key_and_model() {
        let store = crate::store::Store::memory();
        store.save_custom_provider("proxy", "anthropic", "claude-x", "https://proxy/v1", "k1");
        let mut cfg = Config::for_test();
        cfg.switch_provider(&Provider::custom("proxy", Wire::Anthropic), &store).unwrap();
        assert_eq!(cfg.model, "claude-x");
        assert_eq!(cfg.base_url, "https://proxy/v1");
        assert_eq!(cfg.api_key, "k1");
        assert_eq!(cfg.provider.wire(), Wire::Anthropic);
    }

    #[test]
    fn an_unknown_thinking_level_is_rejected_with_the_list() {
        let store = crate::store::Store::memory();
        let err = build(
            &["rift", "--provider", "openai", "--api-key", "k", "--thinking", "turbo"],
            &store,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("unknown thinking level 'turbo'"), "{err}");
        assert!(err.contains("xhigh"), "the error should list the levels: {err}");
    }
}

#[cfg(test)]
impl Config {
    /// A minimal config for tests that need a request body and nothing else.
    pub fn for_test() -> Self {
        Config {
            provider: Provider::builtin("openai").unwrap(),
            model: "mock-1".into(),
            base_url: "http://localhost/v1".into(),
            api_key: "k".into(),
            thinking: "medium".into(),
            timeout: std::time::Duration::from_secs(1),
            max_output: 1000,
            jobs: 1,
            cwd: std::path::PathBuf::from("/tmp"),
            log: None,
            prompt: None,
            print_system: false,
            spinner: true,
            model_flagged: false,
            url_flagged: false,
            key_flagged: false,
        }
    }
}

impl Config {
    /// Precedence per field: CLI flag > stored DB row > provider env var > default.
    /// A key entered through `/provider` is explicit, so it outranks an ambient
    /// environment variable; the env var is still the fallback when none is stored.
    /// `provider_explicit` is true when `--provider`/`SH_PROVIDER` was given;
    /// otherwise the last session's provider is restored.
    pub fn from_cli(cli: Cli, store: &crate::store::Store, provider_explicit: bool) -> Result<Self> {
        // Read before the values move: what the user typed is this run's, not
        // the store's to keep.
        let (model_flagged, url_flagged, key_flagged) = (
            cli.model.is_some(),
            cli.base_url.is_some(),
            cli.api_key.is_some(),
        );
        let provider_name = if provider_explicit {
            cli.provider.trim().to_string()
        } else {
            store
                .active_provider()
                .unwrap_or_else(|| cli.provider.trim().to_string())
        };
        let Some(provider) = Provider::resolve(&provider_name, store) else {
            anyhow::bail!(
                "unknown provider '{provider_name}'. Built-in: {}. Local: {}. Declare a custom \
                 endpoint in {}",
                crate::commands::PROVIDERS.join(", "),
                LOCAL_PROVIDERS.join(", "),
                providers_path().display()
            );
        };
        let saved = crate::provider::state(provider.name(), store);
        // A missing key is not fatal at startup. It is reported when a request is made.
        let api_key = cli
            .api_key
            .or_else(|| {
                saved
                    .as_ref()
                    .map(|s| s.api_key.clone())
                    .filter(|k| !k.is_empty())
            })
            .or_else(|| provider.env_key())
            .unwrap_or_default();
        let cwd = match cli.cwd {
            Some(p) => p.canonicalize()?,
            None => std::env::current_dir()?,
        };
        // A flag or env var wins for this run only: the stored level stays put,
        // and an unknown level is a typo worth stopping for rather than
        // silently sending something else.
        let thinking = match cli.thinking {
            Some(level) => {
                if !THINKING_LEVELS.contains(&level.as_str()) {
                    anyhow::bail!(
                        "unknown thinking level '{level}'. Use one of: {}",
                        THINKING_LEVELS.join(", ")
                    );
                }
                level
            }
            None => store
                .get_thinking(provider.name())
                .unwrap_or_else(|| "medium".to_string()),
        };
        let model = cli
            .model
            .or_else(|| saved.as_ref().map(|s| s.model.clone()).filter(|m| !m.is_empty()))
            .unwrap_or_else(|| provider.default_model().to_string());
        let base_url = normalize_url(
            &cli.base_url
                .or_else(|| saved.as_ref().map(|s| s.base_url.clone()).filter(|u| !u.is_empty()))
                .unwrap_or_else(|| provider.default_url().to_string()),
        );
        Ok(Self {
            provider,
            model,
            base_url,
            api_key,
            thinking,
            timeout: std::time::Duration::from_secs(cli.timeout),
            max_output: cli.max_output,
            jobs: cli.jobs,
            cwd,
            log: cli.log,
            prompt: cli.prompt,
            print_system: cli.print_system,
            spinner: !(cli.no_spinner || no_spinner_env()),
            model_flagged,
            url_flagged,
            key_flagged,
        })
    }
}

/// `--no-spinner` has an env twin, like the provider flags.
fn no_spinner_env() -> bool {
    std::env::var("SH_NO_SPINNER")
        .map(|v| !v.is_empty() && v != "0")
        .unwrap_or(false)
}

/// True when the environment's OpenAI base URL names an endpoint other than
/// OpenAI's own: the accompanying `OPENAI_API_KEY` belongs to that gateway.
fn openai_key_scoped_elsewhere() -> bool {
    ["OPENAI_BASE_URL", "OPENAI_API_BASE"].iter().any(|var| {
        std::env::var(var).ok().is_some_and(|url| {
            let url = url.trim().to_ascii_lowercase();
            !url.is_empty() && !url.contains("api.openai.com")
        })
    })
}

pub fn normalize_url(url: &str) -> String {
    url.trim().trim_end_matches('/').to_string()
}

impl Config {
    /// Switches provider, restoring its stored model/URL/key when present,
    /// else that provider's defaults. A stored key wins over the ambient env
    /// var so a key replaced in the app is the one used next.
    pub fn switch_provider(
        &mut self,
        provider: &Provider,
        store: &crate::store::Store,
    ) -> Result<()> {
        let saved = crate::provider::state(provider.name(), store);
        self.model = saved
            .as_ref()
            .map(|s| s.model.clone())
            .filter(|m| !m.is_empty())
            .unwrap_or_else(|| provider.default_model().to_string());
        self.base_url = normalize_url(
            &saved
                .as_ref()
                .map(|s| s.base_url.clone())
                .filter(|u| !u.is_empty())
                .unwrap_or_else(|| provider.default_url().to_string()),
        );
        self.api_key = saved
            .as_ref()
            .map(|s| s.api_key.clone())
            .filter(|k| !k.is_empty())
            .or_else(|| provider.env_key())
            .unwrap_or_default();
        self.thinking = store
            .get_thinking(provider.name())
            .unwrap_or_else(|| "medium".to_string());
        self.provider = provider.clone();
        Ok(())
    }
}
