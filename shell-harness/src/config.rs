use anyhow::Result;
use clap::{Parser, ValueEnum};
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

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum Provider {
    Anthropic,
    Openai,
    Tokenharbor,
    Openrouter,
}

impl Provider {
    pub fn name(self) -> &'static str {
        match self {
            Provider::Anthropic => "anthropic",
            Provider::Openai => "openai",
            Provider::Tokenharbor => "tokenharbor",
            Provider::Openrouter => "openrouter",
        }
    }

    pub fn default_model(self) -> &'static str {
        match self {
            Provider::Anthropic => "claude-sonnet-4-5",
            Provider::Openai => "gpt-4.1",
            Provider::Tokenharbor => "claude-sonnet-4.6",
            // ponytail: fixed default, use /model for the full catalog
            Provider::Openrouter => "openai/gpt-4.1",
        }
    }

    pub fn default_url(self) -> &'static str {
        match self {
            Provider::Anthropic => "https://api.anthropic.com",
            Provider::Openai => "https://api.openai.com/v1",
            Provider::Tokenharbor => "https://tokenharbor.ai/v1",
            Provider::Openrouter => "https://openrouter.ai/api/v1",
        }
    }

    pub fn key_env(self) -> &'static str {
        match self {
            Provider::Anthropic => "ANTHROPIC_API_KEY",
            Provider::Openai => "OPENAI_API_KEY",
            Provider::Tokenharbor => "TOKENHARBOR_API_KEY",
            Provider::Openrouter => "OPENROUTER_API_KEY",
        }
    }

    /// Reads this provider's key from the environment, if set.
    pub fn env_key(self) -> Option<String> {
        std::env::var(self.key_env()).ok().filter(|k| !k.is_empty())
    }
}

/// Pi-style effort ladder. Levels pass through to the provider verbatim
/// (`reasoning_effort`) or via a budget map (Anthropic); servers clamp.
pub const THINKING_LEVELS: &[&str] = &["off", "minimal", "low", "medium", "high", "xhigh", "max"];

#[derive(Parser, Debug)]
#[command(name = "rift", version, about = "One tool to rule them all: the shell")]
pub struct Cli {
    /// Model provider.
    #[arg(long, env = "SH_PROVIDER", value_enum, default_value = "anthropic")]
    pub provider: Provider,

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

    /// Replace the animated wait spinner with a single static line.
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

impl Provider {
    pub fn parse(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "anthropic" => Some(Provider::Anthropic),
            "openai" => Some(Provider::Openai),
            "tokenharbor" => Some(Provider::Tokenharbor),
            "openrouter" => Some(Provider::Openrouter),
            _ => None,
        }
    }
}

#[cfg(test)]
impl Config {
    /// A minimal config for tests that need a request body and nothing else.
    pub fn for_test() -> Self {
        Config {
            provider: Provider::Openai,
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
    /// Precedence per field: CLI flag > provider env var > stored DB row > default.
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
        let provider = if provider_explicit {
            cli.provider
        } else {
            store
                .active_provider()
                .and_then(|n| Provider::parse(&n))
                .unwrap_or(cli.provider)
        };
        let saved = store.provider_state(provider.name());
        // A missing key is not fatal at startup. It is reported when a request is made.
        let api_key = cli
            .api_key
            .or_else(|| provider.env_key())
            .or_else(|| saved.as_ref().map(|s| s.api_key.clone()))
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
        Ok(Self {
            provider,
            model: cli
                .model
                .or_else(|| saved.as_ref().map(|s| s.model.clone()))
                .unwrap_or_else(|| provider.default_model().to_string()),
            base_url: normalize_url(
                &cli.base_url
                    .or_else(|| saved.as_ref().map(|s| s.base_url.clone()))
                    .unwrap_or_else(|| provider.default_url().to_string()),
            ),
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

pub fn normalize_url(url: &str) -> String {
    url.trim().trim_end_matches('/').to_string()
}

impl Config {
    /// Switches provider, restoring its stored model/URL/key when present
    /// (env key wins), else that provider's defaults.
    pub fn switch_provider(&mut self, provider: Provider, store: &crate::store::Store) -> Result<()> {
        let saved = store.provider_state(provider.name());
        self.provider = provider;
        self.model = saved
            .as_ref()
            .map(|s| s.model.clone())
            .unwrap_or_else(|| provider.default_model().to_string());
        self.base_url = normalize_url(
            &saved
                .as_ref()
                .map(|s| s.base_url.clone())
                .unwrap_or_else(|| provider.default_url().to_string()),
        );
        self.api_key = provider
            .env_key()
            .or_else(|| saved.map(|s| s.api_key))
            .unwrap_or_default();
        self.thinking = store
            .get_thinking(provider.name())
            .unwrap_or_else(|| "medium".to_string());
        Ok(())
    }
}
