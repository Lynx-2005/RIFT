//! rift: a terminal agent whose tools are the shell, skills on disk, and MCP servers.
mod agent;
mod checkpoint;
mod commands;
mod config;
mod mcp;
mod mcp_http;
mod provider;
mod shell;
mod skills;
mod stats;
mod store;
mod term;
mod ui;

use anyhow::Result;
use clap::Parser;
use config::Cli;

fn main() -> Result<()> {
    term::platform_init();
    let cli = Cli::parse();
    let store = store::Store::open().unwrap_or_else(|e| {
        eprintln!("warning: session will not persist ({e:#})");
        store::Store::memory()
    });
    // Explicit flags/env win; otherwise the last session is restored.
    let args: Vec<String> = std::env::args().collect();
    let provider_explicit = args.iter().any(|a| a == "--provider" || a.starts_with("--provider="))
        || std::env::var("SH_PROVIDER").map(|v| !v.is_empty()).unwrap_or(false);
    let cfg = config::Config::from_cli(cli, &store, provider_explicit)?;
    agent::run(cfg, store)
}
