//! MCP: Model Context Protocol servers, over stdio or streamable HTTP.
//!
//! A stdio server is a child process speaking newline-delimited JSON-RPC 2.0;
//! an HTTP server is a URL the same JSON-RPC is POSTed to, with replies coming
//! back as JSON or an SSE stream. Either way the shape after the handshake is
//! `initialize`, then `tools/list`, then `tools/call` per call, and the
//! server's tools are merged into the same list as the harness's own two, so
//! the model sees one toolset and does not care which side a tool comes from.
//!
//! Servers are configured in `mcp.json` (see `config_paths`): `{"command":
//! ..., "args": [...]}` for stdio, or `{"url": ..., "headers": {...}}` for
//! HTTP, the same shape Claude Code accepts. Nothing here is fatal: a server
//! that will not start is reported once and skipped, because a broken optional
//! integration must not stop a shell session from opening.

use crate::mcp_http::{HttpServer, CALL_TIMEOUT as HTTP_CALL_TIMEOUT, HANDSHAKE_TIMEOUT as HTTP_HANDSHAKE_TIMEOUT};
use crate::provider::ToolSpec;
use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// One tool a server offers, with the server it belongs to. The wire name is
/// `server__tool`: MCP tool names are only unique within a server, and the
/// model needs to be told which one to call.
#[derive(Debug, Clone, PartialEq)]
pub struct McpTool {
    pub server: String,
    pub name: String,
    pub description: String,
    pub schema: Value,
}

impl McpTool {
    /// The name the model sees and calls.
    pub fn wire_name(&self) -> String {
        format!("{}__{}", self.server, self.name)
    }
}

/// Split a wire name back into (server, tool). Splits on the first `__`, so a
/// tool whose own name contains one still resolves.
pub fn split_wire_name(wire: &str) -> Option<(&str, &str)> {
    let (server, tool) = wire.split_once("__")?;
    if server.is_empty() || tool.is_empty() {
        return None;
    }
    Some((server, tool))
}

/// How long a request may take. Tool calls get longer than the handshake
/// because a server may legitimately shell out and wait. Both transports use
/// the same budgets: a slow tool is slow whichever wire it arrived on.
pub const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(20);
pub const CALL_TIMEOUT: Duration = Duration::from_secs(120);

/// One configured server, and the process behind it once it has started.
struct Server {
    name: String,
    child: Child,
    stdin: ChildStdin,
    pending: Arc<Mutex<HashMap<u64, Sender<Value>>>>,
    next_id: u64,
}

impl Server {
    /// Send a request and wait for its reply. `timeout` is enforced here rather
    /// than by the caller, so a hung server never hangs the agent loop.
    fn request(&mut self, method: &str, params: Value, timeout: Duration) -> Result<Value> {
        self.next_id += 1;
        let id = self.next_id;
        let (tx, rx) = channel();
        self.pending.lock().unwrap_or_else(|e| e.into_inner()).insert(id, tx);
        let msg = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        let line = format!("{msg}\n");
        if let Err(e) = self.stdin.write_all(line.as_bytes()).and_then(|_| self.stdin.flush()) {
            self.pending.lock().unwrap_or_else(|e| e.into_inner()).remove(&id);
            bail!("writing to {} failed: {e}", self.name);
        }
        match rx.recv_timeout(timeout) {
            Ok(v) => {
                if let Some(err) = v.get("error").filter(|e| !e.is_null()) {
                    bail!(
                        "{} answered with an error: {}",
                        self.name,
                        err["message"].as_str().unwrap_or(&err.to_string())
                    );
                }
                Ok(v.get("result").cloned().unwrap_or(Value::Null))
            }
            Err(_) => {
                self.pending.lock().unwrap_or_else(|e| e.into_inner()).remove(&id);
                bail!("{} did not answer {method} within {}s", self.name, timeout.as_secs())
            }
        }
    }

    /// Read one tool result out of a `tools/call` reply, as the text the model
    /// should see. MCP returns a list of content blocks; text blocks are joined
    /// and anything else is described rather than dropped silently.
    fn text_of(result: &Value) -> String {
        let blocks = result.get("content").and_then(|c| c.as_array()).cloned().unwrap_or_default();
        let mut out = String::new();
        for b in blocks {
            match b["type"].as_str() {
                Some("text") => {
                    if !out.is_empty() {
                        out.push('\n');
                    }
                    out.push_str(b["text"].as_str().unwrap_or(""));
                }
                Some(other) => {
                    if !out.is_empty() {
                        out.push('\n');
                    }
                    out.push_str(&format!("[{other} content]"));
                }
                None => {}
            }
        }
        if result.get("isError").and_then(|e| e.as_bool()).unwrap_or(false) {
            out.insert_str(0, "[tool error]\n");
        }
        if out.trim().is_empty() {
            out = "(no output)".to_string();
        }
        out
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// One connected server, whichever wire it lives on. Both transports answer
/// the same three methods; everything above this line does not care which.
enum Backend {
    Stdio(Server),
    Http(HttpServer),
}

impl Backend {
    fn request(&mut self, method: &str, params: Value, timeout: Duration) -> Result<Value> {
        match self {
            Backend::Stdio(s) => s.request(method, params, timeout),
            Backend::Http(h) => h.request(method, params, timeout),
        }
    }
}

/// Every configured server, connected. Cheap to hold: one process or one
/// connection each, idle until a call arrives.
pub struct Manager {
    servers: Vec<(String, Backend)>,
    tools: Vec<McpTool>,
    notes: Vec<String>,
}

impl Manager {
    /// An empty manager: no servers, so the harness runs on its own two tools.
    pub fn none() -> Self {
        Manager { servers: Vec::new(), tools: Vec::new(), notes: Vec::new() }
    }

    /// Connect to every server configured for `cwd`. Never fails: each failure
    /// becomes a note, and the rest of the session carries on.
    pub fn connect(cwd: &Path) -> Self {
        let cfg = match load_config(cwd) {
            Ok(c) => c,
            Err(e) => return Manager { servers: vec![], tools: vec![], notes: vec![format!("{e:#}")] },
        };
        let mut mgr = Manager::none();
        for (name, spec) in cfg {
            match start(&name, &spec) {
                Ok(mut server) => {
                    let timeout = match server {
                        Backend::Stdio(_) => HANDSHAKE_TIMEOUT,
                        Backend::Http(_) => HTTP_HANDSHAKE_TIMEOUT,
                    };
                    match server.request("tools/list", json!({}), timeout) {
                        Ok(list) => {
                            let tools = tools_from(&name, &list);
                            mgr.notes.push(format!(
                                "mcp: {name} connected ({} tool{})",
                                tools.len(),
                                if tools.len() == 1 { "" } else { "s" }
                            ));
                            mgr.tools.extend(tools);
                            mgr.servers.push((name.clone(), server));
                        }
                        Err(e) => {
                            mgr.notes.push(format!("mcp: {name} listed no tools: {e:#}"));
                        }
                    }
                }
                Err(e) => mgr.notes.push(format!("mcp: {name} did not start: {e:#}")),
            }
        }
        mgr
    }

    /// One line per server, for the arrival banner.
    pub fn notes(&self) -> &[String] {
        &self.notes
    }

    /// What `/mcp` reports: one verdict line per configured server, then the
    /// tools each connected one contributed. Everything was learned at
    /// startup, so asking never touches the servers again.
    pub fn status(&self) -> String {
        if self.notes.is_empty() {
            return "no MCP servers configured. Add one to .rift/mcp.json, e.g. \
                   {\"mcpServers\": {\"name\": {\"command\": \"cmd\", \"args\": [...]}}} for stdio \
                   or {\"mcpServers\": {\"name\": {\"url\": \"https://...\", \"headers\": {}}}} for HTTP."
                .to_string();
        }
        let mut out = self.notes.join("\n");
        for t in &self.tools {
            let desc = t.description.lines().next().unwrap_or("").trim();
            out.push_str(&format!("\n  {}  {}", t.wire_name(), desc));
        }
        out
    }

    /// True when no server offered anything. Tests use it to say what a dead
    /// server did to the toolset.
    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// This session's tool list: the harness's own, then every server's.
    pub fn tool_specs(&self) -> Vec<ToolSpec> {
        let mut specs = crate::provider::builtin_tools();
        for t in &self.tools {
            specs.push(ToolSpec {
                name: t.wire_name(),
                description: format!("[{}] {}", t.server, t.description),
                schema: t.schema.clone(),
            });
        }
        specs
    }

    /// Call one server tool. A failure is returned as text, not an error: the
    /// model is told what went wrong and can try something else, which is
    /// better than the turn dying over one unavailable tool.
    pub fn call(&mut self, wire_name: &str, args: &Value) -> String {
        let Some((server, tool)) = split_wire_name(wire_name) else {
            return format!("[tool error] {wire_name} is not an MCP tool name");
        };
        let mut hit = None;
        for (i, (n, _)) in self.servers.iter().enumerate() {
            if n == server {
                hit = Some(i);
                break;
            }
        }
        let Some(i) = hit else {
            return format!("[tool error] no MCP server named {server}");
        };
        let params = json!({"name": tool, "arguments": args});
        let timeout = match &self.servers[i].1 {
            Backend::Stdio(_) => CALL_TIMEOUT,
            Backend::Http(_) => HTTP_CALL_TIMEOUT,
        };
        match self.servers[i].1.request("tools/call", params, timeout) {
            Ok(result) => Server::text_of(&result),
            Err(e) => format!("[tool error] {e:#}"),
        }
    }
}

/// Turn a `tools/list` reply into tools, skipping entries that could not be
/// called (no name) rather than offering the model a tool that cannot work.
fn tools_from(server: &str, list: &Value) -> Vec<McpTool> {
    list.get("tools")
        .and_then(|t| t.as_array())
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter_map(|t| {
            let name = t["name"].as_str()?.to_string();
            Some(McpTool {
                server: server.to_string(),
                name,
                description: t["description"].as_str().unwrap_or("").to_string(),
                schema: if t["inputSchema"].is_object() {
                    t["inputSchema"].clone()
                } else {
                    json!({"type": "object", "properties": {}})
                },
            })
        })
        .collect()
}

/// Start one server. A spec with a `url` is an HTTP server; one with a
/// `command` is a child process. Anything else has nothing to run.
fn start(name: &str, spec: &Value) -> Result<Backend> {
    if let Some(url) = spec["url"].as_str().filter(|u| !u.is_empty()) {
        let headers: Vec<(String, String)> = spec["headers"]
            .as_object()
            .map(|m| {
                m.iter()
                    .filter_map(|(k, v)| v.as_str().map(|v| (k.clone(), v.to_string())))
                    .collect()
            })
            .unwrap_or_default();
        if !url.is_empty() {
            return HttpServer::connect(url, &headers)
                .map(Backend::Http)
                .with_context(|| format!("server {name} ({url})"));
        }
    }
    start_stdio(name, spec).map(Backend::Stdio)
}

/// Start one stdio server: spawn it, wire a reader thread to its stdout, and
/// run the handshake. The reader thread owns stdout for the process's life and
/// routes each reply to whoever is waiting on that id.
fn start_stdio(name: &str, spec: &Value) -> Result<Server> {
    let command = spec["command"]
        .as_str()
        .filter(|c| !c.is_empty())
        .with_context(|| format!("server {name} has no command"))?;
    let mut cmd = Command::new(command);
    for a in spec["args"].as_array().cloned().unwrap_or_default() {
        if let Some(a) = a.as_str() {
            cmd.arg(a);
        }
    }
    if let Some(env) = spec["env"].as_object() {
        for (k, v) in env {
            if let Some(v) = v.as_str() {
                cmd.env(k, v);
            }
        }
    }
    // A server that cannot find its own working directory is a common failure,
    // so a configured one is honoured.
    if let Some(dir) = spec["cwd"].as_str() {
        cmd.current_dir(dir);
    }
    let mut child = cmd
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .with_context(|| format!("spawning {command}"))?;
    let stdin = child.stdin.take().context("no stdin on the server process")?;
    let stdout = child.stdout.take().context("no stdout on the server process")?;
    let pending: Arc<Mutex<HashMap<u64, Sender<Value>>>> = Arc::new(Mutex::new(HashMap::new()));
    {
        let pending = pending.clone();
        std::thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                let Ok(line) = line else { break };
                let Ok(v) = serde_json::from_str::<Value>(&line) else { continue };
                // Notifications carry no id and are none of our business.
                let Some(id) = v.get("id").and_then(|i| i.as_u64()) else { continue };
                if let Some(tx) = pending.lock().unwrap_or_else(|e| e.into_inner()).remove(&id) {
                    let _ = tx.send(v);
                }
            }
        });
    }
    let mut server = Server {
        name: name.to_string(),
        child,
        stdin,
        pending,
        next_id: 0,
    };
    server.request(
        "initialize",
        json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "rift", "version": env!("CARGO_PKG_VERSION")},
        }),
        HANDSHAKE_TIMEOUT,
    )?;
    // The protocol requires this before any other request. A server that does
    // not want it will ignore it, so a failure here is not worth reporting.
    let _ = server.stdin.write_all(
        format!("{}\n", json!({"jsonrpc": "2.0", "method": "notifications/initialized"})).as_bytes(),
    );
    let _ = server.stdin.flush();
    Ok(server)
}

/// Where a config may live, project first. `RIFT_MCP_CONFIG` replaces the list.
pub fn config_paths(cwd: &Path) -> Vec<PathBuf> {
    if let Ok(p) = std::env::var("RIFT_MCP_CONFIG") {
        return std::env::split_paths(&p)
            .filter(|s| !s.as_os_str().is_empty())
            .collect();
    }
    let mut paths = vec![cwd.join(".rift").join("mcp.json"), cwd.join(".mcp.json")];
    let mut push = |p: PathBuf| {
        if !paths.contains(&p) {
            paths.push(p);
        }
    };
    if let Some(dot) = crate::config::dot_dir() {
        push(dot.join("mcp.json"));
    }
    push(crate::config::config_dir().join("mcp.json"));
    if let Some(home) = crate::config::home_dir() {
        push(home.join(".claude.json"));
    }
    paths
}

/// Every server configured in any config file, merged. A later file cannot
/// override an earlier one: the project's server is the one that wins.
fn load_config(cwd: &Path) -> Result<Vec<(String, Value)>> {
    let mut out: Vec<(String, Value)> = Vec::new();
    for path in config_paths(cwd) {
        let Ok(text) = std::fs::read_to_string(&path) else { continue };
        let Ok(v) = serde_json::from_str::<Value>(&text) else {
            bail!("{} is not valid JSON", path.display());
        };
        // `mcpServers` at the top level, and Claude Code's per-project nesting.
        let mut maps: Vec<&Value> = Vec::new();
        if let Some(m) = v.get("mcpServers") {
            maps.push(m);
        }
        if let Some(projects) = v.get("projects").and_then(|p| p.as_object()) {
            for (dir, entry) in projects {
                if Path::new(dir) == cwd {
                    if let Some(m) = entry.get("mcpServers") {
                        maps.push(m);
                    }
                }
            }
        }
        for m in maps {
            let Some(obj) = m.as_object() else { continue };
            for (name, spec) in obj {
                if !out.iter().any(|(n, _)| n == name) {
                    out.push((name.clone(), spec.clone()));
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fake MCP server: newline-delimited JSON-RPC over stdio, written in
    /// Python because that is the one interpreter every box here has.
    /// It answers `initialize`, lists one tool, and echoes the arguments back
    /// on `tools/call` so the round trip can be asserted end to end.
    const FAKE_SERVER: &str = r#"
import json, sys
def send(v):
    sys.stdout.write(json.dumps(v) + "\n"); sys.stdout.flush()
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    try: msg = json.loads(line)
    except Exception: continue
    m = msg.get("method")
    if m == "initialize":
        send({"jsonrpc":"2.0","id":msg["id"],"result":{"protocolVersion":"2024-11-05","capabilities":{},"serverInfo":{"name":"fake","version":"1"}}})
    elif m == "tools/list":
        send({"jsonrpc":"2.0","id":msg["id"],"result":{"tools":[{"name":"echo","description":"echo the args","inputSchema":{"type":"object","properties":{"text":{"type":"string"}}}}]}})
    elif m == "tools/call":
        a = msg["params"]["arguments"]
        send({"jsonrpc":"2.0","id":msg["id"],"result":{"content":[{"type":"text","text":"echo: " + str(a.get("text",""))}]}})
    elif m == "tools/bad":
        send({"jsonrpc":"2.0","id":msg["id"],"error":{"code":-32000,"message":"nope"}})
    elif "id" in msg:
        send({"jsonrpc":"2.0","id":msg["id"],"result":{}})
"#;

    static ENV_GUARD: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn python() -> Option<&'static str> {
        ["python3", "python"].into_iter().find(|cmd| {
            Command::new(cmd)
                .arg("--version")
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .map(|s| s.success())
                .unwrap_or(false)
        })
    }

    /// A manager connected to the fake server, or `None` without Python.
    fn fake() -> Option<Manager> {
        let _g = ENV_GUARD.lock().unwrap_or_else(|e| e.into_inner());
        let interpreter = python()?;
        let dir = std::env::temp_dir().join(format!("rift-mcp-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let script = dir.join("server.py");
        std::fs::write(&script, FAKE_SERVER).unwrap();
        let cfg = dir.join("mcp.json");
        std::fs::write(
            &cfg,
            json!({"mcpServers": {"fake": {"command": interpreter, "args": [script.to_string_lossy()]}}})
                .to_string(),
        )
        .unwrap();
        let saved = std::env::var("RIFT_MCP_CONFIG").ok();
        std::env::set_var("RIFT_MCP_CONFIG", cfg.to_string_lossy().to_string());
        let mgr = Manager::connect(Path::new("/tmp"));
        match saved {
            Some(v) => std::env::set_var("RIFT_MCP_CONFIG", v),
            None => std::env::remove_var("RIFT_MCP_CONFIG"),
        }
        Some(mgr)
    }

    #[test]
    fn a_server_contributes_its_tools_under_a_namespaced_name() {
        let Some(mgr) = fake() else { return };
        let specs = mgr.tool_specs();
        let names: Vec<&str> = specs.iter().map(|s| s.name.as_str()).collect();
        // The harness's own tools come first, so `shell` is always there.
        assert_eq!(names[0], "shell");
        assert_eq!(names[1], "skill");
        assert!(names.contains(&"fake__echo"), "{names:?}");
        let echo = specs.iter().find(|s| s.name == "fake__echo").unwrap();
        assert!(echo.description.contains("[fake] echo the args"), "{}", echo.description);
        assert_eq!(echo.schema["properties"]["text"]["type"], "string");
    }

    #[test]
    fn a_call_round_trips_through_the_server() {
        let Some(mut mgr) = fake() else { return };
        let out = mgr.call("fake__echo", &json!({"text": "hello"}));
        assert_eq!(out, "echo: hello");
    }

    #[test]
    fn status_reports_a_connected_server_and_its_tools() {
        let Some(mgr) = fake() else { return };
        let st = mgr.status();
        assert!(st.contains("mcp: fake connected (1 tool)"), "{st}");
        assert!(st.contains("fake__echo"), "{st}");
        assert!(st.contains("echo the args"), "{st}");
    }

    #[test]
    fn status_with_no_servers_says_how_to_add_one() {
        let mgr = Manager::none();
        assert!(mgr.status().contains("no MCP servers configured"), "{}", mgr.status());
        assert!(mgr.status().contains(".rift/mcp.json"), "{}", mgr.status());
    }

    #[test]
    fn an_unknown_tool_or_server_is_reported_as_text_not_a_crash() {
        let Some(mut mgr) = fake() else { return };
        assert!(mgr.call("nobody__thing", &json!({})).contains("no MCP server named nobody"));
        assert!(mgr.call("notanmcptool", &json!({})).contains("is not an MCP tool name"));
    }

    #[test]
    fn a_server_that_cannot_start_is_a_note_not_a_failure() {
        let _g = ENV_GUARD.lock().unwrap_or_else(|e| e.into_inner());
        let dir = std::env::temp_dir().join(format!("rift-mcp-bad-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let cfg = dir.join("mcp.json");
        std::fs::write(
            &cfg,
            json!({"mcpServers": {"broken": {"command": "/nonexistent/definitely-not-a-program"}}})
                .to_string(),
        )
        .unwrap();
        let saved = std::env::var("RIFT_MCP_CONFIG").ok();
        std::env::set_var("RIFT_MCP_CONFIG", cfg.to_string_lossy().to_string());
        let mgr = Manager::connect(Path::new("/tmp"));
        match saved {
            Some(v) => std::env::set_var("RIFT_MCP_CONFIG", v),
            None => std::env::remove_var("RIFT_MCP_CONFIG"),
        }
        assert!(mgr.is_empty(), "a dead server must not contribute tools");
        assert_eq!(mgr.tool_specs().len(), 2, "the harness's own tools survive");
        assert!(
            mgr.notes().iter().any(|n| n.contains("broken") && n.contains("did not start")),
            "{:?}",
            mgr.notes()
        );
    }

    #[test]
    fn no_config_means_no_servers_and_no_notes() {
        let _g = ENV_GUARD.lock().unwrap_or_else(|e| e.into_inner());
        let saved = std::env::var("RIFT_MCP_CONFIG").ok();
        std::env::set_var("RIFT_MCP_CONFIG", "/nonexistent/mcp.json");
        let mgr = Manager::connect(Path::new("/tmp"));
        match saved {
            Some(v) => std::env::set_var("RIFT_MCP_CONFIG", v),
            None => std::env::remove_var("RIFT_MCP_CONFIG"),
        }
        assert!(mgr.is_empty());
        assert!(mgr.notes().is_empty());
    }

    #[test]
    fn wire_names_split_on_the_first_separator() {
        assert_eq!(split_wire_name("git__log"), Some(("git", "log")));
        assert_eq!(split_wire_name("a__b__c"), Some(("a", "b__c")));
        assert_eq!(split_wire_name("shell"), None);
        assert_eq!(split_wire_name("__x"), None);
        assert_eq!(split_wire_name("x__"), None);
    }
}

#[cfg(test)]
mod live_tests {
    use super::*;

    /// The real thing: the project's `.rift/mcp.json` against the live Solana
    /// MCP endpoint. Ignored by default because it needs the network; run with
    /// `cargo test live -- --ignored`.
    #[test]
    #[ignore = "needs the network; run explicitly with --ignored"]
    fn the_solana_http_server_connects_and_lists_tools() {
        let dir = std::env::temp_dir().join(format!("rift-mcp-live-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let cfg = dir.join("mcp.json");
        std::fs::write(
            &cfg,
            json!({"mcpServers": {"solana": {"url": "https://mcp.solana.com/mcp"}}}).to_string(),
        )
        .unwrap();
        let saved = std::env::var("RIFT_MCP_CONFIG").ok();
        std::env::set_var("RIFT_MCP_CONFIG", cfg.to_string_lossy().to_string());
        let mut mgr = Manager::connect(Path::new("/tmp"));
        match saved {
            Some(v) => std::env::set_var("RIFT_MCP_CONFIG", v),
            None => std::env::remove_var("RIFT_MCP_CONFIG"),
        }
        let notes = mgr.notes().iter().cloned().collect::<Vec<_>>().join(" | ");
        assert!(
            mgr.notes().iter().any(|n| n.contains("solana connected")),
            "connect failed: {notes}"
        );
        let names: Vec<String> = mgr.tools.iter().map(|t| t.wire_name()).collect();
        assert!(
            names.iter().any(|n| n.contains("Documentation_Search") || n.contains("Expert")),
            "no doc tools among {names:?}"
        );
        // One real call end to end: a docs search must come back with text.
        let search = mgr
            .tools
            .iter()
            .find(|t| t.name.contains("Documentation_Search"))
            .expect("a search tool exists");
        let out = mgr.call(&search.wire_name(), &json!({"query": "compute budget"}));
        assert!(!out.starts_with("[tool error]"), "search failed: {out}");
        assert!(!out.trim().is_empty());
    }
}
