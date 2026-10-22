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

use crate::mcp_http::HttpServer;
use crate::provider::ToolSpec;
use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Sender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

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

/// How long startup may take: one server's spawn, handshake, and tool
/// listing. Every server runs in its own thread, so this is also the whole
/// startup budget, not a per-server slice of a serial wait.
pub const STARTUP_TIMEOUT: Duration = Duration::from_secs(5);

/// How long one tool call may take. Longer than startup because a server may
/// legitimately shell out and wait. Both transports use the same budget: a
/// slow tool is slow whichever wire it arrived on.
pub const CALL_TIMEOUT: Duration = Duration::from_secs(120);

/// The startup budget, with an environment override for slow servers.
pub fn startup_timeout() -> Duration {
    std::env::var("RIFT_MCP_STARTUP_TIMEOUT")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|s| s.is_finite() && *s > 0.0)
        .map(Duration::from_secs_f64)
        .unwrap_or(STARTUP_TIMEOUT)
}

/// One configured server, and the process behind it once it has started. The
/// child is shared so the manager can kill a process whose handshake is still
/// in flight when the session exits.
struct Server {
    name: String,
    child: Arc<Mutex<Child>>,
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
        let mut child = self.child.lock().unwrap_or_else(|e| e.into_inner());
        let _ = child.kill();
        let _ = child.wait();
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

/// One finished background connection attempt.
struct Connected {
    name: String,
    outcome: Result<(Backend, Vec<McpTool>), String>,
    elapsed: Duration,
}

/// Every configured server. Connections run in the background: `connect`
/// returns as soon as the threads are spawned, and `drain` folds finished
/// servers into the toolset. A slow or dead server delays only itself.
pub struct Manager {
    servers: Vec<(String, Backend)>,
    tools: Vec<McpTool>,
    notes: Vec<String>,
    /// Names of configured servers whose connection attempt is still running.
    pending: Vec<String>,
    failed: Vec<(String, String)>,
    disabled: Vec<String>,
    rx: Option<std::sync::mpsc::Receiver<Connected>>,
    cancel: Arc<AtomicBool>,
    /// Every spawned child, so a session that exits mid-handshake can still
    /// kill the servers it started.
    children: Arc<Mutex<Vec<Arc<Mutex<Child>>>>>,
}

impl Manager {
    /// An empty manager: no servers, so the harness runs on its own two tools.
    pub fn none() -> Self {
        Manager {
            servers: Vec::new(),
            tools: Vec::new(),
            notes: Vec::new(),
            pending: Vec::new(),
            failed: Vec::new(),
            disabled: Vec::new(),
            rx: None,
            cancel: Arc::new(AtomicBool::new(false)),
            children: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Start connecting to every server configured for `cwd`. Never fails and
    /// never blocks: each server gets a thread, and the session opens while
    /// they work. `drain` collects them as they land.
    pub fn connect(cwd: &Path) -> Self {
        let mut mgr = Manager::none();
        let cfg = match load_config(cwd) {
            Ok(c) => c,
            Err(e) => {
                mgr.notes.push(format!("{e:#}"));
                return mgr;
            }
        };
        let mut jobs = Vec::new();
        for (name, spec) in cfg {
            if spec.get("disabled").and_then(|d| d.as_bool()).unwrap_or(false) {
                mgr.disabled.push(name.clone());
                mgr.notes.push(format!("mcp: {name} is disabled"));
                continue;
            }
            jobs.push((name, spec));
        }
        if jobs.is_empty() {
            return mgr;
        }
        let count = jobs.len();
        let (tx, rx) = channel();
        mgr.rx = Some(rx);
        for (name, spec) in jobs {
            mgr.pending.push(name.clone());
            let tx = tx.clone();
            let cancel = mgr.cancel.clone();
            let children = mgr.children.clone();
            std::thread::spawn(move || {
                let started = Instant::now();
                let outcome = if cancel.load(Ordering::Relaxed) {
                    Err("cancelled".to_string())
                } else {
                    connect_one(&name, &spec, &cancel, &children).map_err(|e| format!("{e:#}"))
                };
                if cancel.load(Ordering::Relaxed) {
                    return;
                }
                let _ = tx.send(Connected { name, outcome, elapsed: started.elapsed() });
            });
        }
        drop(tx);
        mgr.notes.push(format!(
            "mcp: connecting {count} server{} in the background",
            if count == 1 { "" } else { "s" }
        ));
        mgr
    }

    /// Fold every finished background connection into the manager. Non-blocking
    /// and safe to call at any time.
    fn drain(&mut self) {
        let mut done: Vec<Connected> = Vec::new();
        let mut disconnected = false;
        if let Some(rx) = &self.rx {
            loop {
                match rx.try_recv() {
                    Ok(c) => done.push(c),
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        disconnected = true;
                        break;
                    }
                }
            }
        }
        if disconnected {
            self.rx = None;
        }
        for c in done {
            self.pending.retain(|n| n != &c.name);
            match c.outcome {
                Ok((backend, tools)) => {
                    let count = tools.len();
                    self.notes.push(format!(
                        "mcp: {} connected ({} tool{}, {:.1}s)",
                        c.name,
                        count,
                        if count == 1 { "" } else { "s" },
                        c.elapsed.as_secs_f64()
                    ));
                    self.tools.extend(tools);
                    self.servers.push((c.name, backend));
                }
                Err(e) => {
                    self.notes.push(format!(
                        "mcp: {} failed after {:.1}s: {e}",
                        c.name,
                        c.elapsed.as_secs_f64()
                    ));
                    self.failed.push((c.name, e));
                }
            }
        }
        // A thread that panicked never sends; a disconnected channel must not
        // leave servers looking connected forever.
        if self.rx.is_none() {
            self.pending.clear();
        }
    }

    /// Wait until every background connection settles, or `timeout` passes.
    pub fn wait_ready(&mut self, timeout: Duration) {
        let deadline = Instant::now() + timeout;
        loop {
            self.drain();
            if self.pending.is_empty() {
                break;
            }
            let now = Instant::now();
            if now >= deadline {
                break;
            }
            std::thread::sleep((deadline - now).min(Duration::from_millis(20)));
        }
    }

    /// The notes added since the last call, oldest first. Drained first, so a
    /// caller printing notes also learns about just-finished connections.
    pub fn take_notes(&mut self) -> Vec<String> {
        self.drain();
        std::mem::take(&mut self.notes)
    }

    /// What `/mcp` reports: one verdict line per configured server, then the
    /// tools each connected one contributed. A server still connecting says so.
    pub fn status(&mut self) -> String {
        self.drain();
        if self.servers.is_empty()
            && self.pending.is_empty()
            && self.failed.is_empty()
            && self.disabled.is_empty()
        {
            if !self.notes.is_empty() {
                return self.notes.join("\n");
            }
            return "no MCP servers configured. Add one to .rift/mcp.json, e.g. \
                   {\"mcpServers\": {\"name\": {\"command\": \"cmd\", \"args\": [...]}}} for stdio \
                   or {\"mcpServers\": {\"name\": {\"url\": \"https://...\", \"headers\": {}}}} for HTTP."
                .to_string();
        }
        let mut lines: Vec<String> = Vec::new();
        for (name, _) in &self.servers {
            let count = self.tools.iter().filter(|t| &t.server == name).count();
            lines.push(format!(
                "mcp: {name} connected ({count} tool{})",
                if count == 1 { "" } else { "s" }
            ));
        }
        for name in &self.pending {
            lines.push(format!("mcp: {name} connecting…"));
        }
        for name in &self.disabled {
            lines.push(format!("mcp: {name} disabled"));
        }
        for (name, err) in &self.failed {
            lines.push(format!("mcp: {name} failed: {err}"));
        }
        let mut out = lines.join("\n");
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

    /// This session's tool list: the harness's own, then every server's that
    /// has connected so far. Drained first so a server that just landed is
    /// visible to the next model round.
    pub fn tool_specs(&mut self) -> Vec<ToolSpec> {
        self.drain();
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
        self.drain();
        let Some((server, tool)) = split_wire_name(wire_name) else {
            return format!("[tool error] {wire_name} is not an MCP tool name");
        };
        if self.pending.iter().any(|n| n == server) {
            return format!("[tool error] MCP server {server} is still connecting");
        }
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
        match self.servers[i].1.request("tools/call", params, CALL_TIMEOUT) {
            Ok(result) => Server::text_of(&result),
            Err(e) => format!("[tool error] {e:#}"),
        }
    }
}

impl Drop for Manager {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
        // Pending threads notice the flag within the startup budget; this
        // walks everything already spawned so nothing is orphaned on exit.
        let children = self.children.lock().unwrap_or_else(|e| e.into_inner());
        for child in children.iter() {
            let mut c = child.lock().unwrap_or_else(|e| e.into_inner());
            let _ = c.kill();
        }
    }
}

/// One background connection: start the server, handshake, list its tools.
fn connect_one(
    name: &str,
    spec: &Value,
    cancel: &AtomicBool,
    children: &Arc<Mutex<Vec<Arc<Mutex<Child>>>>>,
) -> Result<(Backend, Vec<McpTool>)> {
    let mut server = start(name, spec, children)?;
    if cancel.load(Ordering::Relaxed) {
        bail!("cancelled");
    }
    let list = server.request("tools/list", json!({}), startup_timeout())?;
    if cancel.load(Ordering::Relaxed) {
        bail!("cancelled");
    }
    Ok((server, tools_from(name, &list)))
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
fn start(name: &str, spec: &Value, children: &Arc<Mutex<Vec<Arc<Mutex<Child>>>>>) -> Result<Backend> {
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
            return HttpServer::connect(url, &headers, startup_timeout())
                .map(Backend::Http)
                .with_context(|| format!("server {name} ({url})"));
        }
    }
    start_stdio(name, spec, children).map(Backend::Stdio)
}

/// Start one stdio server: spawn it, wire a reader thread to its stdout, and
/// run the handshake. The reader thread owns stdout for the process's life and
/// routes each reply to whoever is waiting on that id.
fn start_stdio(name: &str, spec: &Value, children: &Arc<Mutex<Vec<Arc<Mutex<Child>>>>>) -> Result<Server> {
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
    let child = Arc::new(Mutex::new(child));
    // Registered before the handshake: a session that exits mid-connect can
    // still kill a server whose thread is blocked on a reply.
    children.lock().unwrap_or_else(|e| e.into_inner()).push(child.clone());
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
        startup_timeout(),
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
import json, os, sys, time
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
        time.sleep(float(os.environ.get("FAKE_DELAY", "0")))
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

    /// Write a fake-server config and point `RIFT_MCP_CONFIG` at it while
    /// `f` runs, so one test's server cannot leak into another's. Times `f`,
    /// because "connect does not wait" is asserted on that number.
    fn with_fake_config<R>(
        name: &str,
        delay: f64,
        f: impl FnOnce() -> R,
    ) -> Option<(R, Duration)> {
        let _g = ENV_GUARD.lock().unwrap_or_else(|e| e.into_inner());
        let interpreter = python()?;
        let dir = std::env::temp_dir().join(format!("rift-mcp-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let script = dir.join(format!("{name}.py"));
        std::fs::write(&script, FAKE_SERVER).unwrap();
        let cfg = dir.join(format!("{name}.json"));
        std::fs::write(
            &cfg,
            json!({"mcpServers": {name: {
                "command": interpreter,
                "args": [script.to_string_lossy()],
                "env": {"FAKE_DELAY": delay.to_string()},
            }}})
            .to_string(),
        )
        .unwrap();
        let saved = std::env::var("RIFT_MCP_CONFIG").ok();
        std::env::set_var("RIFT_MCP_CONFIG", cfg.to_string_lossy().to_string());
        let started = Instant::now();
        let out = f();
        let took = started.elapsed();
        match saved {
            Some(v) => std::env::set_var("RIFT_MCP_CONFIG", v),
            None => std::env::remove_var("RIFT_MCP_CONFIG"),
        }
        Some((out, took))
    }

    /// A manager whose fake server has finished connecting, or `None`
    /// without Python.
    fn fake() -> Option<Manager> {
        let mut mgr = with_fake_config("fake", 0.0, || Manager::connect(Path::new("/tmp")))
            .map(|(m, _)| m)?;
        mgr.wait_ready(Duration::from_secs(5));
        Some(mgr)
    }

    #[test]
    fn a_server_contributes_its_tools_under_a_namespaced_name() {
        let Some(mut mgr) = fake() else { return };
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
    fn connect_returns_before_a_slow_server_finishes() {
        let Some((mut mgr, took)) =
            with_fake_config("slow", 1.0, || Manager::connect(Path::new("/tmp")))
        else {
            return;
        };
        assert!(
            took < Duration::from_millis(300),
            "connect blocked for {took:?}; startup must not wait for servers"
        );
        assert!(
            mgr.tool_specs().iter().all(|s| s.name != "slow__echo"),
            "the slow server should still be connecting"
        );
        mgr.wait_ready(Duration::from_secs(5));
        assert!(
            mgr.tool_specs().iter().any(|s| s.name == "slow__echo"),
            "the slow server must land once it finishes"
        );
    }

    #[test]
    fn status_reports_a_connected_server_and_its_tools() {
        let Some(mut mgr) = fake() else { return };
        let st = mgr.status();
        assert!(st.contains("mcp: fake connected (1 tool)"), "{st}");
        assert!(st.contains("fake__echo"), "{st}");
        assert!(st.contains("echo the args"), "{st}");
    }

    #[test]
    fn status_with_no_servers_says_how_to_add_one() {
        let mut mgr = Manager::none();
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
        let cfg = dir.join("broken.json");
        std::fs::write(
            &cfg,
            json!({"mcpServers": {"broken": {"command": "/nonexistent/definitely-not-a-program"}}})
                .to_string(),
        )
        .unwrap();
        let saved = std::env::var("RIFT_MCP_CONFIG").ok();
        std::env::set_var("RIFT_MCP_CONFIG", cfg.to_string_lossy().to_string());
        let mut mgr = Manager::connect(Path::new("/tmp"));
        match saved {
            Some(v) => std::env::set_var("RIFT_MCP_CONFIG", v),
            None => std::env::remove_var("RIFT_MCP_CONFIG"),
        }
        mgr.wait_ready(Duration::from_secs(5));
        assert!(mgr.is_empty(), "a dead server must not contribute tools");
        assert_eq!(mgr.tool_specs().len(), 2, "the harness's own tools survive");
        let notes = mgr.take_notes();
        assert!(
            notes.iter().any(|n| n.contains("broken") && n.contains("failed")),
            "{notes:?}"
        );
    }

    #[test]
    fn no_config_means_no_servers_and_no_notes() {
        let _g = ENV_GUARD.lock().unwrap_or_else(|e| e.into_inner());
        let saved = std::env::var("RIFT_MCP_CONFIG").ok();
        std::env::set_var("RIFT_MCP_CONFIG", "/nonexistent/mcp.json");
        let mut mgr = Manager::connect(Path::new("/tmp"));
        match saved {
            Some(v) => std::env::set_var("RIFT_MCP_CONFIG", v),
            None => std::env::remove_var("RIFT_MCP_CONFIG"),
        }
        assert!(mgr.is_empty());
        assert!(mgr.take_notes().is_empty());
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
        mgr.wait_ready(Duration::from_secs(15));
        let notes = mgr.take_notes().join(" | ");
        assert!(notes.contains("solana connected"), "connect failed: {notes}");
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
