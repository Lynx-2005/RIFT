//! Streamable-HTTP transport for MCP servers, the way a remote server wants
//! to be talked to. One endpoint, JSON-RPC 2.0 bodies in; the reply comes back
//! either as a single JSON document or, when the server prefers it, as a
//! server-sent-events stream whose `data:` lines are those same documents.

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::io::BufRead;
use std::sync::mpsc::{channel, Receiver};
use std::time::Duration;

/// One connected HTTP server.
pub struct HttpServer {
    http: reqwest::blocking::Client,
    url: String,
    headers: Vec<(String, String)>,
    /// The server may hand out a session id on `initialize`; every later
    /// request carries it back in `Mcp-Session-Id`.
    session: Option<String>,
    /// The negotiated protocol version, sent on every request after the
    /// handshake as `MCP-Protocol-Version`.
    protocol: Option<String>,
    next_id: u64,
}

impl HttpServer {
    /// Open a connection: build the client and run the initialize handshake.
    /// `timeout` bounds the whole startup, the TCP connect included.
    pub fn connect(url: &str, headers: &[(String, String)], timeout: Duration) -> Result<Self> {
        let http = reqwest::blocking::Client::builder()
            .connect_timeout(timeout)
            .build()
            .context("building the HTTP client")?;
        let mut s = HttpServer {
            http,
            url: url.to_string(),
            headers: headers.to_vec(),
            session: None,
            protocol: None,
            next_id: 0,
        };
        let reply = s.request(
            "initialize",
            json!({
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "rift", "version": env!("CARGO_PKG_VERSION")},
            }),
            timeout,
        )?;
        // The server names the version it will speak; when it omits one, keep
        // the revision the request asked for.
        let agreed = reply
            .get("protocolVersion")
            .and_then(|p| p.as_str())
            .unwrap_or("2025-03-26")
            .to_string();
        s.protocol = Some(agreed);
        s.notify("notifications/initialized", timeout)?;
        Ok(s)
    }

    /// One request, as JSON-RPC over the streamable endpoint. The reply is
    /// either JSON or an SSE stream; both end up as the same result value.
    pub fn request(&mut self, method: &str, params: Value, timeout: Duration) -> Result<Value> {
        self.next_id += 1;
        let id = self.next_id;
        let body = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        let resp = self.post(&body, timeout)?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().unwrap_or_default();
            bail!(
                "{} answered {}: {}",
                self.url,
                status.as_u16(),
                text.chars().take(200).collect::<String>()
            );
        }
        let ctype = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        if ctype.starts_with("text/event-stream") {
            sse_reply(resp, id, timeout)
        } else {
            let v: Value =
                resp.json().with_context(|| format!("{} sent a non-JSON reply", self.url))?;
            take_result(v)
        }
    }

    /// Fire a notification: no id, no reply expected. A `202 Accepted` or an
    /// empty body is success; anything else is ignored, because a notification
    /// must not be able to fail the session.
    fn notify(&mut self, method: &str, timeout: Duration) -> Result<()> {
        let body = json!({"jsonrpc": "2.0", "method": method, "params": {}});
        let _ = self.post(&body, timeout)?;
        Ok(())
    }

    fn post(&mut self, body: &Value, timeout: Duration) -> Result<reqwest::blocking::Response> {
        // The streamable spec: a client must say it can take either reply
        // shape, and servers check. The timeout bounds this request alone, so
        // one slow server cannot ride on a client-wide default.
        let mut req = self
            .http
            .post(&self.url)
            .header("Accept", "application/json, text/event-stream")
            .timeout(timeout)
            .json(body);
        for (k, v) in &self.headers {
            req = req.header(k, v);
        }
        if let Some(s) = &self.session {
            req = req.header("Mcp-Session-Id", s);
        }
        if let Some(p) = &self.protocol {
            req = req.header("MCP-Protocol-Version", p);
        }
        let resp = req.send().context(format!("POST {} failed", self.url))?;
        // A streamable server names the session in the response headers; the
        // first one to appear (on `initialize`) is the one to keep.
        if self.session.is_none() {
            if let Some(v) = resp.headers().get("mcp-session-id").and_then(|v| v.to_str().ok()) {
                self.session = Some(v.to_string());
            }
        }
        Ok(resp)
    }
}

/// Read an SSE body until the reply with `id` arrives, or time out. Events
/// carry their payload in `data:` lines; comments (`:`...) and field names we
/// do not need (`event`, `id`) are skipped.
fn sse_reply(resp: reqwest::blocking::Response, id: u64, timeout: Duration) -> Result<Value> {
    let (tx, rx): (std::sync::mpsc::Sender<Value>, Receiver<Value>) = channel();
    let deadline = std::time::Instant::now() + timeout;
    std::thread::spawn(move || {
        let reader = std::io::BufReader::new(resp);
        let mut data = String::new();
        for line in reader.lines() {
            let Ok(line) = line else { break };
            let line = line.trim_end();
            if let Some(rest) = line.strip_prefix("data:") {
                if !data.is_empty() {
                    data.push('\n');
                }
                data.push_str(rest.trim_start());
            } else if line.is_empty() {
                // End of one event: what accumulated in `data` is the message.
                if let Ok(v) = serde_json::from_str::<Value>(&data) {
                    if tx.send(v).is_err() {
                        break;
                    }
                }
                data.clear();
            }
        }
    });
    loop {
        let wait = deadline.saturating_duration_since(std::time::Instant::now());
        if wait.is_zero() {
            bail!("the SSE stream did not answer within {}s", timeout.as_secs());
        }
        match rx.recv_timeout(wait) {
            Ok(v) => {
                if v.get("id").and_then(|i| i.as_u64()) == Some(id) {
                    return take_result(v);
                }
                // Someone else's reply or a server-initiated request; the
                // stream keeps running and neither stops us.
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                bail!("the SSE stream did not answer within {}s", timeout.as_secs())
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                bail!("the SSE stream closed before the reply arrived")
            }
        }
    }
}

/// Pull `result` out of a reply, or turn an error object into one of ours.
fn take_result(v: Value) -> Result<Value> {
    if let Some(err) = v.get("error").filter(|e| !e.is_null()) {
        bail!(err["message"].as_str().unwrap_or(&err.to_string()).to_string());
    }
    Ok(v.get("result").cloned().unwrap_or(Value::Null))
}
