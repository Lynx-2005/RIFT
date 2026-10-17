//! Runs a command in its own process group, enforces a timeout, caps output.
use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

#[cfg(unix)]
use std::os::unix::process::CommandExt;

pub struct Outcome {
    pub text: String,
}

/// The shell tool commands run through, named in the system prompt so the
/// model writes syntax this shell understands.
#[cfg(unix)]
pub const SHELL: &str = "bash";

#[cfg(windows)]
pub const SHELL: &str = "powershell";

/// The command that starts the shell. Unix puts it in its own process group so
/// the whole tree can be killed at once; Windows has no process groups and
/// uses `taskkill /T` instead.
#[cfg(unix)]
fn shell_command(command: &str) -> Command {
    let mut cmd = Command::new("bash");
    cmd.arg("-c").arg(command);
    cmd.process_group(0);
    cmd
}

/// PowerShell without the user profile: cheap startup, and no alias of theirs
/// can change what a tool command does.
#[cfg(windows)]
fn shell_command(command: &str) -> Command {
    let mut cmd = Command::new("powershell");
    cmd.args(["-NoProfile", "-Command", command]);
    cmd
}

/// Kill the child and everything below it: the process group on Unix, the
/// process tree on Windows.
#[cfg(unix)]
fn kill_tree(child: &mut std::process::Child) {
    let pgid = child.id() as i32;
    unsafe {
        kill(-pgid, 9);
    }
}

#[cfg(windows)]
fn kill_tree(child: &mut std::process::Child) {
    let _ = Command::new("taskkill")
        .args(["/T", "/F", "/PID", &child.id().to_string()])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
}

/// `check_stop`: non-blocking callback to check if user pressed Esc.
/// If true is returned, kill the group and mark the output as stopped.
pub fn run(
    command: &str,
    cwd: &Path,
    timeout: Duration,
    max_output: usize,
    check_stop: impl Fn() -> bool,
) -> Outcome {
    let mut cmd = shell_command(command);
    cmd.current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            return Outcome { text: format!("failed to spawn {SHELL}: {e}") };
        }
    };

    let mut out = child.stdout.take().unwrap();
    let mut err = child.stderr.take().unwrap();
    let t_out = thread::spawn(move || {
        let mut b = Vec::new();
        let _ = out.read_to_end(&mut b);
        b
    });
    let t_err = thread::spawn(move || {
        let mut b = Vec::new();
        let _ = err.read_to_end(&mut b);
        b
    });

    let start = Instant::now();
    let (status, timed_out, stopped) = loop {
        // Check user cancel (Esc) periodically
        if check_stop() {
            kill_tree(&mut child);
            let _ = child.wait();
            break (None, false, true);
        }
        match child.try_wait() {
            Ok(Some(s)) => break (Some(s), false, false),
            Ok(None) if start.elapsed() > timeout => {
                kill_tree(&mut child);
                let _ = child.wait();
                break (None, true, false);
            }
            Ok(None) => thread::sleep(Duration::from_millis(15)),
            Err(_) => break (None, false, false),
        }
    };

    let stdout = join_with_deadline(t_out);
    let stderr = join_with_deadline(t_err);

    let mut text = String::new();
    if !stdout.is_empty() {
        text.push_str(&String::from_utf8_lossy(&stdout));
    }
    if !stderr.is_empty() {
        if !text.is_empty() && !text.ends_with('\n') { text.push('\n'); }
        text.push_str("[stderr]\n");
        text.push_str(&String::from_utf8_lossy(&stderr));
    }
    let exit = status.and_then(|s| s.code());
    if stopped {
        text.push_str("\n[stopped by user]");
    } else if timed_out {
        text.push_str(&format!("\n[killed: exceeded {}s timeout]", timeout.as_secs()));
    } else {
        text.push_str(&format!("\n[exit {}]", exit.map_or("signal".to_string(), |c| c.to_string())));
    }
    Outcome { text: truncate(text, max_output) }
}

fn join_with_deadline(h: thread::JoinHandle<Vec<u8>>) -> Vec<u8> {
    let deadline = Instant::now() + Duration::from_secs(2);
    while !h.is_finished() && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(10));
    }
    if h.is_finished() { h.join().unwrap_or_default() } else { Vec::new() }
}

/// Whether a command's output represents success. The runner's own `[exit N]`
/// trailer is the only authority: it is present exactly once, and a command
/// killed by timeout or Esc never wrote `[exit 0]`. Text is scanned rather than
/// a field because the output is also what the model is handed.
pub fn exit_ok(text: &str) -> bool {
    text.lines().rev().any(|l| l.trim() == "[exit 0]")
}

pub fn truncate(text: String, max: usize) -> String {
    if text.len() <= max { return text; }
    let half = max / 2;
    let mut head = half;
    while !text.is_char_boundary(head) { head -= 1; }
    let mut tail = text.len() - half;
    while !text.is_char_boundary(tail) { tail += 1; }
    format!(
        "{}\n\n[... {} bytes truncated ...]\n\n{}",
        &text[..head], tail - head, &text[tail..]
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exit_ok_reads_the_trailer_the_runner_appends() {
        assert!(exit_ok("out\n[exit 0]\n"));
        assert!(!exit_ok("out\n[exit 1]\n"));
        assert!(!exit_ok("out\n[exit 2]"));
        // A kill and a timeout report no exit code and are never success.
        assert!(!exit_ok("out\n[killed: exceeded 1s timeout]"));
        assert!(!exit_ok("out\n[stopped by user]"));
        // A spawn failure has no trailer at all.
        assert!(!exit_ok("failed to spawn bash: no such file"));
        assert!(!exit_ok(""));
        // The trailer is a whole line, not a substring anywhere in the text.
        assert!(!exit_ok("echo [exit 0] printed this\n[exit 1]"));
    }
}

#[cfg(unix)]
extern "C" {
    fn kill(pid: i32, sig: i32) -> i32;
}
