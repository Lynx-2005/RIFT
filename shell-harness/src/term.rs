//! Terminal layer: raw mode, size, resize and input.
//!
//! The OS-specific pieces live in `term::unix` (termios, SIGWINCH, poll/read
//! on fd 0) and `term::windows` (console modes, ReadFile on the console
//! handle); everything below is shared between them.

use std::io::Write;

#[cfg(unix)]
mod unix;
#[cfg(windows)]
mod windows;

#[cfg(unix)]
pub use unix::*;
#[cfg(windows)]
pub use windows::*;

/// stdin is a terminal: raw mode, the editor and the input watcher need it.
pub(crate) fn is_tty() -> bool {
    use std::io::IsTerminal;
    std::io::stdin().is_terminal()
}

/// Windows consoles can hand back `\r\n` for a single Enter; collapsing an LF
/// that directly follows a CR keeps Enter one key there. A lone LF (Ctrl-J)
/// still arrives as LF on its own.
#[cfg(any(windows, test))]
fn collapse_crlf(buf: &mut [u8]) -> usize {
    let mut w = 0;
    for r in 0..buf.len() {
        if buf[r] == b'\n' && w > 0 && buf[w - 1] == b'\r' {
            continue;
        }
        buf[w] = buf[r];
        w += 1;
    }
    w
}

/// Columns available for drawing, with a floor for terminals that refuse.
pub fn cols() -> usize {
    size().map_or(80, |(cols, _)| cols).max(20)
}

pub fn rows() -> usize {
    size().map_or(24, |(_, rows)| rows).max(4)
}

/// Where the watcher is inside a VT escape sequence.
///
/// A console read can stop anywhere inside a sequence: Windows hands back a
/// whole mouse report per input record (`ESC[<35;25;24M`), and a small read
/// buffer splits it. Skipping only the bytes of one read would leave the tail
/// (`5;24M`) to be treated as typed text, which is how mouse movement used to
/// type `;25M;24M;...` into the prompt. The state therefore lives across reads.
#[derive(Clone, Copy, PartialEq)]
enum Escape {
    None,
    /// `ESC` seen, waiting for the byte that says which sequence this is.
    Start,
    /// `ESC [` seen: parameter bytes, then one final byte in `0x40..=0x7e`.
    Csi,
    /// `ESC O` seen: exactly one final byte follows.
    Ss3,
}

impl Escape {
    /// Advance by one byte. Returns true when the byte is ordinary input.
    fn feed(&mut self, c: u8) -> bool {
        match *self {
            Escape::None if c == 0x1b => {
                *self = Escape::Start;
                false
            }
            Escape::None => true,
            Escape::Start => {
                *self = match c {
                    b'[' => Escape::Csi,
                    b'O' => Escape::Ss3,
                    // Alt-<key>: both bytes belong to the shortcut.
                    _ => Escape::None,
                };
                false
            }
            Escape::Csi => {
                if (0x40..=0x7e).contains(&c) {
                    *self = Escape::None;
                }
                false
            }
            Escape::Ss3 => {
                *self = Escape::None;
                false
            }
        }
    }
}

/// Watches stdin while the agent works. Esc / Ctrl-C cancel the turn; typing
/// edits the live line, Enter queues the line to run when the turn ends. Owns
/// stdin polling so tool execution just reads the flag instead of racing on
/// the fd. No-op without a TTY.
pub fn input_watcher(
    cancel: std::sync::Arc<std::sync::atomic::AtomicBool>,
    done: std::sync::Arc<std::sync::atomic::AtomicBool>,
    typed: std::sync::Arc<Typed>,
) {
    use std::sync::atomic::Ordering;
    if !is_tty() {
        return;
    }
    let mut buf = String::new();
    let mut pending: Vec<u8> = Vec::new();
    let mut esc = Escape::None;
    'watch: while !done.load(Ordering::Relaxed) && !cancel.load(Ordering::Relaxed) {
        if !poll_in(50) {
            continue;
        }
        // Larger than one escape sequence, so a mouse report or an arrow key
        // normally arrives whole; the state machine handles the rare split.
        let mut b = [0u8; 64];
        // A resize interrupts the read: nothing was typed, so poll again.
        let n = match read_stdin(&mut b) {
            Ok(n) => n,
            Err(_) => continue,
        };
        if n == 0 {
            continue;
        }
        for &c in &b[..n] {
            if !esc.feed(c) {
                continue;
            }
            match c {
                // Raw mode: Ctrl-C arrives as 0x03 (ISIG off).
                0x03 => {
                    cancel.store(true, Ordering::Relaxed);
                    break 'watch;
                }
                b'\r' | b'\n' => {
                    let line = buf.trim().to_string();
                    buf.clear();
                    if !line.is_empty() {
                        submit(&line, &typed, crate::ui::queued_note());
                    }
                    live_end();
                }
                0x7f | 0x08 => {
                    buf.pop();
                    live_key(&buf);
                }
                c if c >= 0x20 => {
                    // Decode what is complete: a read can split a multi-byte char.
                    pending.push(c);
                    match std::str::from_utf8(&pending) {
                        Ok(s) => {
                            buf.push_str(s);
                            pending.clear();
                            live_key(&buf);
                        }
                        Err(e) if e.error_len().is_some() => pending.clear(),
                        Err(_) => {}
                    }
                }
                _ => {}
            }
        }
        // The read ended between ESC and the rest of its sequence. A lone Esc
        // still cancels, so give the tail a moment to arrive before deciding.
        if esc == Escape::Start {
            if poll_in(30) {
                continue;
            }
            cancel.store(true, Ordering::Relaxed);
            break;
        }
    }
    // A turn that ends on a half-typed line keeps it instead of dropping it.
    let tail = buf.trim().to_string();
    if done.load(Ordering::Relaxed) && !tail.is_empty() {
        submit(&tail, &typed, crate::ui::queued_note());
    }
    live_end();
}

/// Lines typed while the agent works. The reader thread pushes them; the main
/// loop drains them when the current turn ends, so nothing typed is lost.
#[derive(Default)]
pub struct Typed {
    queue: std::sync::Mutex<Vec<String>>,
}

impl Typed {
    pub fn new() -> std::sync::Arc<Self> {
        std::sync::Arc::new(Self::default())
    }

    pub fn push(&self, line: String) {
        if let Ok(mut q) = self.queue.lock() {
            q.push(line);
        }
    }

    /// Oldest first: queued lines run in the order they were typed.
    pub fn take(&self) -> Option<String> {
        let mut q = self.queue.lock().ok()?;
        if q.is_empty() {
            None
        } else {
            Some(q.remove(0))
        }
    }
}

static LIVE_OPEN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static LIVE_DRAWN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static LIVE_BUF: std::sync::Mutex<String> = std::sync::Mutex::new(String::new());
static ROW: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Whole blocks of output take this, so a line never splices into the tokens
/// of the turn above it. Poisoning is ignored: a panic in one writer must not
/// take the terminal down with it.
pub fn row_lock() -> std::sync::MutexGuard<'static, ()> {
    ROW.lock().unwrap_or_else(|e| e.into_inner())
}

/// True while the row under the output belongs to the prompt: an open turn,
/// something typed, or a wait whose indicator lives there.
pub fn row_wanted() -> bool {
    crate::ui::turn_running()
        || LIVE_OPEN.load(std::sync::atomic::Ordering::Relaxed)
        || crate::ui::working()
}

/// A writer is about to print a full line: take the row back from the prompt.
pub fn live_take_row() {
    if LIVE_DRAWN.swap(false, std::sync::atomic::Ordering::Relaxed) {
        let mut o = std::io::stdout().lock();
        let _ = write!(o, "\r\x1b[K");
        let _ = o.flush();
    }
}

/// Draw the prompt row: the working indicator, if any, then the mark and
/// whatever has been typed. The single place that paints this row. Callers hold
/// the row lock: this is called from inside writers that already own it.
pub fn draw_live() {
    let buf = LIVE_BUF.lock().map(|b| b.clone()).unwrap_or_default();
    let mut o = std::io::stdout().lock();
    let _ = write!(o, "\r\x1b[K{}", crate::ui::live_line(&buf));
    let _ = o.flush();
    LIVE_DRAWN.store(true, std::sync::atomic::Ordering::Relaxed);
}

/// A writer just ended a line: put the prompt row back, under the new output.
pub fn live_reprint() {
    if !row_wanted() || LIVE_DRAWN.load(std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    draw_live();
}

fn live_key(buf: &str) {
    if let Ok(mut b) = LIVE_BUF.lock() {
        b.clear();
        b.push_str(buf);
    }
    LIVE_OPEN.store(true, std::sync::atomic::Ordering::Relaxed);
    let _row = row_lock();
    draw_live();
}

/// The line is over: submitted, or abandoned by a cancel.
fn live_end() {
    live_take_row();
    LIVE_OPEN.store(false, std::sync::atomic::Ordering::Relaxed);
    LIVE_DRAWN.store(false, std::sync::atomic::Ordering::Relaxed);
    if let Ok(mut b) = LIVE_BUF.lock() {
        b.clear();
    }
}

/// Commit a typed line: echo it in the "yours" role, say when it will run.
fn submit(line: &str, typed: &Typed, note: String) {
    let _row = row_lock();
    live_end();
    let mut o = std::io::stdout().lock();
    let _ = writeln!(o, "{}{}", crate::ui::echo_block(line), note);
    let _ = o.flush();
    drop(o);
    live_reprint();
    typed.push(line.to_string());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_lines_come_back_in_order() {
        let t = Typed::new();
        assert_eq!(t.take(), None);
        t.push("first line".into());
        t.push("second line".into());
        assert_eq!(t.take().as_deref(), Some("first line"));
        assert_eq!(t.take().as_deref(), Some("second line"));
        assert_eq!(t.take(), None);
    }

    #[test]
    fn an_open_turn_owns_the_prompt_row_on_its_own() {
        crate::ui::turn_begin();
        assert!(
            row_wanted(),
            "a turn in flight must keep the row without a spinner or typed text"
        );
        crate::ui::turn_close();
        assert!(!crate::ui::turn_running(), "the turn flag outlived the turn");
    }

    /// Feed chunks through one parser, as the watcher does across reads, and
    /// return the bytes it treats as typed text.
    fn kept(chunks: &[&[u8]]) -> Vec<u8> {
        let mut esc = Escape::None;
        let mut out = Vec::new();
        for chunk in chunks {
            for &c in *chunk {
                if esc.feed(c) {
                    out.push(c);
                }
            }
        }
        out
    }

    #[test]
    fn an_escape_sequence_split_anywhere_leaves_no_text_behind() {
        // A Windows console can end a read at any byte of a mouse report. Every
        // split point must still swallow the whole sequence: the tail used to
        // be typed into the prompt as literal `5;24M`.
        let report = b"\x1b[<35;25;24M";
        for cut in 0..=report.len() {
            assert!(
                kept(&[&report[..cut], &report[cut..]]).is_empty(),
                "split at byte {cut} leaked text"
            );
        }
        // Arrows, Home/End and Alt-<key> are sequences too, not text.
        assert!(kept(&[b"\x1b[A"]).is_empty());
        assert!(kept(&[b"\x1bOA"]).is_empty());
        assert!(kept(&[b"\x1bx"]).is_empty());
    }

    #[test]
    fn ordinary_bytes_around_a_sequence_survive() {
        assert_eq!(
            kept(&[b"hi", b"\x1b[<35;25;24M", b"there\r"]),
            b"hithere\r"
        );
        // A reset after a sequence is what keeps later typing working.
        assert_eq!(kept(&[b"\x1b[A", b"ok"]), b"ok");
    }

    #[test]
    fn crlf_collapses_to_a_single_enter() {
        let mut b = *b"a\r\nb";
        let n = collapse_crlf(&mut b);
        assert_eq!(&b[..n], b"a\rb");

        // A lone LF, and an LF that follows a collapsed CR, both survive.
        let mut b = *b"\n\r\n\n";
        let n = collapse_crlf(&mut b);
        assert_eq!(&b[..n], b"\n\r");

        let mut b = *b"plain";
        let n = collapse_crlf(&mut b);
        assert_eq!(&b[..n], b"plain");
    }
}
