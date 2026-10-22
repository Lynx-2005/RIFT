//! Terminal I/O: colors, prompt, Codex-style line input with a live `/` menu.
use crate::config::Config;
use std::io::{self, BufRead, IsTerminal, Write};

// The color system, one role per code. The terminal owns the canvas, surfaces
// and borders, so this app paints only what it can own: text at the default
// weight, and the roles below. Nothing here is ever disabled, so there is no
// disabled role.
//
// The vocabulary is plain ASCII on purpose. Box drawing, bullets and arrows
// look right in the editor and wrong in half the monospace fonts shipped, and
// the fix for that is never worth the second font it takes to render.
//   GREEN     the live action: the tool echo, the prompt mark, the line you
//             submitted, and the outcome of a command you typed
//   YELLOW    warnings, always with a `!`
//   RED       failures, always with an `error: ` prefix
//   DIM       the frame only: the output gutter, runner markers, reasoning,
//             table rules, limits and hints. Never the payload, so file
//             contents and code stay legible instead of grey.
//   BOLD      the wordmark and a line you typed
//   REVERSE   selection and focus: the highlighted menu row (the caret is the
//             terminal's own cursor)
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const RED: &str = "\x1b[31m";
const DIM: &str = "\x1b[2m";
const BOLD: &str = "\x1b[1m";
const RESET: &str = "\x1b[0m";
const REVERSE: &str = "\x1b[7m";

// A heading is dim: the words carry it, and dimming is the one weight that
// says "this is a label" without turning the line into decoration.
const H_HEAD: &str = DIM;

thread_local! {
    /// Set only by a test that wants a fixed answer. Thread-local on purpose:
    /// tests each own a thread, so one test cannot decide for another, and no
    /// real run ever sets it.
    static COLOR: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Whether to paint escape codes.
///
/// Under `cargo test`, fd 1 is still whatever terminal started the run, because
/// libtest captures `print!` output rather than the descriptor. Asking
/// `isatty` there makes the tests depend on where they were run: in a terminal
/// they got colour and the assertions about plain text failed, in a pipe they
/// passed. So a test run defaults to plain, and a test that is about colour
/// says so itself.
fn color() -> bool {
    if let Some(on) = COLOR.with(|c| c.get()) {
        return on;
    }
    !cfg!(test) && io::stdout().is_terminal()
}

/// Fix the colour answer for this thread until the guard is dropped.
#[cfg(test)]
fn color_fixed(on: bool) -> ColorFixed {
    COLOR.with(|c| c.set(Some(on)));
    ColorFixed
}

#[cfg(test)]
struct ColorFixed;

#[cfg(test)]
impl Drop for ColorFixed {
    fn drop(&mut self) {
        COLOR.with(|c| c.set(None));
    }
}

fn paint(c: &str, s: &str) -> String {
    if color() { format!("{c}{s}{RESET}") } else { s.to_string() }
}

/// The mark in front of a line you typed. Kept as a glyph on purpose: it is
/// the one character that says "yours", and both fonts this has shipped
/// against carry it.
const PROMPT: &str = "❯";

/// The tool echo's dot and the corner its output hangs off. The dot is the one
/// cell that carries colour: green when the command exited 0, red when it did
/// not. The command text next to it stays the terminal's own weight, so the
/// outcome is a single cell rather than a repainted line.
const TOOL: &str = "●";
const OUT: &str = "└";

/// Marks a reasoning block, in words as well as in weight: deliberation must
/// never read as the answer.
const THINK_MARK: &str = "[thinking] ";

thread_local! {
    /// Reasoning deltas, held until a line is whole, plus whether the
    /// `[thinking]` marker still has to ride on the next line out.
    static THINK: std::cell::RefCell<(String, bool)> = const { std::cell::RefCell::new((String::new(), false)) };
}

/// Break `s` into rows the terminal will not wrap on its own: a row the app
/// prints must occupy exactly one app row, or the writers that move the cursor
/// (the prompt row, a landing block) count rows the terminal folded
/// differently. Breaks land on a space where there is one, so prose still
/// reads; an answer line that fits is passed through untouched.
fn display_rows(s: &str) -> Vec<String> {
    display_rows_at(s, crate::term::cols().saturating_sub(1).max(10))
}

/// The same, at a width the test names.
fn display_rows_at(s: &str, width: usize) -> Vec<String> {
    let width = width.max(10);
    let mut out = Vec::new();
    for line in s.split('\n') {
        let c: Vec<char> = line.chars().collect();
        if c.len() <= width {
            out.push(line.to_string());
            continue;
        }
        let mut i = 0usize;
        while i < c.len() {
            let limit = (i + width).min(c.len());
            // Break at the last space in reach; consume it so the
            // continuation does not open with a stray blank.
            let e = if limit == c.len() {
                limit
            } else {
                (i + 1..limit).rev().find(|&k| c[k] == ' ').unwrap_or(limit)
            };
            let row: String = c[i..e].iter().collect();
            out.push(row.trim_end().to_string());
            i = if e < c.len() && c[e] == ' ' { e + 1 } else { e };
        }
    }
    if out.is_empty() {
        out.push(String::new());
    }
    out
}

/// Write one finished line of streamed output. Whole lines only, so the prompt
/// row can be moved down under each one instead of being erased per token:
/// that is what keeps the prompt on screen, and typable, during a turn.
fn stream_line(s: &str) {
    if s.trim().is_empty() {
        return;
    }
    let _row = crate::term::row_lock();
    for row in display_rows(s) {
        crate::term::live_take_row();
        println!("{row}");
    }
    crate::term::live_reprint();
}

/// Print streamed answer text. The renderer hands over whole lines, so this is
/// the same discipline every other writer uses.
pub fn stream_text(r: &str) {
    let s = r.trim_end_matches('\n');
    if s.is_empty() {
        return;
    }
    let _row = crate::term::row_lock();
    for row in display_rows(s) {
        crate::term::live_take_row();
        println!("{row}");
    }
    crate::term::live_reprint();
}

/// Close the stream: flush the reasoning tail, then put the prompt row back.
pub fn stream_end() {
    think_flush();
    let _row = crate::term::row_lock();
    crate::term::live_reprint();
}

/// Open a reasoning block. Deliberation must never read as the answer, so the
/// block is marked in words as well as in weight.
pub fn think_start() {
    THINK.with(|t| t.borrow_mut().1 = true);
}

/// Thinking deltas: dim, written a line at a time so reasoning streams without
/// taking the row the prompt lives on. A long unbroken line flushes at the
/// window edge, otherwise nothing would appear until the model broke it.
pub fn stream_think(d: &str) {
    let width = crate::term::cols().max(20);
    let (lines, marker) = THINK.with(|t| {
        let mut t = t.borrow_mut();
        t.0.push_str(d);
        let mut lines: Vec<String> = Vec::new();
        while let Some(i) = t.0.find('\n') {
            lines.push(t.0[..i].to_string());
            t.0.drain(..=i);
        }
        // An unbroken line flushes at the window edge. The first line also
        // carries the marker, so it has that much less room.
        let mut avail = if t.1 { width.saturating_sub(THINK_MARK.len()) } else { width };
        while t.0.chars().count() >= avail.max(20) {
            let cut = avail.max(20);
            let rest: String = t.0.chars().skip(cut).collect();
            let line: String = t.0.chars().take(cut).collect();
            t.0 = rest;
            lines.push(line);
            avail = width;
        }
        // The marker is only spent when a line is actually going out.
        let marker = if lines.is_empty() { false } else { std::mem::take(&mut t.1) };
        (lines, marker)
    });
    for (n, line) in lines.iter().enumerate() {
        let marked = n == 0 && marker;
        stream_line(&if marked {
            format!("{}{}", paint(DIM, THINK_MARK), paint(DIM, line))
        } else {
            paint(DIM, line)
        });
    }
}

/// The remainder of the reasoning block, as its own line.
fn think_flush() {
    let (tail, marker) = THINK.with(|t| {
        let mut t = t.borrow_mut();
        (std::mem::take(&mut t.0), std::mem::take(&mut t.1))
    });
    if tail.trim().is_empty() && !marker {
        return;
    }
    stream_line(&if marker {
        format!("{}{}", paint(DIM, THINK_MARK), paint(DIM, &tail))
    } else {
        paint(DIM, &tail)
    });
}

/// A submitted line, echoed back under the prompt mark. No working indicator:
/// the line is history now, not something still being typed.
pub fn echo_line(buf: &str) -> String {
    format!("{} {}", paint(GREEN, PROMPT), paint(BOLD, buf))
}

/// A typed newline, shown as a visible break instead of a raw `\n`: a raw one
/// would put the rest of the buffer on a row the redraw arithmetic does not
/// know about. The idle prompt wraps real newlines; this is the mid-turn row,
/// which is one row by contract.
const BREAK: &str = " \u{21b5} ";

/// One row's worth of a buffer with its newlines made visible.
fn one_line(buf: &str) -> String {
    buf.replace('\n', BREAK)
}

/// A submitted line as it goes back on screen: the mark on the first row, the
/// continuation rows aligned under it. A prompt that was typed over three lines
/// is printed over three lines, because the transcript should show what was
/// sent rather than a joined-up version of it.
pub fn echo_block(buf: &str) -> String {
    let mut out = String::new();
    for (i, line) in buf.split('\n').enumerate() {
        let text = if i == 0 {
            echo_line(line)
        } else {
            format!("  {}", paint(BOLD, line))
        };
        out.push_str("\r\x1b[K");
        out.push_str(&text);
        out.push('\n');
    }
    out
}

/// Cells a painted string occupies, ignoring the SGR escapes `paint` emits.
fn visible_len(s: &str) -> usize {
    let mut n = 0usize;
    let mut it = s.chars();
    while let Some(c) = it.next() {
        if c == '\x1b' {
            for d in it.by_ref() {
                if d.is_ascii_alphabetic() {
                    break;
                }
            }
            continue;
        }
        n += 1;
    }
    n
}

/// The prompt mark and a line in the "yours" role. Shared by the idle prompt
/// and the live line, so a line typed mid-turn looks like any other line.
/// While a wait is running the animated indicator sits in front of the mark,
/// which is why the prompt never disappears mid-turn: it becomes the status row.
///
/// The row is clamped to the window: it is repainted in place with a carriage
/// return, so a line wider than the terminal would wrap and every redraw would
/// start on the wrong row and scroll the screen instead of overwriting it.
pub fn live_line(buf: &str) -> String {
    let status = live_status();
    let reserved = status.as_ref().map_or(0, |s| visible_len(s) + 2) + 2;
    let room = crate::term::cols().saturating_sub(reserved).max(4);
    let shown = clamp(&one_line(buf), room);
    match status {
        Some(s) => format!("{s}  {}", echo_line(&shown)),
        None => echo_line(&shown),
    }
}

/// True while a wait is running: the row below the output carries the indicator.
pub fn working() -> bool {
    RUNNING.load(std::sync::atomic::Ordering::Relaxed) > 0
}

/// A turn keeps the prompt row for its whole length, whether or not a wait is
/// running and whether or not anything has been typed. Without this the row
/// vanished the moment the model started thinking, because the stream's first
/// token stops the spinner, and stopping the spinner erases an unwanted row.
pub fn turn_begin() {
    TURN_OPEN.store(true, std::sync::atomic::Ordering::Relaxed);
    let _row = crate::term::row_lock();
    crate::term::live_reprint();
}

pub fn turn_close() {
    TURN_OPEN.store(false, std::sync::atomic::Ordering::Relaxed);
}

pub fn turn_running() -> bool {
    TURN_OPEN.load(std::sync::atomic::Ordering::Relaxed)
}

/// The animated prefix: a frame, what the wait is doing, and for how long.
/// `None` when nothing is running. Reduced motion keeps the words, drops the
/// animation, so a still `-` says the same thing a spinning frame does.
fn live_status() -> Option<String> {
    if !working() {
        return None;
    }
    let (label, since) = WORK.lock().ok()?.clone()?;
    // Reduced motion keeps the words and drops the movement: the first frame
    // is a still glyph that still says "working".
    let frame = if QUIET.load(std::sync::atomic::Ordering::Relaxed) {
        frames()[0].to_string()
    } else {
        frame_at(FRAME_IDX.load(std::sync::atomic::Ordering::Relaxed)).to_string()
    };
    Some(format!(
        "{} {}{}",
        paint(DIM, &frame),
        paint(DIM, &label),
        paint(DIM, &elapsed_suffix(since.elapsed()))
    ))
}

/// Says out loud that a line typed mid-turn has not run yet.
pub fn queued_note() -> String {
    paint(DIM, "  queued, runs when this turn ends")
}

/// Print a full line without eating the prompt row: take the row, print, put
/// the row back underneath. Every writer that opens a line uses this.
fn row(s: &str) {
    let _row = crate::term::row_lock();
    crate::term::live_take_row();
    println!("{s}");
    crate::term::live_reprint();
}

/// The one-liner under the wordmark. It is a promise, so it is the one line
/// here that keeps the terminal's own weight; `>_` is the shell mark, green so
/// it reads as the live cursor that wrote it.
const TAGLINE: &str = "your personal hacker living in terminal";

/// The static shell mark in front of the tagline.
const TAGLINE_MARK: &str = ">_";

/// Glyphs the reveal uses before a character locks. ASCII on purpose: every
/// monospace font carries these, and the line stays one cell per character on
/// terminals where box drawing and braille fall back to a proportional font.
const CIPHER: &[u8] = b"#%&*+=-/\\<>?01{}[]$@";

/// Steps in the rift reveal. Each step locks one slice in the middle, so the
/// line opens outward from its centre instead of typing left to right.
const TAGLINE_STEPS: usize = 7;

/// Hold time per reveal step. The whole flourish is about 100ms: long enough
/// to read as motion, short enough that it stays inside the interval where a
/// terminal action still feels instantaneous.
const TAGLINE_STEP_MS: u64 = 10;

/// The phrase holds green for one beat when it locks, then settles.
const TAGLINE_FLASH_MS: u64 = 25;

/// The arrival line. Segments fall off the end when the window is narrow,
/// rather than wrapping: it is a marker, and a wrapped marker reads as damage.
/// The number of enabled MCP servers sits next to the model, so the state is
/// visible without a sentence printed above the wordmark.
///
/// Returns the raw-mode guard when the tagline animated. The reader holds it
/// for the first prompt, so keys typed during the reveal are queued in raw
/// mode rather than echoed onto the animation row and then flushed.
pub fn banner(cfg: &Config, mcp_active: usize) -> Option<crate::term::RawGuard> {
    let width = crate::term::cols().saturating_sub(1);
    // Hold the row lock across the wordmark and the reveal. A background
    // catalog refresh can warn at any moment, and a warning printed between two
    // frames would splice into the animation row; holding the lock makes the
    // header and tagline one block, with any warning after it.
    let raw = {
        let _row = crate::term::row_lock();
        println!("{}", header_line(cfg.provider.name(), &cfg.model, mcp_active, width));
        tagline()
    };
    if cfg.api_key.is_empty() && cfg.provider.requires_key() {
        warn(&format!("set {} or run /provider to switch", cfg.provider.key_env()));
    }
    raw
}

/// The wordmark row, built without printing so its layout can be asserted.
fn header_line(provider: &str, model: &str, mcp_active: usize, width: usize) -> String {
    let mut plain = String::from("rift - shell agent");
    let mut line = format!("{} {}", paint(BOLD, "rift"), paint(DIM, "- shell agent"));
    for seg in [provider, model] {
        let next = format!("{plain} - {seg}");
        if next.chars().count() > width {
            break;
        }
        plain = next;
        line.push_str(&paint(DIM, &format!(" - {seg}")));
    }
    // The count is the last thing on the row and the first thing dropped: the
    // model is what the user chose, while this is a glanceable status. Keep it
    // beside the model instead of padding to the terminal edge, which strands
    // the badge in empty space on a wide window.
    let status = format!("mcp {mcp_active}");
    if mcp_active > 0 && plain.chars().count() + 3 + status.chars().count() <= width {
        line.push_str(&paint(DIM, " - mcp "));
        line.push_str(&paint(GREEN, &mcp_active.to_string()));
    }
    line
}

/// The settled tagline. `text` is already clamped to the window.
fn tagline_final(text: &str) -> String {
    format!("{} {}", paint(GREEN, TAGLINE_MARK), text)
}

/// Print the tagline, animated on a real terminal and static everywhere else.
/// Reduced motion (`--no-spinner`) and pipes get the final line immediately, so
/// the flourish can never be the reason a script or an accessibility setup
/// waits on rift.
///
/// The caller holds the row lock for the whole reveal, sleeps included, so a
/// background warning cannot print between frames and splice into the row.
///
/// The animation runs in raw mode: echo is off, so a keystroke cannot land in
/// the middle of the frame, and anything typed is left in the input queue for
/// the first prompt. The guard goes back to the caller instead of being
/// dropped, because dropping it would restore cooked mode and the reader would
/// have to enter raw mode again, flushing exactly those keys.
fn tagline() -> Option<crate::term::RawGuard> {
    let width = crate::term::cols().saturating_sub(1);
    let text = clamp(TAGLINE, width.saturating_sub(TAGLINE_MARK.len() + 1));
    let animate = !cfg!(test)
        && crate::term::is_tty()
        && io::stdout().is_terminal()
        && !QUIET.load(std::sync::atomic::Ordering::Relaxed);
    if !animate {
        println!("{}", tagline_final(&text));
        return None;
    }
    // No raw mode means echo would paint over the animation. A still tagline
    // is better than a line the user's first keystroke can corrupt.
    let raw = crate::term::RawGuard::new();
    if raw.is_none() {
        println!("{}", tagline_final(&text));
        return None;
    }
    // A different seed per launch means the same reveal never looks canned.
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
        ^ (u64::from(std::process::id()) << 32);
    let mut out = io::stdout().lock();
    for step in 0..=TAGLINE_STEPS {
        let line = format!(
            "{} {}",
            paint(GREEN, TAGLINE_MARK),
            tagline_frame(&text, step, TAGLINE_STEPS, seed)
        );
        // The row is ours until the newline below: the banner holds the row
        // lock across the whole reveal, so background writers wait rather than
        // splicing a warning into a frame.
        let _ = write!(out, "\r\x1b[K{line}");
        let _ = out.flush();
        if step < TAGLINE_STEPS {
            std::thread::sleep(std::time::Duration::from_millis(TAGLINE_STEP_MS));
        }
    }
    // One green beat when the whole phrase is in place, then the quiet settle.
    let flash = format!("{} {}", paint(GREEN, TAGLINE_MARK), paint(GREEN, &text));
    let _ = write!(out, "\r\x1b[K{flash}");
    let _ = out.flush();
    std::thread::sleep(std::time::Duration::from_millis(TAGLINE_FLASH_MS));
    let _ = writeln!(out, "\r\x1b[K{}", tagline_final(&text));
    let _ = out.flush();
    raw
}

/// The reveal window for `n` characters at `step` of `steps`: the number of
/// characters locked, as a half-open range growing from the middle out.
fn reveal_range(n: usize, step: usize, steps: usize) -> (usize, usize) {
    if n == 0 || steps == 0 || step == 0 {
        return (0, 0);
    }
    let locked = n * step.min(steps) / steps;
    let center = n / 2;
    let lo = center.saturating_sub(locked / 2);
    (lo, (lo + locked).min(n))
}

/// One frame: spaces stay put, unlocked cells are cipher noise, locked cells
/// carry the real text, and the two edges of the rift are green. At the last
/// step every cell is locked and the line is the plain tagline.
fn tagline_frame(text: &str, step: usize, steps: usize, seed: u64) -> String {
    let total = text.chars().filter(|c| !c.is_whitespace()).count();
    let (lo, hi) = reveal_range(total, step, steps);
    let mut out = String::new();
    let mut rank = 0usize;
    for ch in text.chars() {
        if ch.is_whitespace() {
            out.push(ch);
            continue;
        }
        if rank >= lo && rank < hi {
            let edge = (rank == lo && lo > 0) || (rank + 1 == hi && hi < total);
            if edge {
                out.push_str(&paint(GREEN, &ch.to_string()));
            } else {
                out.push(ch);
            }
        } else {
            out.push_str(&paint(DIM, &cipher(rank, step, seed).to_string()));
        }
        rank += 1;
    }
    out
}

/// A deterministic cipher glyph for a cell and frame. A small integer hash is
/// enough: no rng crate, no shared state, and a test can pin the seed.
fn cipher(slot: usize, frame: usize, seed: u64) -> char {
    let mut x = seed
        ^ (slot as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (frame as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    CIPHER[(x % CIPHER.len() as u64) as usize] as char
}

/// The model's answer, printed as plain text.
///
/// This deliberately renders nothing. Assistant text arrives as markdown: `##`
/// headings, `**bold**`, backtick spans, `|` tables, `---` rules. Rendering it
/// means a second type system squeezed out of ANSI, and the result reads as
/// decoration in a terminal where the content already has the user's own font
/// weight to work with. So markers are stripped instead of styled: the sentence
/// survives, the punctuation does not.
///
/// A fence is the one exception, because its contents are code and code is not
/// prose. The fence lines go; the block between them is passed through byte for
/// byte, including indentation and blank lines.
pub struct Md {
    buf: String,
    fence: bool,
    table: Vec<String>,
}

impl Md {
    pub fn new() -> Self {
        Md { buf: String::new(), fence: false, table: Vec::new() }
    }

    /// Feed a streamed delta; returns rendered complete lines (may be empty).
    /// Lines are the render unit, so markers split across chunks still work.
    /// Table rows are buffered: columns can only be sized once the block ends.
    pub fn push(&mut self, d: &str) -> String {
        self.buf.push_str(d);
        let mut out = String::new();
        while let Some(i) = self.buf.find('\n') {
            let line: String = self.buf.drain(..=i).collect();
            let line = line.trim_end_matches('\n');
            if is_table_row(line, self.fence) {
                self.table.push(line.to_string());
                continue;
            }
            out.push_str(&self.take_table());
            out.push_str(&render_line(line, &mut self.fence, color()));
            out.push('\n');
        }
        out
    }

    /// Render the trailing partial line (no newline appended).
    pub fn flush(&mut self) -> String {
        let mut out = String::new();
        if !self.buf.is_empty() {
            let line = std::mem::take(&mut self.buf);
            if is_table_row(&line, self.fence) {
                self.table.push(line);
            } else {
                out.push_str(&self.take_table());
                out.push_str(&render_line(&line, &mut self.fence, color()));
            }
        }
        out.push_str(&self.take_table());
        out
    }

    fn take_table(&mut self) -> String {
        if self.table.is_empty() {
            return String::new();
        }
        render_table(&std::mem::take(&mut self.table), color())
    }
}

fn style(c: &str, s: &str, color: bool) -> String {
    if color { format!("{c}{s}{RESET}") } else { s.to_string() }
}

/// Re-add a table as aligned plain text.
///
/// A `|` table is structure, and structure is worth keeping: the columns are
/// the point. The rules become `-`, the padding does the aligning, and no row
/// ever wraps. A separate renderer is needed because columns can only be sized
/// once the whole block has been read.
fn is_table_row(line: &str, fence: bool) -> bool {
    !fence && line.trim_start().starts_with('|')
}

fn is_separator(cells: &[String]) -> bool {
    !cells.is_empty() && cells.iter().all(|c| c.chars().all(|ch| ch == '-' || ch == ':'))
}

fn table_cells(line: &str) -> Vec<String> {
    let mut cells: Vec<String> = line.split('|').map(str::trim).map(str::to_string).collect();
    if cells.first().is_some_and(String::is_empty) {
        cells.remove(0);
    }
    if cells.last().is_some_and(String::is_empty) {
        cells.pop();
    }
    cells
}

fn is_valid_sep(line: &str) -> bool {
    is_separator(&table_cells(line))
}

fn render_table(rows: &[String], color: bool) -> String {
    let parsed: Vec<(bool, Vec<String>)> =
        rows.iter().map(|l| (is_valid_sep(l), table_cells(l))).collect();
    let cols = parsed.iter().map(|(_, c)| c.len()).max().unwrap_or(0);
    if cols == 0 {
        return String::new();
    }
    let plain: Vec<Vec<String>> = parsed
        .iter()
        .map(|(sep, cells)| {
            (0..cols)
                .map(|i| {
                    if *sep {
                        String::new()
                    } else {
                        strip_inline(cells.get(i).map_or("", String::as_str))
                    }
                })
                .collect()
        })
        .collect();
    let mut widths = vec![0usize; cols];
    for row in &plain {
        for (i, cell) in row.iter().enumerate() {
            widths[i] = widths[i].max(cell.chars().count());
        }
    }
    let header = parsed.iter().position(|(sep, _)| !*sep);
    let mut out = String::new();
    for (r, (sep, _)) in parsed.iter().enumerate() {
        let mut line = String::new();
        for i in 0..cols {
            if i > 0 {
                line.push_str("  ");
            }
            if *sep {
                line.push_str(&"-".repeat(widths[i].max(1)));
                continue;
            }
            let cell = &plain[r][i];
            let pad = widths[i].saturating_sub(cell.chars().count());
            line.push_str(cell);
            if i + 1 < cols {
                line.push_str(&" ".repeat(pad));
            }
        }
        if *sep {
            out.push_str(&style(DIM, &line, color));
        } else if header == Some(r) {
            out.push_str(&style(BOLD, &line, color));
        } else {
            out.push_str(&line);
        }
        out.push('\n');
    }
    out
}

/// Remove markdown's decoration from one line, keeping its words. Styling is
/// only ever added where it marks a role this UI owns (a heading's weight, a
/// rule's dim), never to reproduce the model's formatting.
fn render_line(line: &str, fence: &mut bool, color: bool) -> String {
    let t = line.trim_start();
    if t.starts_with("```") {
        *fence = !*fence;
        return String::new(); // the fence line itself is scaffolding
    }
    if *fence {
        // Code keeps the terminal's own foreground and its exact text.
        return line.to_string();
    }
    // Horizontal rules are punctuation with no words behind them. A heading
    // keeps its words and is dimmed, so structure survives without a glyph.
    if is_rule(t) {
        return String::new();
    }
    let hashes = t.bytes().take_while(|&b| b == b'#').count();
    if (1..=6).contains(&hashes) && t[hashes..].starts_with(' ') {
        let body = strip_inline(t[hashes..].trim());
        return style(H_HEAD, &body, color);
    }
    let indent = line.len() - t.len();
    // `*` and `+` bullets become `- `. The single marker is a decision; the
    // model reaching for the other two must not leave them on screen.
    let body = if t.starts_with("* ") || t.starts_with("+ ") {
        format!("- {}", &t[2..])
    } else {
        t.to_string()
    };
    let body = if let Some(rest) = body.strip_prefix("> ") {
        // A quote keeps a marker that is already ASCII and already means this.
        format!("> {rest}")
    } else {
        body
    };
    format!("{}{}", " ".repeat(indent), strip_inline(&body))
}

/// A markdown horizontal rule: three or more of the same rule character, with
/// nothing else on the line.
fn is_rule(t: &str) -> bool {
    let mut it = t.chars();
    let Some(first) = it.next() else { return false };
    if !matches!(first, '-' | '*' | '_') {
        return false;
    }
    it.all(|c| c == first || c == ' ')
        && t.chars().filter(|&c| c == first).count() >= 3
}

/// Remove inline markers, keeping only what they wrapped: `**bold**` -> the
/// words, `` `code` `` -> the words, `[text](url)` -> text. An unmatched marker
/// is left alone, because then it is text the model meant literally.
fn strip_inline(s: &str) -> String {
    let b: Vec<char> = s.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        // Code span: the contents are literal, so they are kept verbatim.
        if c == '`' {
            if let Some(j) = find_from(&b, i + 1, '`') {
                out.extend(&b[i + 1..j]);
                i = j + 1;
                continue;
            }
            out.push(c);
            i += 1;
            continue;
        }
        // `*` is never printed. `**bold**`, `*italic*` and a bare `*` all lose
        // the asterisk: it is markdown's emphasis marker, and this surface
        // carries no emphasis. `2 * 3` keeps only the arithmetic, which is the
        // right reading of a multiplication sign in a sentence.
        if c == '*' {
            // Where the asterisk was padded by spaces, one space goes too, so
            // removing a multiplication sign does not leave a double gap.
            if out.ends_with(' ') && b.get(i + 1) == Some(&' ') {
                i += 1;
            }
            i += 1;
            continue;
        }
        // `[text](url)`: keep `text`, drop the address.
        if c == '[' {
            if let Some(close) = find_from(&b, i + 1, ']') {
                if b.get(close + 1) == Some(&'(') {
                    if let Some(end) = find_from(&b, close + 2, ')') {
                        out.push_str(&strip_inline(&b[i + 1..close].iter().collect::<String>()));
                        i = end + 1;
                        continue;
                    }
                }
            }
            out.push(c);
            i += 1;
            continue;
        }
        out.push(c);
        i += 1;
    }
    out
}

/// Index of the next `c` at or after `from`.
fn find_from(b: &[char], from: usize, c: char) -> Option<usize> {
    (from..b.len()).find(|&i| b[i] == c)
}



/// Spinner on its own thread while a wait runs. It holds still for the first
/// 250ms, so a call that beats the clock never flashes a line; after that it
/// animates on a short cycle and adds elapsed time once the wait is long enough
/// to wonder about. Quiet motion prints the label once, unanimated.
/// TTY only; pipes get nothing.
pub struct Spinner {
    stop: Option<std::sync::mpsc::Sender<()>>,
    handle: Option<std::thread::JoinHandle<()>>,
}

static QUIET: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Reduced motion for a terminal: waits still announce themselves, the
/// animation stops. Set once at startup.
pub fn quiet_motion(on: bool) {
    QUIET.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// The wait indicator's frames: the quarter and half filled circles, ordered
/// the way the filled part travels around the dial, left then upper-left then
/// upper and so on back to lower-left. Eight frames read as one continuous
/// turn where the usual four read as a stutter, and each is a single cell in
/// the monospace fonts people actually run.
///
/// Braille is deliberately not used, though `⠋⠙⠹⠸` is the most copied spinner
/// on GitHub. No monospace font on a stock Linux box carries U+28xx, so those
/// cells get served from a proportional fallback font, and a proportional
/// glyph inside a monospace row breaks the column arithmetic the prompt row
/// is built on. This project has been bitten by exactly that once already.
const FRAMES: [&str; 8] = ["◐", "◴", "◓", "◷", "◑", "◶", "◒", "◵"];

/// The same rotation for a terminal that is not UTF-8. These four are ASCII,
/// so no font can fail to draw them.
const FRAMES_ASCII: [&str; 4] = ["-", "\\", "|", "/"];

/// Whether a locale name promises UTF-8. Pure, so the decision can be tested
/// without mutating the environment of a process whose tests run in parallel.
#[cfg(any(not(windows), test))]
fn locale_is_utf8(name: &str) -> bool {
    let v = name.to_ascii_uppercase();
    v.contains("UTF-8") || v.contains("UTF8")
}

#[cfg(windows)]
fn utf8_locale() -> bool {
    // The console output code page is forced to UTF-8 at startup, so the
    // multi-byte frames render as single cells.
    true
}

/// Whether the terminal is expected to render multi-byte glyphs as single
/// cells. A UTF-8 locale is the signal for that; without one, the three bytes
/// of `◐` may be drawn as three separate cells and the row comes apart.
#[cfg(not(windows))]
fn utf8_locale() -> bool {
    ["LC_ALL", "LC_CTYPE", "LANG"]
        .iter()
        .filter_map(|k| std::env::var(k).ok())
        .any(|v| locale_is_utf8(&v))
}

/// The frame set for a terminal with this locale answer.
fn frames_for(utf8: bool) -> &'static [&'static str] {
    if utf8 { &FRAMES } else { &FRAMES_ASCII }
}

/// The frame set for this terminal.
fn frames() -> &'static [&'static str] {
    frames_for(utf8_locale())
}

/// The frame to draw for a tick from a given set. Wraps, so the caller can
/// count forever. Takes the set explicitly so the sequence is testable without
/// the process environment in the way.
fn frame_in(set: &[&'static str], tick: usize) -> &'static str {
    set[tick % set.len()]
}

/// The frame to draw for a tick in this terminal.
fn frame_at(tick: usize) -> &'static str {
    frame_in(frames(), tick)
}

/// How long each frame is held. Eight frames at this rate is one full turn
/// every 640ms: fast enough to read as motion while waiting, slow enough that
/// the eye can follow the direction of the turn instead of seeing a flicker.
const FRAME_MS: u64 = 80;

/// Waits in flight, the frame being shown, and the label of the current one.
static RUNNING: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static TURN_OPEN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static FRAME_IDX: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static WORK: std::sync::Mutex<Option<(String, std::time::Instant)>> = std::sync::Mutex::new(None);

/// Elapsed time, once a wait is long enough to wonder about.
fn elapsed_suffix(d: std::time::Duration) -> String {
    let secs = d.as_secs();
    if secs < 2 {
        return String::new();
    }
    if secs < 60 {
        return format!(" ({secs}s)");
    }
    format!(" ({}m{:02}s)", secs / 60, secs % 60)
}

impl Spinner {
    pub fn start(label: &str) -> Self {
        // A test gets no indicator: on a terminal it would spawn a paint thread
        // that scribbles over the real output and over the spinner state its
        // neighbours assert on.
        if cfg!(test) || !io::stdout().is_terminal() {
            return Spinner { stop: None, handle: None };
        }
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        let label = clamp(label, crate::term::cols().saturating_sub(10));
        let quiet = QUIET.load(std::sync::atomic::Ordering::Relaxed);
        RUNNING.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if let Ok(mut w) = WORK.lock() {
            *w = Some((label, std::time::Instant::now()));
        }
        let handle = std::thread::spawn(move || {
            if rx.recv_timeout(std::time::Duration::from_millis(250)).is_ok() {
                return;
            }
            if quiet {
                // Reduced motion: the words land once, nothing moves.
                let _row = crate::term::row_lock();
                crate::term::draw_live();
                drop(_row);
                let _ = rx.recv();
                return;
            }
            let mut i = 0usize;
            loop {
                // The frame advances on the prompt row itself, so a wait and a
                // half-typed line share one row instead of fighting for it.
                FRAME_IDX.store(i, std::sync::atomic::Ordering::Relaxed);
                {
                    let _row = crate::term::row_lock();
                    crate::term::draw_live();
                }
                i += 1;
                if rx.recv_timeout(std::time::Duration::from_millis(FRAME_MS)).is_ok() {
                    break;
                }
            }
        });
        Spinner { stop: Some(tx), handle: Some(handle) }
    }

    /// Rename the wait without ending it. Streaming used to stop the spinner on
    /// its first token, which took the whole row away; the indicator now stays,
    /// and only its word changes.
    pub fn relabel(&mut self, label: &str) {
        if self.stop.is_none() {
            return; // inert: not a terminal
        }
        let label = clamp(label, crate::term::cols().saturating_sub(10));
        if let Ok(mut w) = WORK.lock() {
            if let Some((l, _)) = w.as_mut() {
                *l = label;
            }
        }
    }

    pub fn stop(mut self) {
        if self.stop.is_none() && self.handle.is_none() {
            return;
        }
        if let Some(tx) = self.stop.take() {
            let _ = tx.send(());
        }
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
        RUNNING.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        if let Ok(mut w) = WORK.lock() {
            *w = None;
        }
        // With the wait over the row is still a prompt: repaint it without the
        // indicator, or hand it back to the output when nothing is left there.
        let _row = crate::term::row_lock();
        if crate::term::row_wanted() {
            crate::term::draw_live();
        } else {
            crate::term::live_take_row();
        }
    }
}

/// The tool echo, with the outcome in the dot and nowhere else.
fn tool_echo(cmd: &str, ok: bool) -> String {
    let dot = if ok { paint(GREEN, TOOL) } else { paint(RED, TOOL) };
    format!("{dot} {cmd}")
}

/// A command the agent ran, once its outcome is known.
pub fn command(cmd: &str, ok: bool) {
    row(&tool_echo(cmd, ok));
}

/// Tool output is for skimming: head and tail are shown, the middle summarized.
/// The tail matters as much as the head: the runner appends `[stderr]` and
/// `[exit N]` at the end, so a head-only window would hide the verdict.
/// The full text still goes to the model; only the display is shortened.
pub fn output(text: &str) {
    let width = crate::term::cols().saturating_sub(3);
    let (head, hidden, tail) = output_parts(text, HEAD, TAIL);
    let mut block = String::new();
    // The corner belongs to the block, not to each piece of it: the tail is the
    // same output continued, so it only lines up under the head.
    let mut corner = true;
    if !head.trim().is_empty() {
        block.push_str(&indent_block(&head, width, corner));
        block.push('\n');
        corner = false;
    }
    if let Some(rest) = hidden {
        let noun = if rest == 1 { "line" } else { "lines" };
        let notice = format!("... {rest} more {noun} (full output sent to model)");
        block.push_str(&paint(DIM, &format!("  {}", clamp_words(&notice, width))));
        block.push('\n');
    }
    if let Some(tail) = tail {
        block.push_str(&indent_block(&tail, width, corner));
        block.push('\n');
    }
    if !block.trim_end().is_empty() {
        row(block.trim_end());
    }
    row("");
}

/// Paint a restored conversation back onto the screen. Compact on purpose: what
/// was said, and the gist of what ran. Resuming is orientation, not a replay of
/// every tool output the model already has in its history.
pub fn transcript(msgs: &[crate::provider::Msg]) {
    let width = crate::term::cols().saturating_sub(3);
    for line in transcript_text(msgs, width).lines() {
        row(line);
    }
}

/// The transcript as text, so the layout can be unit-tested.
fn transcript_text(msgs: &[crate::provider::Msg], width: usize) -> String {
    use crate::provider::Msg;
    // A restored call gets the same dot colour a live one had, so the pairing
    // has to be resolved before any line is written.
    let outcomes: std::collections::HashMap<&str, bool> = msgs
        .iter()
        .filter_map(|m| match m {
            Msg::ToolResult { id, output } => Some((id.as_str(), crate::shell::exit_ok(output))),
            _ => None,
        })
        .collect();
    let mut out = String::new();
    let mut first = true;
    for m in msgs {
        // A tool result belongs to the call above it, so no break between them.
        if !first && !matches!(m, Msg::ToolResult { .. }) {
            out.push('\n');
        }
        first = false;
        match m {
            Msg::User(t) => {
                let mut lines = t.lines().filter(|l| !l.trim().is_empty());
                if let Some(first) = lines.next() {
                    out.push_str(&format!("{}\n", echo_line(&clamp(first, width))));
                }
                for l in lines {
                    out.push_str(&format!("  {}\n", clamp(l, width)));
                }
            }
            Msg::Assistant { text, calls } => {
                let mut md = Md::new();
                let mut rendered = md.push(text);
                rendered.push_str(&md.flush());
                for line in rendered.lines() {
                    if !line.trim().is_empty() {
                        out.push_str(line);
                        out.push('\n');
                    }
                }
                for c in calls {
                    let cmd = clamp(&c.label(), width);
                    let ok = outcomes.get(c.id.as_str()).copied().unwrap_or(false);
                    out.push_str(&format!("{}\n", tool_echo(&cmd, ok)));
                }
            }
            Msg::ToolResult { output, .. } => {
                let lines: Vec<&str> =
                    output.lines().filter(|l| !l.trim().is_empty()).collect();
                let Some(first) = lines.first() else { continue };
                let head = clamp(first, width);
                out.push_str(&format!("{}\n", paint(DIM, &format!("{OUT} {head}"))));
                if lines.len() > 1 {
                    let rest = lines.len() - 1;
                    out.push_str(&format!("{}\n", paint(DIM, &format!("  ... {rest} more lines"))));
                }
            }
        }
    }
    out.trim_end().to_string()
}

/// Tool output hangs off its command: the first line takes the corner, the rest
/// line up under it, so a call and its result read as one shape. The runner's
/// own markers stay dim; the payload keeps the terminal's default weight,
/// because it is usually code or file contents and has to be readable.
fn indent_block(text: &str, width: usize, corner: bool) -> String {
    let mut out = String::new();
    let mut cornered = !corner;
    for l in text.lines() {
        let l = clamp(l, width);
        // Leading blanks would leave the corner dangling, so they are dropped.
        if !cornered && l.trim().is_empty() {
            continue;
        }
        let body = if is_marker(&l) { paint(DIM, &l) } else { l };
        if !cornered {
            cornered = true;
            out.push_str(&format!("{} {body}\n", paint(DIM, OUT)));
            continue;
        }
        out.push_str(&format!("  {body}\n"));
    }
    out.trim_end_matches('\n').to_string()
}

/// A line the runner appends, not output the command produced.
fn is_marker(l: &str) -> bool {
    let l = l.trim_start();
    l.starts_with("[exit ")
        || l.starts_with("[stderr]")
        || l.starts_with("[killed")
        || l.starts_with("[stopped")
}

const HEAD: usize = 3;
const TAIL: usize = 2;

/// (head, hidden line count, tail). `None` hidden means everything fits.
fn output_parts(text: &str, head: usize, tail: usize) -> (String, Option<usize>, Option<String>) {
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() <= head + tail + 1 {
        return (text.to_string(), None, None);
    }
    let hidden = lines.len() - head - tail;
    (
        lines[..head].join("\n"),
        Some(hidden),
        Some(lines[lines.len() - tail..].join("\n")),
    )
}

/// What a truncation leaves behind. Three ASCII dots, because a terminal that
/// renders a single ellipsis glyph is the terminal this UI is not written for.
const ELLIPSIS: &str = "...";

/// Truncate to `width` chars, marking the cut with `...`. Width is a char
/// count: a cell's real width is the terminal's business, but the result must
/// still fit in `width`, so the marker is reserved rather than added.
fn clamp(s: &str, width: usize) -> String {
    if s.chars().count() <= width {
        return s.to_string();
    }
    let n = width.saturating_sub(ELLIPSIS.len());
    let mut out: String = s.chars().take(n).collect();
    out.push_str(ELLIPSIS);
    out
}

/// Like `clamp`, but backs up to the last word break: used where a broken word
/// would read worse than a lost one.
fn clamp_words(s: &str, width: usize) -> String {
    if s.chars().count() <= width {
        return s.to_string();
    }
    let budget = width.saturating_sub(ELLIPSIS.len());
    let head: String = s.chars().take(budget).collect();
    match head.rfind(' ') {
        Some(i) => format!("{}...", &head[..i]),
        None => format!("{head}..."),
    }
}

/// The `/stats` report, laid out for reading.
///
/// Two columns and a heading per section: the labels are dim and right-padded
/// so the values line up into a column, the values carry the terminal's own
/// weight because they are the point, and the notes are dim because they only
/// qualify. The heading is dim too, so the numbers stay the brightest thing on
/// screen without a single hue being spent on structure.
pub fn stats_report(ctx: crate::stats::ContextFigures) {
    row(stats_frame(ctx).trim_end());
}

/// The report body, built without printing so its layout can be asserted.
/// Layout lives in `stats::render`; this only supplies the dimming.
fn stats_frame(ctx: crate::stats::ContextFigures) -> String {
    let sections = crate::stats::lines(ctx);
    crate::stats::render(&sections, &|s| paint(DIM, s))
}

/// A report or a state notice: neutral weight, no hue. `/help`, `/status`,
/// "thinking set to high", "stopped (Esc)".
pub fn info(s: &str) {
    row(s);
}

/// A warning: yellow and always marked, so severity never rides on hue alone.
pub fn warn(s: &str) {
    row(&paint(YELLOW, &format!("! {s}")));
}

/// A failure: red and always prefixed, for the same reason.
pub fn error(s: &str) {
    let _row = crate::term::row_lock();
    let line = paint(RED, &format!("error: {s}"));
    crate::term::live_take_row();
    eprintln!("{line}");
    crate::term::live_reprint();
}

/// The outcome of something you typed. Green is the "yours" role, so your turn
/// and what it did share one hue; the glyph carries the state without it.
pub fn ok(s: &str) {
    row(&paint(GREEN, &format!("ok {s}")));
}

/// One dot per character of a secret. The key itself never reaches the screen,
/// but its length does: a paste that dropped characters is visible before the
/// key is saved.
fn masked(len: usize) -> String {
    "•".repeat(len)
}

/// The masked input line, kept to one terminal row. Long keys cap their dots
/// and carry an exact count, so a 100-character paste is still verifiable
/// without wrapping the prompt and corrupting the next repaint.
fn secret_line(label: &str, len: usize) -> String {
    let room = crate::term::cols().saturating_sub(label.chars().count() + 12).max(4);
    let dots = len.min(room);
    let mut s = format!("{label}{}", masked(dots));
    if len > dots {
        s.push('…');
    }
    s.push_str(&format!("  {len} chars"));
    s
}

/// Prompt for a secret (API key). Characters are masked with dots rather than
/// hidden completely, so the paste can be checked at a glance. Enter saves,
/// Esc or Ctrl-C cancels. `None` on cancel or an empty line.
pub fn prompt_secret(label: &str) -> Option<String> {
    if !crate::term::is_tty() {
        // No terminal to draw dots on: read the line silently, as before.
        print!("{label}");
        io::stdout().flush().ok();
        return read_raw().map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
    }
    let _raw = crate::term::RawGuard::new();
    let mut secret = String::new();
    out(&secret_line(label, 0));
    loop {
        match read_key() {
            Some(Key::Char(c)) if !c.is_control() => {
                secret.push(c);
                out(&format!("\r\x1b[K{}", secret_line(label, secret.chars().count())));
            }
            Some(Key::Backspace) => {
                secret.pop();
                out(&format!("\r\x1b[K{}", secret_line(label, secret.chars().count())));
            }
            Some(Key::Enter) | Some(Key::Newline) => {
                out("\n");
                let v = secret.trim().to_string();
                return if v.is_empty() { None } else { Some(v) };
            }
            Some(Key::CtrlC) | Some(Key::CtrlD) | Some(Key::Esc) | None => {
                out("\n");
                return None;
            }
            // Resize and editing keys a one-line secret does not use.
            _ => {}
        }
    }
}

fn read_raw() -> Option<String> {
    let mut s = String::new();
    match io::stdin().lock().read_line(&mut s) {
        Ok(0) | Err(_) => None,
        Ok(_) => Some(s),
    }
}

/// One menu row: token to insert + dim description.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Item {
    name: String,
    desc: String,
}

/// Pure completion query: word under `cursor` (char index) -> (word start, items).
/// The live menu, Tab and Enter all read this, so it is unit-tested below.
fn completion_items(
    buffer: &str,
    cursor: usize,
    current_provider: &str,
    current_model: &str,
    catalog: &[(String, String)],
    providers: &[(String, String)],
    thinking: &[&'static str],
) -> Option<(usize, Vec<Item>)> {
    let chars: Vec<char> = buffer.chars().collect();
    let cursor = cursor.min(chars.len());
    let mut start = cursor;
    while start > 0 && !chars[start - 1].is_whitespace() {
        start -= 1;
    }
    let word: String = chars[start..cursor].iter().collect();
    if start == 0 {
        if !word.starts_with('/') {
            return None;
        }
        let items: Vec<Item> = crate::commands::candidates(&word)
            .into_iter()
            .map(|(n, d)| Item { name: format!("{n} "), desc: d.to_string() })
            .collect();
        return if items.is_empty() { None } else { Some((start, items)) };
    }
    let head: String = chars[..start].iter().collect();
    let items: Vec<Item> = if head.trim_end() == "/provider" {
        let frag = word.to_ascii_lowercase();
        // `add` and `remove` are subcommands, not provider names: while one is
        // being typed, the menu must not hijack Enter with a provider row.
        if frag == "add" || frag == "remove" {
            return None;
        }
        providers
            .iter()
            .filter(|(p, _)| p.to_ascii_lowercase().starts_with(&frag))
            .map(|(p, d)| Item { name: p.clone(), desc: d.clone() })
            .collect()
    } else if head.trim_end() == "/model" {
        crate::commands::model_candidates(catalog, current_provider, current_model, &word)
            .into_iter()
            .map(|(n, d)| Item { name: n, desc: d })
            .collect()
    } else if head.trim_end() == "/thinking" {
        let frag = word.to_ascii_lowercase();
        thinking
            .iter()
            .filter(|l| l.starts_with(frag.as_str()))
            .map(|l| Item { name: l.to_string(), desc: "thinking level".to_string() })
            .collect()
    } else {
        return None;
    };
    if items.is_empty() { None } else { Some((start, items)) }
}

fn exact_hit(
    buffer: &str,
    cursor: usize,
    catalog: &[(String, String)],
    providers: &[(String, String)],
    thinking: &[&'static str],
) -> bool {
    let chars: Vec<char> = buffer.chars().collect();
    let cursor = cursor.min(chars.len());
    let mut start = cursor;
    while start > 0 && !chars[start - 1].is_whitespace() {
        start -= 1;
    }
    let word: String = chars[start..cursor].iter().collect();
    if start == 0 {
        crate::commands::COMMANDS.iter().any(|(n, _)| *n == word)
    } else {
        let head: String = chars[..start].iter().collect();
        match head.trim_end() {
            "/provider" => providers.iter().any(|(p, _)| *p == word),
            "/model" => catalog.iter().any(|(_, id)| *id == word),
            "/thinking" => thinking.iter().any(|l| *l == word),
            _ => false,
        }
    }
}

fn byte_idx(s: &str, char_idx: usize) -> usize {
    s.char_indices().nth(char_idx).map(|(i, _)| i).unwrap_or(s.len())
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Key {
    Char(char),
    Enter,
    Tab,
    ShiftTab,
    Backspace,
    Delete,
    Left,
    Right,
    Up,
    Down,
    Home,
    End,
    Esc,
    CtrlC,
    CtrlD,
    /// A hard line break inside the line being typed (Ctrl-J, Alt-Enter).
    Newline,
    /// Row movement inside a multi-line block, for terminals that send them.
    ShiftUp,
    ShiftDown,
    /// The terminal changed size; the reader repaints.
    Resize,
    Unknown,
}

enum Read {
    Byte(u8),
    Eof,
    Resize,
}

fn read_byte_blocking() -> Read {
    let mut b = [0u8; 1];
    // Raw read: this loop is stdin's single owner.
    loop {
        match crate::term::read_stdin(&mut b) {
            Ok(1) => return Read::Byte(b[0]),
            Ok(_) => return Read::Eof,
            Err(e) if e.kind() != std::io::ErrorKind::Interrupted => return Read::Eof,
            // EINTR: a resize landed on a blocked read.
            Err(_) => {
                if crate::term::took_resize() {
                    return Read::Resize;
                }
            }
        }
    }
}

/// The next byte, ignoring anything that is not a keystroke.
fn next_byte() -> Option<u8> {
    match read_byte_blocking() {
        Read::Byte(b) => Some(b),
        _ => None,
    }
}

fn read_char(first: u8) -> Option<char> {
    let len = if first < 0x80 {
        1
    } else if first & 0xE0 == 0xC0 {
        2
    } else if first & 0xF0 == 0xE0 {
        3
    } else if first & 0xF8 == 0xF0 {
        4
    } else {
        return None;
    };
    let mut buf = vec![first];
    for _ in 1..len {
        buf.push(next_byte()?);
    }
    std::str::from_utf8(&buf).ok()?.chars().next()
}

fn read_esc() -> Key {
    if !crate::term::poll_in(30) {
        return Key::Esc;
    }
    match next_byte() {
        Some(b'[') => {
            let mut param: Vec<u8> = Vec::new();
            loop {
                let c = match next_byte() {
                    Some(c) => c,
                    None => return Key::Esc,
                };
                if (0x40..=0x7E).contains(&c) {
                    // A parameter means a modifier: `1;2A` is Shift-Up. Only
                    // Shift is read, because it is the one that changes what an
                    // arrow means in a multi-line block; anything else is left
                    // as an unknown key rather than misread as a bare arrow.
                    let shifted = param.ends_with(b";2");
                    return match c {
                        b'A' if shifted => Key::ShiftUp,
                        b'B' if shifted => Key::ShiftDown,
                        b'A' => Key::Up,
                        b'B' => Key::Down,
                        b'C' => Key::Right,
                        b'D' => Key::Left,
                        b'H' => Key::Home,
                        b'F' => Key::End,
                        b'Z' => Key::ShiftTab,
                        b'~' => match param.as_slice() {
                            [b'3'] => Key::Delete,
                            [b'1'] | [b'7'] => Key::Home,
                            [b'4'] | [b'8'] => Key::End,
                            _ => Key::Unknown,
                        },
                        _ => Key::Unknown,
                    };
                }
                param.push(c);
                if param.len() > 8 {
                    return Key::Esc;
                }
            }
        }
        Some(b'O') => match next_byte() {
            Some(b'A') => Key::Up,
            Some(b'B') => Key::Down,
            Some(b'C') => Key::Right,
            Some(b'D') => Key::Left,
            Some(b'H') => Key::Home,
            Some(b'F') => Key::End,
            _ => Key::Esc,
        },
        // Alt-Enter is how most terminals send a newline: ESC then CR.
        Some(b'\r') | Some(b'\n') => Key::Newline,
        _ => Key::Esc,
    }
}

/// Block for the next keystroke, waking every 200ms to check the resize flag.
/// A process-wide signal is delivered to an arbitrary thread, so the blocked
/// read here is not reliably interrupted: polling is what makes it prompt.
fn read_key() -> Option<Key> {
    loop {
        if crate::term::took_resize() {
            return Some(Key::Resize);
        }
        if !crate::term::poll_in(200) {
            continue;
        }
        let b = match read_byte_blocking() {
            Read::Byte(b) => b,
            Read::Eof => return None,
            Read::Resize => return Some(Key::Resize),
        };
        return Some(key_for(b));
    }
}

fn key_for(b: u8) -> Key {
    match b {
        0x0d => Key::Enter,
        // A bare LF is Ctrl-J, which is a newline inside the line, not a
        // submit: that is how a multi-line prompt is typed without a modifier.
        0x0a => Key::Newline,
        0x09 => Key::Tab,
        0x7f | 0x08 => Key::Backspace,
        0x03 => Key::CtrlC,
        0x04 => Key::CtrlD,
        0x1b => read_esc(),
        _ => match read_char(b) {
            Some(c) if !c.is_control() => Key::Char(c),
            _ => Key::Unknown,
        },
    }
}

fn out(s: &str) {
    let mut o = io::stdout().lock();
    let _ = o.write_all(s.as_bytes());
    let _ = o.flush();
}

/// One visual row of the input block: the buffer characters it shows.
///
/// Rows are ranges of *buffer* characters, never copies of text, so the cursor
/// maps to a cell by subtracting the row's start. That is the whole reason the
/// layout is computed from the buffer instead of from rendered strings.
#[derive(Debug, Clone, PartialEq)]
struct Row {
    start: usize,
    end: usize,
}

/// Wrap the current line, then break on each newline, to get the whole block.
///
/// A hard newline always breaks. Past `width` the break is soft: at the last
/// space in reach when there is one, so a long prompt wraps where a reader
/// would break it instead of mid-word. The space a break falls on is consumed
/// (it starts no row), which is why a row range is not always followed by one
/// beginning exactly at its end.
fn rows_of(chars: &[char], width: usize) -> Vec<Row> {
    let width = width.max(1);
    let mut rows = Vec::new();
    let mut i = 0usize;
    loop {
        // The hard line this row belongs to.
        let mut j = i;
        while j < chars.len() && chars[j] != '\n' {
            j += 1;
        }
        let hard = j < chars.len();
        let mut s = i;
        let mut first = true;
        loop {
            // A soft break that landed on a space leaves it behind, so a
            // continuation row never opens with a stray blank.
            if !first && chars.get(s) == Some(&' ') {
                s += 1;
            }
            first = false;
            if j.saturating_sub(s) <= width {
                rows.push(Row { start: s.min(j), end: j });
                break;
            }
            let limit = s + width;
            let brk = (s + 1..limit).rev().find(|&k| chars[k] == ' ' || chars[k] == '\t');
            let e = brk.unwrap_or(limit);
            rows.push(Row { start: s, end: e });
            s = if brk.is_some() { e + 1 } else { e };
            if s > j {
                s = j;
            }
        }
        if !hard {
            break;
        }
        i = j + 1;
        if i >= chars.len() {
            // A trailing newline gets its own empty row: the cursor needs a
            // place to sit when the line has just been opened.
            rows.push(Row { start: chars.len(), end: chars.len() });
            break;
        }
    }
    if rows.is_empty() {
        rows.push(Row { start: 0, end: 0 });
    }
    rows
}

/// Which row the cursor is on. A cursor at a row's end stays on that row: the
/// position after the last character of a line is the end of that line.
fn row_of(rows: &[Row], cursor: usize) -> usize {
    rows.iter()
        .position(|r| cursor >= r.start && cursor <= r.end)
        .unwrap_or(rows.len().saturating_sub(1))
}

/// Max visible menu rows; longer menus scroll around the selection.
const MENU_ROWS: usize = 10;

/// Visible window (start, end exclusive) keeping `sel` in view. Pure.
fn window(len: usize, sel: usize, max: usize) -> (usize, usize) {
    if len <= max {
        return (0, len);
    }
    let start = sel.saturating_sub(max / 2).min(len - max);
    (start, start + max)
}

/// The input block, as rows of text with the prompt mark on the first.
///
/// Returns the rows, how far the block scrolls, and the row the cursor is on.
/// A scrolled block opens and closes with a count of what is off screen, so a
/// long prompt is legible without a second read of the file it came from.
fn input_block(
    chars: &[char],
    cursor: usize,
    width: usize,
    room: usize,
) -> (Vec<String>, usize, usize) {
    let wrap = width.saturating_sub(2).max(1);
    let rows = rows_of(chars, wrap);
    let cur = row_of(&rows, cursor);
    // Scrolling costs two rows for the hints, so they are only spent when
    // there is something hidden to name.
    let (vstart, vend, body) = if rows.len() <= room {
        (0, rows.len(), room)
    } else {
        let body = room.saturating_sub(2).max(1);
        let (a, b) = window(rows.len(), cur, body);
        (a, b, body)
    };
    let _ = body;
    let mut out = Vec::new();
    if vstart > 0 {
        out.push(paint(DIM, &clamp(&format!("  ^^ {vstart} more lines"), width)));
    }
    for (i, r) in rows.iter().enumerate().skip(vstart).take(vend - vstart) {
        let text: String = chars[r.start..r.end.min(chars.len())].iter().collect();
        let mark = if i == 0 { PROMPT } else { "  " };
        let head = if i == 0 { format!("{mark} ") } else { "  ".to_string() };
        out.push(format!("{head}{}", clamp(&text, wrap)));
    }
    if vend < rows.len() {
        out.push(paint(DIM, &clamp(&format!("  vv {} more lines", rows.len() - vend), width)));
    }
    let offset = usize::from(vstart > 0);
    (out, offset + cur.saturating_sub(vstart), rows.len())
}

/// The bounds of the hard line the cursor is on: (start, end), where end is
/// the position of the newline itself or the buffer's length. Home and End are
/// scoped to this, not to the whole buffer, which is what they mean in a
/// multi-line editor.
fn line_bounds(chars: &[char], cursor: usize) -> (usize, usize) {
    let cursor = cursor.min(chars.len());
    let start = chars[..cursor].iter().rposition(|&c| c == '\n').map_or(0, |i| i + 1);
    let end = chars[cursor..]
        .iter()
        .position(|&c| c == '\n')
        .map_or(chars.len(), |i| cursor + i);
    (start, end)
}

/// Move the cursor one visual row up or down, keeping its column where the row
/// allows. `Ok(new)` is the new position; `Err(())` means the block's edge was
/// already reached, which is the caller's cue to walk history instead.
fn move_row(chars: &[char], cursor: usize, width: usize, up: bool) -> Result<usize, ()> {
    let rows = rows_of(chars, width.saturating_sub(2).max(1));
    let r = row_of(&rows, cursor.min(chars.len()));
    let target = if up {
        if r == 0 {
            return Err(());
        }
        r - 1
    } else {
        if r + 1 >= rows.len() {
            return Err(());
        }
        r + 1
    };
    let cur = cursor.min(chars.len());
    // A cursor on its row's newline counts as parked at the end, not as a
    // column one past the row's last character.
    let at_end = cur == rows[r].end || chars.get(cur) == Some(&'\n');
    let col = cur.saturating_sub(rows[r].start);
    let span = rows[target].end - rows[target].start;
    if at_end {
        // At a row's end, up or down lands on the next row's end: a cursor
        // parked after the last character keeps its claim to that place, and
        // column preservation would instead put it mid-word of the row above.
        return Ok(rows[target].end);
    }
    Ok(rows[target].start + col.min(span))
}

/// Where the cursor sits on its own row, as a column from the mark's left edge.
/// Recomputed from the buffer rather than carried out of `input_block`, so the
/// frame never has to trust a second number to agree with the first.
fn cursor_col(chars: &[char], cursor: usize, width: usize) -> usize {
    let wrap = width.saturating_sub(2).max(1);
    let rows = rows_of(chars, wrap);
    let r = &rows[row_of(&rows, cursor.min(chars.len()))];
    2 + cursor.min(chars.len()).saturating_sub(r.start)
}

/// Build one screen frame: the input block, then the live menu under it, with
/// the cursor put back where it belongs. Every row is clamped to `width` so the
/// terminal never wraps one logical row into two, which would break the row
/// arithmetic below.
///
/// `prev` is the previous frame's (rows drawn, cursor row): the cursor is left
/// on its own row, so the frame first walks back up to the block's top before
/// repainting. Surplus rows are wiped blank so a closed menu or a shorter
/// prompt never leaves ghost text behind. Returns the frame and this frame's
/// (rows, cursor row), which the next call needs.
/// Pure (no I/O) so tests assert exact sequences.
fn frame(
    buf: &str,
    cursor: usize,
    menu: &[Item],
    sel: usize,
    prev: (usize, usize),
    width: usize,
    height: usize,
) -> (String, (usize, usize)) {
    let chars: Vec<char> = buf.chars().collect();
    let cursor = cursor.min(chars.len());
    let n = menu.len();
    let max_rows = MENU_ROWS.min(height.saturating_sub(2)).max(3);
    let (wstart, wend) = window(n, sel.min(n.saturating_sub(1)), max_rows);
    let mut menu_rows = if n > 0 { wend - wstart } else { 0 };
    if wstart > 0 {
        menu_rows += 1;
    }
    if wend < n {
        menu_rows += 1;
    }
    // The block gets the terminal's height, less the menu and one row of
    // breathing space, and never less than one row of input.
    let room = height.saturating_sub(menu_rows + 1).max(1);
    let (mut lines, cursor_row, _) = input_block(&chars, cursor, width, room);

    if wstart > 0 {
        lines.push(paint(DIM, &clamp(&format!("  ^ {wstart} more"), width)));
    }
    if n > 0 {
        let name_w = menu.iter().map(|i| i.name.trim_end().chars().count()).max().unwrap_or(0);
        for (idx, it) in menu[wstart..wend].iter().enumerate() {
            let prefix = format!("  {:<name_w$}  ", it.name.trim_end());
            let desc = clamp(&it.desc, width.saturating_sub(prefix.chars().count()));
            if wstart + idx == sel {
                lines.push(paint(REVERSE, &format!("{prefix}{desc}")));
            } else {
                lines.push(format!("{prefix}{}", paint(DIM, &desc)));
            }
        }
    }
    if wend < n {
        lines.push(paint(DIM, &clamp(&format!("  v {} more", n - wend), width)));
    }
    for _ in lines.len()..prev.0 {
        lines.push(String::new());
    }
    let total = lines.len().max(prev.0);

    let mut s = String::from("\x1b[?25l");
    if prev.1 > 0 {
        s.push_str(&format!("\x1b[{}A", prev.1));
    }
    s.push_str("\r\x1b[K");
    for (i, l) in lines.iter().enumerate() {
        if i > 0 {
            s.push_str("\n\r\x1b[K");
        }
        s.push_str(l);
    }
    // Back to the block's top row, then down to the cursor's row and across.
    // The cursor sits on the last row painted, so the walk back is `total - 1`:
    // asking for `total` lands a row above the block, and the next frame paints
    // there, leaving every earlier frame's text behind one row further down.
    if total > 1 {
        s.push_str(&format!("\x1b[{}A", total - 1));
    }
    s.push('\r');
    if cursor_row > 0 {
        s.push_str(&format!("\x1b[{cursor_row}B"));
    }
    s.push_str(&format!("\x1b[{}C", cursor_col(&chars, cursor, width)));
    s.push_str("\x1b[?25h");
    (s, (total, cursor_row))
}

/// Erase the whole input block, leaving the cursor on the row the block began
/// on. `block` is what `frame` returned.
fn clear_menu(block: (usize, usize)) {
    let (rows, cur) = block;
    if rows == 0 {
        return;
    }
    let mut s = String::new();
    if cur > 0 {
        s.push_str(&format!("\x1b[{cur}A"));
    }
    for i in 0..rows {
        s.push_str("\r\x1b[K");
        if i + 1 < rows {
            s.push_str("\x1b[B");
        }
    }
    if rows > 1 {
        s.push_str(&format!("\x1b[{}A", rows - 1));
    }
    s.push('\r');
    out(&s);
}

/// What one keypress means for the input loop. All logic lives in `Input`;
/// `Reader` only does I/O around it, which is why this is unit-testable.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Outcome {
    Redraw,
    Submit,
    Quit,
}

pub(crate) struct Input<'h> {
    buf: String,
    cursor: usize, // char index
    sel: usize,
    menu: Vec<Item>,
    mstart: usize,
    menu_off: bool, // Esc hides the menu until the next edit
    provider: String,
    current: String,
    catalog: Vec<(String, String)>,
    providers: Vec<(String, String)>,
    /// The levels the current model documents; `/thinking` completes from it.
    thinking: Vec<&'static str>,
    hist: &'h mut Vec<String>,
    hpos: Option<usize>,
    draft: String,
}

impl<'h> Input<'h> {
    pub fn new(hist: &'h mut Vec<String>) -> Self {
        Input {
            buf: String::new(),
            cursor: 0,
            sel: 0,
            menu: Vec::new(),
            mstart: 0,
            menu_off: false,
            provider: "anthropic".to_string(),
            current: String::new(),
            catalog: Vec::new(),
            providers: Vec::new(),
            thinking: Vec::new(),
            hist,
            hpos: None,
            draft: String::new(),
        }
    }

    /// Refresh the model catalog context (called per prompt; provider can
    /// change mid-session via `/provider`, and picks can switch provider).
    pub fn set_context(
        &mut self,
        provider: &str,
        current: &str,
        catalog: &[(String, String)],
        providers: &[(String, String)],
        thinking: &[&'static str],
    ) {
        self.provider = provider.to_string();
        self.current = current.to_string();
        self.catalog = catalog.to_vec();
        self.providers = providers.to_vec();
        self.thinking = thinking.to_vec();
    }

    pub fn text(&self) -> &str {
        &self.buf
    }
    pub fn cursor_pos(&self) -> usize {
        self.cursor
    }
    pub fn menu(&self) -> &[Item] {
        &self.menu
    }
    pub fn sel(&self) -> usize {
        self.sel
    }

    pub fn set_text(&mut self, text: &str) {
        self.buf = text.to_string();
        self.cursor = self.buf.chars().count();
        self.edit_done();
    }

    /// Forget the current line (empty submit / cancel): no scroll spam.
    pub fn clear_line(&mut self) {
        self.buf.clear();
        self.cursor = 0;
        self.sel = 0;
        self.menu.clear();
        self.menu_off = false;
        self.hpos = None;
    }

    fn recompute(&mut self) {
        if self.menu_off {
            self.menu.clear();
            return;
        }
        match completion_items(
            &self.buf,
            self.cursor,
            &self.provider,
            &self.current,
            &self.catalog,
            &self.providers,
            &self.thinking,
        ) {
            Some((st, items)) => {
                self.mstart = st;
                self.menu = items;
                self.sel = self.sel.min(self.menu.len().saturating_sub(1));
            }
            None => self.menu.clear(),
        }
    }

    fn edit_done(&mut self) {
        self.hpos = None;
        self.sel = 0;
        self.menu_off = false;
        self.recompute();
    }

    /// Complete the highlighted row as-is (Tab never skips ahead).
    fn accept_sel(&mut self) {
        if self.menu.is_empty() {
            return;
        }
        let item = self.menu[self.sel].clone();
        accept(&mut self.buf, &mut self.cursor, &item, self.mstart);
        self.edit_done();
    }

    /// Walk back through history from wherever the walk currently sits.
    fn history_up(&mut self) {
        if self.hist.is_empty() {
            return;
        }
        match self.hpos {
            None => {
                self.draft = self.buf.clone();
                self.hpos = Some(self.hist.len() - 1);
            }
            Some(p) if p > 0 => self.hpos = Some(p - 1),
            _ => return,
        }
        self.buf = self.hist[self.hpos.unwrap()].clone();
        self.cursor = self.buf.chars().count();
        self.sel = 0;
        self.menu_off = false;
        self.recompute();
    }

    /// Walk forward through history; off the end restores the draft.
    fn history_down(&mut self) {
        if let Some(p) = self.hpos {
            if p + 1 < self.hist.len() {
                self.hpos = Some(p + 1);
                self.buf = self.hist[p + 1].clone();
            } else {
                self.hpos = None;
                self.buf = self.draft.clone();
            }
            self.cursor = self.buf.chars().count();
            self.sel = 0;
            self.menu_off = false;
            self.recompute();
        }
    }

    pub fn key(&mut self, k: Key) -> Outcome {
        match k {
            Key::Unknown => Outcome::Redraw,
            Key::Resize => Outcome::Redraw,
            Key::Char(c) => {
                self.buf.insert(byte_idx(&self.buf, self.cursor), c);
                self.cursor += 1;
                self.edit_done();
                Outcome::Redraw
            }
            Key::Backspace => {
                if self.cursor == 0 {
                    return Outcome::Redraw;
                }
                self.cursor -= 1;
                self.buf.remove(byte_idx(&self.buf, self.cursor));
                self.edit_done();
                Outcome::Redraw
            }
            Key::Delete => {
                if self.cursor >= self.buf.chars().count() {
                    return Outcome::Redraw;
                }
                self.buf.remove(byte_idx(&self.buf, self.cursor));
                self.edit_done();
                Outcome::Redraw
            }
            Key::Left => {
                if self.cursor > 0 {
                    self.cursor -= 1;
                    self.recompute();
                }
                Outcome::Redraw
            }
            Key::Right => {
                if self.cursor < self.buf.chars().count() {
                    self.cursor += 1;
                    self.recompute();
                }
                Outcome::Redraw
            }
            Key::Home => {
                // In a multi-line block Home is the line's start, not the
                // buffer's: jumping to the top of a three-line prompt would
                // lose the place a row-wise walk had built.
                let chars: Vec<char> = self.buf.chars().collect();
                self.cursor = line_bounds(&chars, self.cursor).0;
                self.recompute();
                Outcome::Redraw
            }
            Key::End => {
                let chars: Vec<char> = self.buf.chars().collect();
                self.cursor = line_bounds(&chars, self.cursor).1;
                self.recompute();
                Outcome::Redraw
            }
            Key::Up => {
                if !self.menu.is_empty() {
                    self.sel = (self.sel + self.menu.len() - 1) % self.menu.len();
                } else if self.buf.contains('\n') {
                    // A multi-line block: up is up a row, and history only
                    // waits at the block's first row, where it always was.
                    let chars: Vec<char> = self.buf.chars().collect();
                    let width = crate::term::cols();
                    match move_row(&chars, self.cursor, width, true) {
                        Ok(c) => {
                            self.cursor = c;
                            self.recompute();
                        }
                        Err(()) => self.history_up(),
                    }
                } else if !self.hist.is_empty() {
                    self.history_up();
                }
                Outcome::Redraw
            }
            Key::ShiftUp => {
                let chars: Vec<char> = self.buf.chars().collect();
                if let Ok(c) = move_row(&chars, self.cursor, crate::term::cols(), true) {
                    self.cursor = c;
                    self.recompute();
                }
                Outcome::Redraw
            }
            Key::ShiftDown => {
                let chars: Vec<char> = self.buf.chars().collect();
                if let Ok(c) = move_row(&chars, self.cursor, crate::term::cols(), false) {
                    self.cursor = c;
                    self.recompute();
                }
                Outcome::Redraw
            }
            Key::Down => {
                if !self.menu.is_empty() {
                    self.sel = (self.sel + 1) % self.menu.len();
                } else if self.buf.contains('\n') && self.hpos.is_none() {
                    let chars: Vec<char> = self.buf.chars().collect();
                    let width = crate::term::cols();
                    match move_row(&chars, self.cursor, width, false) {
                        Ok(c) => {
                            self.cursor = c;
                            self.recompute();
                        }
                        // The block's last row, walking into history: the draft
                        // is the block itself, so it is kept for the way back.
                        Err(()) => self.history_down(),
                    }
                } else if !self.hist.is_empty() {
                    self.history_down();
                }
                Outcome::Redraw
            }
            Key::Tab => {
                self.accept_sel();
                Outcome::Redraw
            }
            Key::ShiftTab => {
                if !self.menu.is_empty() {
                    self.sel = (self.sel + self.menu.len() - 1) % self.menu.len();
                }
                Outcome::Redraw
            }
            Key::Enter => {
                // Menu open: Enter picks the highlighted row, unless the word
                // is already exactly what is highlighted — then it submits.
                if !self.menu.is_empty()
                    && !exact_hit(&self.buf, self.cursor, &self.catalog, &self.providers, &self.thinking)
                {
                    self.accept_sel();
                    Outcome::Redraw
                } else {
                    if !self.buf.trim().is_empty() {
                        self.hist.push(self.buf.clone());
                    }
                    self.hpos = None;
                    Outcome::Submit
                }
            }
            Key::Newline => {
                // The menu is a one-line affair: while it is open the key
                // completes instead of breaking the line, because a `\n` inside
                // a slash command is never what was meant.
                if !self.menu.is_empty() {
                    self.accept_sel();
                } else {
                    self.buf.insert(byte_idx(&self.buf, self.cursor), '\n');
                    self.cursor += 1;
                    self.edit_done();
                }
                Outcome::Redraw
            }
            Key::Esc => {
                if !self.menu.is_empty() {
                    self.menu.clear();
                    self.menu_off = true;
                    Outcome::Redraw
                } else {
                    // Cancel the line; the loop skips the newline when empty.
                    self.clear_line();
                    Outcome::Submit
                }
            }
            Key::CtrlC => {
                self.clear_line();
                Outcome::Submit
            }
            Key::CtrlD => {
                if self.buf.is_empty() {
                    Outcome::Quit
                } else {
                    Outcome::Redraw
                }
            }
        }
    }
}

pub struct Reader {
    hist: Vec<String>,
    /// Raw mode already entered by the startup reveal. The first prompt adopts
    /// it instead of entering raw mode again and flushing the keys typed while
    /// the tagline was animating.
    raw: Option<crate::term::RawGuard>,
}

impl Reader {
    pub fn new() -> Self {
        Reader { hist: Vec::new(), raw: None }
    }

    /// Hand the reader a raw-mode guard to use for the first prompt.
    pub fn hold_raw(&mut self, raw: Option<crate::term::RawGuard>) {
        self.raw = raw;
    }

    pub fn read_line(
        &mut self,
        cfg: &Config,
        catalog: &[(String, String)],
        providers: &[(String, String)],
        prefill: Option<&str>,
    ) -> anyhow::Result<Option<String>> {
        // Take the guard before the TTY check, so a guard acquired by the
        // reveal is restored on every return path. The first prompt inherits
        // it; later prompts enter raw mode themselves.
        let raw = self.raw.take();
        if !crate::term::is_tty() {
            print!("{PROMPT} ");
            io::stdout().flush().ok();
            return Ok(read_raw().map(|s| s.trim_end_matches(['\n', '\r']).to_string()));
        }
        let _raw = raw.or_else(crate::term::RawGuard::new);
        let mut input = Input::new(&mut self.hist);
        // Context first: `set_text` recomputes the menu, and the menu needs
        // the catalog to have anything to show.
        input.set_context(
            cfg.provider.name(),
            &cfg.model,
            catalog,
            providers,
            &cfg.thinking_choices(),
        );
        if let Some(text) = prefill {
            input.set_text(text);
        }
        let (mut width, mut height) = (crate::term::cols(), crate::term::rows());
        let (f, mut drawn) = frame(
            input.text(),
            input.cursor_pos(),
            input.menu(),
            input.sel(),
            (0, 0),
            width,
            height,
        );
        out(&f);
        loop {
            let key = match read_key() {
                Some(k) => k,
                None => {
                    clear_menu(drawn);
                    out("\n");
                    return Ok(None);
                }
            };
            if key == Key::Resize {
                (width, height) = (crate::term::cols(), crate::term::rows());
                let (f, nd) = frame(
                    input.text(),
                    input.cursor_pos(),
                    input.menu(),
                    input.sel(),
                    drawn,
                    width,
                    height,
                );
                drawn = nd;
                out(&f);
                continue;
            }
            match input.key(key) {
                Outcome::Quit => {
                    clear_menu(drawn);
                    out("\n");
                    return Ok(None);
                }
                Outcome::Submit => {
                    let line = input.text().to_string();
                    if line.trim().is_empty() {
                        // Stay on the same line: no newline, no `>` stacking.
                        input.clear_line();
                        let (f, _) = frame(
                            input.text(),
                            input.cursor_pos(),
                            input.menu(),
                            input.sel(),
                            drawn,
                            width,
                            height,
                        );
                        out(&f);
                        return Ok(Some(String::new()));
                    }
                    clear_menu(drawn);
                    // Reprint the submitted line in the "yours" role, above the
                    // machine's tool echo. A multi-line prompt is echoed as it
                    // was written, so what was sent is what is on screen.
                    out(&echo_block(&line));
                    return Ok(Some(line));
                }
                Outcome::Redraw => {
                    let (f, nd) = frame(
                        input.text(),
                        input.cursor_pos(),
                        input.menu(),
                        input.sel(),
                        drawn,
                        width,
                        height,
                    );
                    drawn = nd;
                    out(&f);
                }
            }
        }
    }
}

fn accept(buf: &mut String, cursor: &mut usize, item: &Item, mstart: usize) {
    let end = byte_idx(buf, *cursor);
    buf.replace_range(byte_idx(buf, mstart)..end, &item.name);
    *cursor = mstart + item.name.chars().count();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The built-in provider rows the agent passes to the reader.
    fn providers() -> Vec<(String, String)> {
        crate::commands::PROVIDERS
            .iter()
            .map(|n| (n.to_string(), "built-in".to_string()))
            .collect()
    }

    #[test]
    fn a_secret_is_shown_as_one_dot_per_character() {
        assert_eq!(masked(0), "");
        assert_eq!(masked(3).chars().count(), 3);
        // The characters are dots, never the secret itself.
        assert!(masked(5).chars().all(|c| c == '•'));
        let line = secret_line("Paste KEY: ", 5);
        assert_eq!(line.matches('•').count(), 5);
        assert!(line.ends_with("5 chars"), "{line}");
        // A long paste caps its dots but still reports its exact length.
        let long = secret_line("Paste KEY: ", 500);
        assert!(long.matches('•').count() < 500, "{long}");
        assert!(long.contains("500 chars"), "{long}");
    }

    #[test]
    fn the_provider_menu_includes_custom_entries() {
        let mut all = providers();
        all.push(("ollama".into(), "custom openai API, no key needed".into()));
        let (_, items) = completion_items("/provider o", 11, "anthropic", "", &[], &all, crate::config::THINKING_LEVELS).unwrap();
        let names: Vec<_> = items.iter().map(|i| i.name.as_str()).collect();
        assert_eq!(names, vec!["openai", "openrouter", "ollama"]);
    }

    #[test]
    fn provider_subcommands_are_not_hijacked_by_a_lookalike_name() {
        let mut all = providers();
        all.push(("adder".into(), "custom openai API".into()));
        assert!(completion_items("/provider add", 13, "anthropic", "", &[], &all, crate::config::THINKING_LEVELS).is_none());
        assert!(completion_items("/provider remove", 16, "anthropic", "", &[], &all, crate::config::THINKING_LEVELS).is_none());
    }

    #[test]
    fn slash_lists_every_command() {
        let (start, items) = completion_items("/", 1, "anthropic", "", &[], &providers(), crate::config::THINKING_LEVELS).unwrap();
        assert_eq!(start, 0);
        assert_eq!(items.len(), crate::commands::COMMANDS.len());
    }

    #[test]
    fn partial_filters() {
        // `/mcp` shares the /m prefix with `/model`, so both must be offered.
        let (_, items) = completion_items("/m", 2, "anthropic", "", &[], &providers(), crate::config::THINKING_LEVELS).unwrap();
        assert_eq!(items.len(), 2);
        assert!(items.iter().all(|i| i.name.starts_with("/m")));
    }

    #[test]
    fn plain_text_has_no_menu() {
        assert!(completion_items("hello", 5, "anthropic", "", &[], &providers(), crate::config::THINKING_LEVELS).is_none());
        assert!(completion_items("", 0, "anthropic", "", &[], &providers(), crate::config::THINKING_LEVELS).is_none());
    }

    #[test]
    fn provider_second_word_completes() {
        let (start, items) = completion_items("/provider an", 12, "anthropic", "", &[], &providers(), crate::config::THINKING_LEVELS).unwrap();
        assert_eq!(start, 10);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].name, "anthropic");
    }

    #[test]
    fn thinking_second_word_completes_only_the_model_levels() {
        // The menu is built from what the current model documents, so a model
        // that reasons in low/high does not offer `max`.
        let levels: &[&str] = &["off", "low", "high"];
        let (start, items) = completion_items(
            "/thinking h",
            11,
            "anthropic",
            "",
            &[],
            &providers(),
            levels,
        )
        .unwrap();
        assert_eq!(start, 10);
        let names: Vec<_> = items.iter().map(|i| i.name.as_str()).collect();
        assert_eq!(names, vec!["high"]);
        let (_, items) = completion_items(
            "/thinking ",
            10,
            "anthropic",
            "",
            &[],
            &providers(),
            levels,
        )
        .unwrap();
        assert_eq!(items.len(), 3);
        assert!(completion_items(
            "/thinking m",
            11,
            "anthropic",
            "",
            &[],
            &providers(),
            levels
        )
        .is_none());
    }

    #[test]
    fn provider_bare_lists_all() {
        let (_, items) = completion_items("/provider ", 10, "anthropic", "", &[], &providers(), crate::config::THINKING_LEVELS).unwrap();
        assert_eq!(items.len(), crate::commands::PROVIDERS.len());
    }

    #[test]
    fn unknown_second_word_has_no_menu() {
        assert!(completion_items("/model xyz", 10, "anthropic", "", &[], &providers(), crate::config::THINKING_LEVELS).is_none());
    }

    fn catalog() -> Vec<(String, String)> {
        vec![
            ("openrouter".to_string(), "openai/gpt-4.1".to_string()),
            ("openrouter".to_string(), "anthropic/claude-sonnet-4.5".to_string()),
            ("openrouter".to_string(), "x-ai/grok-4".to_string()),
            ("anthropic".to_string(), "claude-sonnet-4-5".to_string()),
        ]
    }

    #[test]
    fn model_search_matches_substring() {
        let (_, items) = completion_items(
            "/model gro",
            11,
            "openrouter",
            "openai/gpt-4.1",
            &catalog(),
            &providers(),
            crate::config::THINKING_LEVELS,
        )
        .unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].name, "x-ai/grok-4");
    }

    #[test]
    fn model_search_is_case_insensitive_and_marks_current() {
        let (_, items) = completion_items(
            "/model SONNET",
            13,
            "openrouter",
            "openai/gpt-4.1",
            &catalog(),
            &providers(),
            crate::config::THINKING_LEVELS,
        )
        .unwrap();
        // Matches across connected providers, not just the current one.
        let names: Vec<_> = items.iter().map(|i| i.name.as_str()).collect();
        assert_eq!(names, vec!["anthropic/claude-sonnet-4.5", "claude-sonnet-4-5"]);
        let (_, items) = completion_items(
            "/model ",
            7,
            "openrouter",
            "openai/gpt-4.1",
            &catalog(),
            &providers(),
            crate::config::THINKING_LEVELS,
        )
        .unwrap();
        assert_eq!(items.len(), 4);
        assert!(items.iter().any(|i| i.name == "openai/gpt-4.1" && i.desc == "openrouter, current"));
    }

    #[test]
    fn model_menu_flows_to_submit() {
        let mut h = Vec::new();
        let mut inp = Input::new(&mut h);
        inp.set_context("openrouter", "openai/gpt-4.1", &catalog(), &providers(), crate::config::THINKING_LEVELS);
        type_str(&mut inp, "/model gro");
        assert_eq!(inp.menu().len(), 1);
        assert_eq!(inp.key(Key::Enter), Outcome::Redraw);
        assert_eq!(inp.text(), "/model x-ai/grok-4");
        assert_eq!(inp.key(Key::Enter), Outcome::Submit);
    }

    fn type_str(inp: &mut Input, s: &str) {
        for c in s.chars() {
            assert_eq!(inp.key(Key::Char(c)), Outcome::Redraw);
        }
    }

    #[test]
    fn backspace_narrows_menu() {
        let mut h = Vec::new();
        let mut inp = Input::new(&mut h);
        type_str(&mut inp, "/h");
        assert_eq!(inp.menu().len(), 1);
        assert_eq!(inp.key(Key::Backspace), Outcome::Redraw);
        assert_eq!(inp.text(), "/");
        assert_eq!(inp.menu().len(), crate::commands::COMMANDS.len());
    }

    #[test]
    fn set_text_opens_the_menu() {
        let mut h = Vec::new();
        let mut inp = Input::new(&mut h);
        inp.set_context("openrouter", "openai/gpt-4.1", &catalog(), &providers(), crate::config::THINKING_LEVELS);
        inp.set_text("/model ");
        assert_eq!(inp.menu().len(), 4);
        assert_eq!(inp.text(), "/model ");
        // Typing narrows the prefilled picker like any other menu.
        type_str(&mut inp, "gro");
        assert_eq!(inp.menu().len(), 1);
        assert_eq!(inp.menu()[0].name.trim_end(), "x-ai/grok-4");
    }

    #[test]
    fn window_keeps_selection_visible() {
        assert_eq!(window(9, 0, 10), (0, 9));
        assert_eq!(window(9, 8, 10), (0, 9));
        assert_eq!(window(30, 0, 10), (0, 10));
        assert_eq!(window(30, 29, 10), (20, 30));
        assert_eq!(window(30, 7, 10), (2, 12));
        assert_eq!(window(0, 0, 10), (0, 0));
    }

    /// Visible cells on a frame row, ignoring the escapes the frame emits.
    fn visible(row: &str) -> usize {
        let mut n = 0;
        let mut it = row.chars().peekable();
        while let Some(c) = it.next() {
            if c == '\x1b' {
                while let Some(&d) = it.peek() {
                    it.next();
                    if d.is_ascii_alphabetic() {
                        break;
                    }
                }
                continue;
            }
            if c != '\r' {
                n += 1;
            }
        }
        n
    }

    /// Replay frames on a small screen, so a redraw's arithmetic can be checked
    /// against the rows a terminal would really be holding. Enough of one: the
    /// carriage return, the line feed, erase-to-end-of-line, and the three
    /// cursor moves the frame uses.
    fn replay(frames: &[String]) -> (Vec<String>, (usize, usize)) {
        let mut grid = vec![vec![' '; 24]; 12];
        let (mut y, mut x) = (5usize, 0usize);
        for f in frames {
            let b: Vec<char> = f.chars().collect();
            let mut i = 0;
            while i < b.len() {
                match b[i] {
                    '\r' => {
                        x = 0;
                        i += 1;
                    }
                    '\n' => {
                        y += 1;
                        i += 1;
                    }
                    '\x1b' => {
                        // `ESC [`, optional digits, final letter. `?25l` and
                        // `?25h` are the same shape and are skipped.
                        let mut j = i + 2;
                        let mut n = String::new();
                        while b[j].is_ascii_digit() {
                            n.push(b[j]);
                            j += 1;
                        }
                        let step: usize = n.parse().unwrap_or(1);
                        match b[j] {
                            'A' => y -= step,
                            'B' => y += step,
                            'C' => x += step,
                            'K' => grid[y].iter_mut().skip(x).for_each(|c| *c = ' '),
                            _ => {}
                        }
                        while !b[j].is_ascii_alphabetic() {
                            j += 1;
                        }
                        i = j + 1;
                    }
                    c => {
                        // The frame's own rows are one cell per char, prompt
                        // mark included.
                        grid[y][x] = c;
                        x += 1;
                        i += 1;
                    }
                }
            }
        }
        let rows: Vec<String> =
            grid.iter().map(|r| r.iter().collect::<String>().trim_end().to_string()).collect();
        (rows, (y, x))
    }

    /// A redraw owns the rows it paints: text left on a row the cursor has
    /// moved off is the ghosting of one frame per keystroke, which is what the
    /// walk back out of the block exists to prevent.
    #[test]
    fn a_redraw_overwrites_the_rows_the_last_one_painted() {
        // A one-row block that grows a character at a time stays on its row.
        let mut drawn = (0, 0);
        let mut frames = Vec::new();
        for buf in ["", "a", "ab", "abc", "abcd"] {
            let (f, nd) = frame(buf, buf.chars().count(), &[], 0, drawn, 20, 24);
            frames.push(f);
            drawn = nd;
        }
        let (screen, cursor) = replay(&frames);
        assert_eq!(screen[5], "❯ abcd", "{screen:?}");
        assert!(
            screen.iter().enumerate().all(|(i, l)| i == 5 || l.is_empty()),
            "an earlier frame's text is still on screen: {screen:?}"
        );
        assert_eq!(cursor, (5, 6), "the cursor left the block's row: {screen:?}");

        // A block two rows tall: the walk back spans one row, not two, and the
        // cursor ends on the row it reports.
        let text = "one\ntwo";
        let (f, drawn) = frame(text, text.chars().count(), &[], 0, (0, 0), 20, 24);
        let (f2, _) = frame(text, 0, &[], 0, drawn, 20, 24);
        let (screen, cursor) = replay(&[f, f2]);
        assert_eq!(screen[5], "❯ one", "{screen:?}");
        assert_eq!(screen[6], "  two", "{screen:?}");
        assert_eq!(cursor, (5, 2), "the cursor is not on the first row: {screen:?}");
    }

    #[test]
    fn frame_fits_rows_to_width_and_wraps_long_input() {
        let items = vec![Item {
            name: "/provider ".to_string(),
            desc: "show or switch provider (key=/url= inline)".to_string(),
        }];
        let (f, _) = frame("", 0, &items, 0, (0, 0), 40, 24);
        for row in f.split('\n') {
            assert!(visible(row) <= 40, "{row:?} is {} cells", visible(row));
        }
        assert!(f.contains("..."), "{f}");
        // A long line wraps instead of scrolling sideways: every character is
        // on screen, and the cursor row follows it down.
        let long = "a".repeat(60);
        let (f, drawn) = frame(&long, 60, &[], 0, (0, 0), 20, 24);
        assert_eq!(f.matches('a').count(), 60, "{f}");
        assert_eq!(drawn.0, 4, "60 chars over 18-cell rows is 4 rows: {f}");
        assert_eq!(drawn.1, 3, "the cursor is at the end, on the last row: {f}");
        // A row that runs out breaks at a space, not mid-word.
        let (f, drawn) = frame("hello there world", 17, &[], 0, (0, 0), 12, 24);
        assert!(f.contains("❯ hello"), "{f}");
        assert!(f.contains("  there"), "{f}");
        assert!(f.contains("  world"), "{f}");
        assert!(!f.contains("hellothere"), "{f}");
        assert_eq!(drawn.0, 3, "{f}");
        // The menu window is capped by the terminal height.
        let items: Vec<Item> =
            (0..30).map(|i| Item { name: format!("m-{i}"), desc: "p".to_string() }).collect();
        let (_, drawn) = frame("", 0, &items, 0, (0, 0), 100, 6);
        assert_eq!(drawn.0, 6); // the input row + 4 menu rows + v more
    }

    #[test]
    fn a_multi_line_prompt_shows_every_line_with_the_mark_on_the_first() {
        let (f, drawn) = frame("one\ntwo\nthree", 13, &[], 0, (0, 0), 40, 24);
        assert!(f.contains("❯ one"), "{f}");
        assert!(f.contains("  two"), "{f}");
        assert!(f.contains("  three"), "{f}");
        // The mark is spent once: continuation rows align under the text.
        assert_eq!(f.matches('❯').count(), 1, "{f}");
        assert_eq!(drawn, (3, 2), "{f}");
    }

    #[test]
    fn frame_truncates_long_menus_with_hints() {
        let items: Vec<Item> = (0..30)
            .map(|i| Item { name: format!("m-{i}"), desc: "p".to_string() })
            .collect();
        let (f, drawn) = frame("", 0, &items, 25, (0, 0), 100, 30);
        assert_eq!(drawn.0, 12); // the input row + ^ more header + 10 rows
        assert!(f.contains("^ 20 more"), "{f}");
        assert!(!f.contains("v "), "{f}");
        assert!(f.contains("m-25"), "{f}");
        assert!(!f.contains("m-0 "), "{f}");
        // Scrolling back up wipes the old window rows.
        let (f, drawn) = frame("", 0, &items, 2, drawn, 100, 30);
        assert_eq!(drawn.0, 12); // the input row + 10 rows + v more footer
        assert!(f.contains("v 20 more"), "{f}");
        assert!(f.contains("m-0"), "{f}");
        assert!(!f.contains("m-25"), "{f}");
    }

    #[test]
    fn frame_wipes_stale_rows() {
        let (_, items) = completion_items("/", 1, "anthropic", "", &[], &providers(), crate::config::THINKING_LEVELS).unwrap();
        assert_eq!(items.len(), crate::commands::COMMANDS.len());
        // Rows drawn for a menu: the window plus a footer when it overflows.
        let rows_for = |n: usize| 1 + n.min(MENU_ROWS) + usize::from(n > MENU_ROWS);
        let full = rows_for(items.len());
        // Full menu drawn over the one input row: the CLEAR count is the input
        // row plus every menu row.
        let (f, drawn) = frame("/", 1, &items, 0, (1, 0), 100, 30);
        assert_eq!(drawn.0, full);
        assert_eq!(f.matches("\x1b[K").count(), full);
        // The walk back is to the block's top row, which is the row above the
        // last one painted: asking for `full` would leave the cursor a row high.
        assert!(f.contains(&format!("\x1b[{}A", full - 1)));
        // Backspace to "": menu gone, all its stale rows wiped blank.
        let (f, drawn) = frame("", 0, &[], 0, drawn, 100, 30);
        assert_eq!(drawn.0, full);
        assert_eq!(f.matches("\x1b[K").count(), full);
        // Shrinking the menu wipes exactly the surplus rows.
        let small = &items[..3];
        let (f, drawn) = frame("/", 1, small, 0, drawn, 100, 30);
        assert_eq!(drawn.0, full);
        assert_eq!(f.matches("\x1b[K").count(), 1 + small.len() + (full - 1 - small.len()));
    }

    #[test]
    fn a_prompt_taller_than_the_window_scrolls_with_a_count() {
        // 12 lines into a 6-row budget: rows are named, not silently dropped,
        // and the cursor's line is the one kept in view.
        let text: String = (0..12).map(|i| format!("line{i}\n")).collect();
        let text = text.trim_end_matches('\n');
        let chars: Vec<char> = text.chars().collect();
        let (_lines, cursor_row, _n) = input_block(&chars, chars.len(), 40, 6);
        let (f, drawn) = frame(text, chars.len(), &[], 0, (0, 0), 40, 6);
        assert!(f.contains("more lines"), "{f}");
        assert!(f.contains("line11"), "the cursor's line must be on screen: {f}");
        assert!(drawn.0 <= 6, "{drawn:?}");
        assert!(cursor_row > 0, "the cursor is not on the first row: {drawn:?}");
    }

    #[test]
    fn rows_break_on_newlines_and_at_words() {
        let c: Vec<char> = "abc\ndef".chars().collect();
        assert_eq!(
            rows_of(&c, 10),
            vec![Row { start: 0, end: 3 }, Row { start: 4, end: 7 }]
        );
        // A trailing newline opens an empty row for the cursor to sit on.
        let c: Vec<char> = "x\n".chars().collect();
        assert_eq!(rows_of(&c, 10), vec![Row { start: 0, end: 1 }, Row { start: 2, end: 2 }]);
        // Too long for one row: the break lands on the space and the space is
        // consumed, so the continuation does not start with a stray blank.
        let c: Vec<char> = "aa bb cc dd".chars().collect();
        assert_eq!(
            rows_of(&c, 5),
            vec![Row { start: 0, end: 2 }, Row { start: 3, end: 5 }, Row { start: 6, end: 11 }]
        );
        // Every character still belongs to some row: wrapping hides nothing.
        let covered: usize = rows_of(&c, 5).iter().map(|r| r.end - r.start).sum();
        assert_eq!(covered + 2, c.len(), "the two consumed spaces are the only gaps");
        // One unbroken word longer than the row is cut at the edge.
        let c: Vec<char> = "aaaaaaaa".chars().collect();
        assert_eq!(
            rows_of(&c, 3),
            vec![Row { start: 0, end: 3 }, Row { start: 3, end: 6 }, Row { start: 6, end: 8 }]
        );
        assert_eq!(rows_of(&[], 80), vec![Row { start: 0, end: 0 }]);
    }

    #[test]
    fn up_and_down_move_a_row_inside_a_block() {
        let mut h = Vec::new();
        let mut inp = Input::new(&mut h);
        type_str(&mut inp, "one");
        inp.key(Key::Newline);
        type_str(&mut inp, "two");
        inp.key(Key::Newline);
        type_str(&mut inp, "three");
        assert_eq!(inp.text(), "one\ntwo\nthree");
        // Up from the last row lands on "two", keeping the column.
        assert_eq!(inp.key(Key::Up), Outcome::Redraw);
        assert_eq!(inp.cursor_pos(), 7, "the end of two, which is its newline");
        assert_eq!(inp.key(Key::Up), Outcome::Redraw);
        assert_eq!(inp.cursor_pos(), 3, "the end of one");
        // The block's top row walks history when there is any: with none, it
        // stays put, the same hold a one-line prompt has.
        assert_eq!(inp.key(Key::Up), Outcome::Redraw);
        assert_eq!(inp.text(), "one\ntwo\nthree", "no history to walk, the block stays");
    }

    #[test]
    fn a_row_shorter_than_the_cursor_clamps_the_column() {
        let c: Vec<char> = "abcd\nx\nefghij".chars().collect();
        // Cursor at the end of the last row, column 10: moving up to the row
        // with one character on it stops at its end, not past it.
        let up = move_row(&c, c.len(), 40, true).unwrap();
        assert_eq!(up, 6, "the x row's end, which is its newline");
        let up2 = move_row(&c, up, 40, true).unwrap();
        assert_eq!(up2, 4, "the end of abcd is its newline");
        assert_eq!(move_row(&c, up2, 40, true), Err(()));
        let down = move_row(&c, up2, 40, false).unwrap();
        assert_eq!(down, 6, "back to the x row's end");
    }

    #[test]
    fn home_and_end_are_scoped_to_the_line() {
        let mut h = Vec::new();
        let mut inp = Input::new(&mut h);
        type_str(&mut inp, "one");
        inp.key(Key::Newline);
        type_str(&mut inp, "two");
        assert_eq!(inp.cursor_pos(), 7);
        assert_eq!(inp.key(Key::Home), Outcome::Redraw);
        assert_eq!(inp.cursor_pos(), 4, "the start of the second line, not the buffer");
        assert_eq!(inp.key(Key::End), Outcome::Redraw);
        assert_eq!(inp.cursor_pos(), 7);
        // A single-line prompt keeps the buffer-wide behaviour.
        let mut h2 = Vec::new();
        let mut inp2 = Input::new(&mut h2);
        type_str(&mut inp2, "hello");
        inp2.key(Key::Home);
        assert_eq!(inp2.cursor_pos(), 0);
    }

    #[test]
    fn answer_lines_are_broken_into_app_sized_rows() {
        // A line the terminal would fold is broken here instead: every row
        // the app prints stays one row, so nothing lands mid-line.
        let long = "a".repeat(150);
        let rows = display_rows_at(&long, 60);
        assert!(rows.iter().all(|r| r.chars().count() <= 60), "{rows:?}");
        assert_eq!(rows.concat().chars().count(), 150, "no character is lost");
        // A short line passes through untouched.
        assert_eq!(display_rows_at("hello", 60), vec!["hello".to_string()]);
        assert_eq!(display_rows_at("", 60), vec![String::new()]);
        // Prose breaks at a space and the space does not start the next row.
        let rows = display_rows_at("alpha beta gamma", 12);
        assert_eq!(rows[0], "alpha beta", "{rows:?}");
        assert_eq!(rows[1], "gamma", "{rows:?}");
        // Newlines survive as row breaks.
        assert_eq!(
            display_rows_at("one\ntwo", 60),
            vec!["one".to_string(), "two".to_string()]
        );
    }

    #[test]
    fn line_bounds_find_the_hard_line() {
        let c: Vec<char> = "ab\ncd\nef".chars().collect();
        assert_eq!(line_bounds(&c, 0), (0, 2));
        assert_eq!(line_bounds(&c, 2), (0, 2), "the cursor on the newline owns that line");
        assert_eq!(line_bounds(&c, 3), (3, 5));
        assert_eq!(line_bounds(&c, c.len()), (6, 8));
    }

    #[test]
    fn the_cursor_stays_on_the_row_it_belongs_to() {
        let c: Vec<char> = "abcdefghij".chars().collect();
        let rows = rows_of(&c, 4);
        assert_eq!(row_of(&rows, 0), 0);
        assert_eq!(row_of(&rows, 4), 0, "the end of a row is still that row");
        assert_eq!(row_of(&rows, 5), 1);
        assert_eq!(cursor_col(&c, 5, 4), 3);
        assert_eq!(cursor_col(&c, 0, 4), 2);
    }

    #[test]
    fn arrows_cycle_selection() {
        let mut h = Vec::new();
        let mut inp = Input::new(&mut h);
        type_str(&mut inp, "/");
        assert_eq!(inp.sel(), 0);
        assert_eq!(inp.key(Key::Down), Outcome::Redraw);
        assert_eq!(inp.sel(), 1);
        assert_eq!(inp.key(Key::Down), Outcome::Redraw);
        assert_eq!(inp.sel(), 2);
        assert_eq!(inp.key(Key::Up), Outcome::Redraw);
        assert_eq!(inp.sel(), 1);
    }

    #[test]
    fn tab_accepts_highlighted_row() {
        let mut h = Vec::new();
        let mut inp = Input::new(&mut h);
        type_str(&mut inp, "/");
        assert_eq!(inp.key(Key::Down), Outcome::Redraw); // highlight /status
        assert_eq!(inp.key(Key::Tab), Outcome::Redraw);
        assert_eq!(inp.text(), "/status ");
    }

    #[test]
    fn enter_accepts_prefix_then_submits_exact() {
        let mut h = Vec::new();
        let mut inp = Input::new(&mut h);
        type_str(&mut inp, "/h");
        assert_eq!(inp.key(Key::Enter), Outcome::Redraw);
        assert_eq!(inp.text(), "/help ");
        assert_eq!(inp.key(Key::Enter), Outcome::Submit);
        assert_eq!(inp.text(), "/help ");
    }

    #[test]
    fn provider_flow_submits_full_command() {
        let mut h = Vec::new();
        let mut inp = Input::new(&mut h);
        inp.set_context("anthropic", "", &[], &providers(), crate::config::THINKING_LEVELS);
        type_str(&mut inp, "/provider openrouter");
        assert_eq!(inp.key(Key::Enter), Outcome::Submit);
        assert_eq!(inp.text(), "/provider openrouter");
    }

    #[test]
    fn enter_picks_highlighted_provider() {
        let mut h = Vec::new();
        let mut inp = Input::new(&mut h);
        inp.set_context("anthropic", "", &[], &providers(), crate::config::THINKING_LEVELS);
        type_str(&mut inp, "/provider ");
        // Empty fragment + Enter completes the highlighted row instead of
        // submitting the bare prefix.
        assert_eq!(inp.key(Key::Enter), Outcome::Redraw);
        assert_eq!(inp.text(), "/provider anthropic");
        assert_eq!(inp.key(Key::Enter), Outcome::Submit);
    }

    #[test]
    fn arrows_then_enter_picks_openrouter() {
        let mut h = Vec::new();
        let mut inp = Input::new(&mut h);
        inp.set_context("anthropic", "", &[], &providers(), crate::config::THINKING_LEVELS);
        type_str(&mut inp, "/provider ");
        inp.key(Key::Down);
        inp.key(Key::Down);
        inp.key(Key::Down);
        assert_eq!(inp.sel(), 3);
        assert_eq!(inp.key(Key::Enter), Outcome::Redraw);
        assert_eq!(inp.text(), "/provider openrouter");
        assert_eq!(inp.key(Key::Enter), Outcome::Submit);
    }

    #[test]
    fn empty_enter_is_quiet_cancel() {
        let mut h = Vec::new();
        let mut inp = Input::new(&mut h);
        assert_eq!(inp.key(Key::Enter), Outcome::Submit);
        assert_eq!(inp.text(), "");
    }

    #[test]
    fn esc_hides_menu_until_next_edit() {
        let mut h = Vec::new();
        let mut inp = Input::new(&mut h);
        type_str(&mut inp, "/");
        assert_eq!(inp.menu().len(), crate::commands::COMMANDS.len());
        assert_eq!(inp.key(Key::Esc), Outcome::Redraw);
        assert!(inp.menu().is_empty());
        type_str(&mut inp, "h");
        assert_eq!(inp.menu().len(), 1);
    }

    #[test]
    fn history_up_down_restores_draft() {
        let mut h = vec!["aa".to_string(), "bb".to_string()];
        let mut inp = Input::new(&mut h);
        assert_eq!(inp.key(Key::Up), Outcome::Redraw);
        assert_eq!(inp.text(), "bb");
        assert_eq!(inp.key(Key::Up), Outcome::Redraw);
        assert_eq!(inp.text(), "aa");
        assert_eq!(inp.key(Key::Down), Outcome::Redraw);
        assert_eq!(inp.text(), "bb");
        assert_eq!(inp.key(Key::Down), Outcome::Redraw);
        assert_eq!(inp.text(), "");
    }

    #[test]
    fn submit_pushes_history() {
        let mut h = Vec::new();
        {
            let mut inp = Input::new(&mut h);
            type_str(&mut inp, "hello");
            assert_eq!(inp.key(Key::Enter), Outcome::Submit);
        }
        assert_eq!(h, vec!["hello".to_string()]);
    }

    #[test]
    fn ctrl_c_clears_line() {
        let mut h = Vec::new();
        let mut inp = Input::new(&mut h);
        type_str(&mut inp, "hello");
        assert_eq!(inp.key(Key::CtrlC), Outcome::Submit);
        assert_eq!(inp.text(), "");
    }

    #[test]
    fn output_parts_keeps_both_ends() {
        // Fits: one block, nothing hidden.
        let (head, hidden, tail) = output_parts("a\nb\nc", 8, 4);
        assert_eq!((head.as_str(), hidden, tail.as_deref()), ("a\nb\nc", None, None));
        // Too long: head, count, tail — so the `[exit N]` trailer survives.
        let text: String = (1..=20).map(|i| format!("l{i}\n")).collect();
        let (head, hidden, tail) = output_parts(&text, 8, 4);
        assert_eq!(head.lines().count(), 8);
        assert_eq!(hidden, Some(8));
        let tail = tail.unwrap();
        assert_eq!(tail.lines().count(), 4);
        assert!(tail.ends_with("l20"), "{tail}");
        assert!(head.starts_with("l1"), "{head}");
    }

    #[test]
    fn clamp_marks_truncation() {
        assert_eq!(clamp("abc", 5), "abc");
        // The result never exceeds the budget, marker included.
        assert_eq!(clamp("abcdef", 4), "a...");
        assert_eq!(clamp("abcdef", 6), "abcdef");
        assert_eq!(clamp("aaa", 5), "aaa");
        assert_eq!(clamp("aaaaa", 3), "...");
        assert_eq!(clamp("aaaaaa", 3), "...");
    }

    #[test]
    fn clamp_words_breaks_at_a_space() {
        let notice = "... 3 more lines (full output sent to model)";
        assert_eq!(clamp_words(notice, 80), notice);
        assert_eq!(clamp_words(notice, 40), "... 3 more lines (full output sent...");
        assert_eq!(clamp_words("aaaaaaaaaa", 5), "aa...");
        assert!(clamp_words("aaaaaaaaaaaaaaaa", 5).chars().count() <= 5);
    }

    #[test]
    fn the_banner_puts_the_mcp_count_beside_the_model() {
        let line = strip_sgr(&header_line("tokenharbor", "deepseek-v4.1-flash-1017", 1, 80));
        assert!(line.starts_with("rift - shell agent"), "{line:?}");
        assert!(
            line.ends_with("deepseek-v4.1-flash-1017 - mcp 1"),
            "{line:?}"
        );
        // The badge should not be padded out to the terminal edge.
        assert!(line.chars().count() < 80, "{line:?}");
        // No servers is quiet rather than a `mcp 0` badge.
        let none = strip_sgr(&header_line("tokenharbor", "deepseek-v4.1-flash-1017", 0, 80));
        assert!(!none.contains("mcp"), "{none:?}");
        // A narrow window drops the status instead of wrapping.
        let narrow = strip_sgr(&header_line("tokenharbor", "deepseek-v4.1-flash-1017", 3, 20));
        assert!(narrow.chars().count() <= 20, "{narrow:?}");
        assert!(!narrow.contains("mcp"), "{narrow:?}");
    }

    #[test]
    fn the_tagline_opens_from_the_middle_and_settles_plain() {
        assert_eq!(reveal_range(0, 3, 7), (0, 0));
        assert_eq!(reveal_range(10, 0, 7), (0, 0));
        assert_eq!(reveal_range(10, 7, 7), (0, 10));
        let (lo, hi) = reveal_range(10, 3, 7);
        assert_eq!((lo, hi), (3, 7));
        // The last frame is the exact line; earlier frames keep its width and
        // leave spaces where the words will sit.
        assert_eq!(tagline_frame("your personal hacker", 7, 7, 1), "your personal hacker");
        let early = tagline_frame("ab cd", 0, 7, 1);
        assert_eq!(early.chars().count(), 5, "{early:?}");
        assert_eq!(early.chars().nth(2), Some(' '), "{early:?}");
        // The middle locks before the edges, so both ends are still cipher.
        let middle = tagline_frame("abcdef", 4, 7, 1);
        let chars: Vec<char> = middle.chars().collect();
        assert_eq!(&chars[2..5], ['c', 'd', 'e']);
        assert!(CIPHER.iter().all(|b| b.is_ascii_graphic()), "a glyph is not ASCII");
        assert_eq!(tagline_final("x"), ">_ x");
    }

    #[test]
    fn other_bullet_characters_become_the_one_this_ui_uses() {
        let mut fence = false;
        assert_eq!(render_line("* one", &mut fence, false), "- one");
        assert_eq!(render_line("+ two", &mut fence, false), "- two");
        assert_eq!(render_line("  * indented", &mut fence, false), "  - indented");
        // Arithmetic keeps its numbers, loses the marker, and does not leave
        // a double gap behind.
        assert_eq!(render_line("2 * 3", &mut fence, false), "2 3");
        assert_eq!(render_line("- already fine", &mut fence, false), "- already fine");
    }

    #[test]
    fn a_restored_call_keeps_the_outcome_its_result_recorded() {
        // A resumed session has calls and results as separate messages, so the
        // dot colour has to be recovered by pairing them.
        let _on = color_fixed(true);
        use crate::provider::{Msg, ToolCall};
        let msgs = vec![
            Msg::Assistant {
                text: String::new(),
                calls: vec![
                    ToolCall::shell("ok1", "ls"),
                    ToolCall::shell("bad1", "false"),
                    ToolCall::shell("none1", "echo hi"),
                ],
            },
            Msg::ToolResult { id: "ok1".into(), output: "a\n[exit 0]".into() },
            Msg::ToolResult { id: "bad1".into(), output: "a\n[exit 1]".into() },
            // A result that never arrived has no outcome, so it is not green.
            Msg::ToolResult { id: "other".into(), output: "x\n[exit 0]".into() },
        ];
        let t = transcript_text(&msgs, 80);
        assert!(t.contains("\x1b[32m\u{25cf}\x1b[0m ls"), "{t:?}");
        assert!(t.contains("\x1b[31m\u{25cf}\x1b[0m false"), "{t:?}");
        assert!(t.contains("\x1b[31m\u{25cf}\x1b[0m echo hi"), "{t:?}");
    }

    #[test]
    fn transcript_shows_what_was_said_and_what_ran() {
        use crate::provider::{Msg, ToolCall};
        let msgs = vec![
            Msg::User("fix the parser".into()),
            Msg::Assistant {
                text: "## Done\n\nChanged the lexer.\n".into(),
                calls: vec![ToolCall::shell("1", "cargo test")],
            },
            Msg::ToolResult {
                id: "1".into(),
                output: "running 78 tests\nall ok\n[exit 0]\n".into(),
            },
            Msg::User("now the docs".into()),
        ];
        let t = transcript_text(&msgs, 60);
        assert!(t.contains("❯ fix the parser"), "{t}");
        assert!(t.contains("Done"), "{t}");
        assert!(t.contains("Changed the lexer."), "{t}");
        assert!(t.contains("● cargo test"), "{t}");
        assert!(t.contains("└ running 78 tests"), "{t}");
        assert!(t.contains("... 2 more lines"), "{t}");
        assert!(t.contains("❯ now the docs"), "{t}");
        // Nothing ran in the last message, and no raw tool output is dumped.
        assert!(!t.contains("all ok"), "{t}");
    }

    /// Strip SGR sequences, for asserting on layout without the paint.
    fn strip_sgr(s: &str) -> String {
        let mut out = String::new();
        let mut it = s.chars();
        while let Some(c) = it.next() {
            if c == '\u{1b}' {
                for d in it.by_ref() {
                    if d.is_ascii_alphabetic() {
                        break;
                    }
                }
                continue;
            }
            out.push(c);
        }
        out
    }

    #[test]
    fn the_stats_frame_is_aligned_and_quietly_coloured() {
        let _on = color_fixed(true);
        let body = stats_frame(None);
        // A heading per section, and the numbers are the only unstyled text:
        // every label and note is dim, so the values are the brightest thing.
        assert!(body.contains("model usage"), "{body:?}");
        assert!(body.contains("this process"), "{body:?}");
        let plain = strip_sgr(&body);
        let value_col = plain
            .lines()
            .filter(|l| l.starts_with("  ") && !l.trim().is_empty())
            .map(|l| l.find(|c: char| c.is_ascii_alphanumeric()).unwrap_or(0))
            .collect::<Vec<_>>();
        assert!(value_col.len() >= 4, "{plain}");
        assert!(value_col.windows(2).all(|w| w[0] == w[1]), "ragged: {value_col:?}");
        // The value itself is never painted: the only dim spans on a row are
        // the label and the note, so the escape pairs must bracket those and
        // not the text between them.
        let mut checked = 0;
        for line in body.lines() {
            let Some(rest) = line.strip_prefix("  ") else { continue };
            if rest.trim().is_empty() || !rest.contains("\x1b[") {
                continue;
            }
            // Label first, then unpainted text, then an optional dim note.
            assert!(rest.starts_with("\x1b[2m"), "label not dim: {line:?}");
            let after_label = rest.find("\x1b[0m").map(|i| &rest[i + 4..]).unwrap_or("");
            let value = after_label.split("  \x1b[2m").next().unwrap_or("");
            assert!(!value.is_empty(), "no value: {line:?}");
            assert!(!value.contains('\u{1b}'), "value is painted: {value:?}");
            checked += 1;
        }
        assert!(checked >= 4, "checked only {checked} rows");
    }

    #[test]
    fn only_the_first_piece_of_output_takes_the_corner() {
        assert_eq!(indent_block("one\ntwo", 40, true), "└ one\n  two");
        assert_eq!(indent_block("nine\nten", 40, false), "  nine\n  ten");
        // Leading blanks would leave the corner dangling on its own line.
        assert_eq!(indent_block("\n\nreal", 40, true), "└ real");
        // A runner marker on the cornered line keeps its dim treatment.
        assert_eq!(indent_block("[exit 0]", 40, true), "└ [exit 0]");
    }

    #[test]
    fn working_indicator_sits_before_the_mark() {
        use std::sync::atomic::Ordering;
        assert_eq!(live_line("hi"), "❯ hi");
        RUNNING.fetch_add(1, Ordering::Relaxed);
        *WORK.lock().unwrap() = Some(("running".to_string(), std::time::Instant::now()));
        let s = live_line("hi");
        assert!(s.ends_with("running  ❯ hi"), "{s}");
        // Whichever frame is current, it must be one of the set in use.
        assert!(frames().iter().any(|f| s.starts_with(f)), "{s}");
        RUNNING.fetch_sub(1, Ordering::Relaxed);
        *WORK.lock().unwrap() = None;
        assert_eq!(live_line("hi"), "❯ hi");
    }

    #[test]
    fn a_long_live_line_stays_on_one_row() {
        let long = "x".repeat(crate::term::cols() + 40);
        let s = live_line(&long);
        assert!(
            visible_len(&s) <= crate::term::cols(),
            "live row wraps: {} cells in {} columns",
            visible_len(&s),
            crate::term::cols()
        );
        assert!(s.ends_with("..."), "the cut is marked: {s:?}");
    }

    #[test]
    fn elapsed_suffix_waits_for_a_real_wait() {
        use std::time::Duration;
        assert_eq!(elapsed_suffix(Duration::from_millis(900)), "");
        assert_eq!(elapsed_suffix(Duration::from_millis(1900)), "");
        assert_eq!(elapsed_suffix(Duration::from_secs(2)), " (2s)");
        assert_eq!(elapsed_suffix(Duration::from_secs(59)), " (59s)");
        assert_eq!(elapsed_suffix(Duration::from_secs(64)), " (1m04s)");
    }

    #[test]
    fn md_renders_and_strips() {
        let mut fence = false;
        // Headings keep their words and lose their hashes.
        assert_eq!(render_line("### What it does:", &mut fence, false), "What it does:");
        // Bold, italic, code and links all keep only what they wrapped.
        assert_eq!(render_line("**rift** - x", &mut fence, false), "rift - x");
        assert_eq!(render_line("use `shell` tool", &mut fence, false), "use shell tool");
        assert_eq!(render_line("[text](http://u)", &mut fence, false), "text");
        // A rule is punctuation with no words behind it, so it goes entirely.
        assert_eq!(render_line("---", &mut fence, false), "");
        assert_eq!(render_line("***", &mut fence, false), "");
        // Bullets are unified, and a quote keeps its own marker.
        assert_eq!(render_line("- plain item", &mut fence, false), "- plain item");
        assert_eq!(render_line("> quoted", &mut fence, false), "> quoted");
        // Indentation survives, because it is the model's own structure.
        assert_eq!(render_line("    indented", &mut fence, false), "    indented");
        // A stray asterisk is dropped even when it matches nothing: this
        // surface has no emphasis, so the character never reaches the screen.
        assert_eq!(render_line("a **b and `c", &mut fence, false), "a b and `c");
        assert_eq!(render_line("2 * 3 * 4", &mut fence, false), "2 3 4");
        // An unmatched backtick still passes through: it is not emphasis.
        assert_eq!(render_line("odd ` here", &mut fence, false), "odd ` here");
    }

    #[test]
    fn md_tables_stay_aligned_in_ascii() {
        let mut md = Md::new();
        // The rule spans its column, rows line up under it, no box drawing.
        assert_eq!(
            md.push("| file | role |\n|---|---|\n| `ui.rs` | terminal I/O |\n| `x` | runner |\nend\n"),
            "file   role\n-----  ------------\nui.rs  terminal I/O\nx      runner\nend\n"
        );
        // A table that never ends still renders on flush.
        let mut md = Md::new();
        assert_eq!(md.push("| a | b |\n|---|---|\n"), "");
        assert_eq!(md.flush(), "a  b\n-  -\n");
        assert_eq!(md.flush(), "");
    }

    #[test]
    fn the_output_surface_is_ascii() {
        // Every string a user reads must survive a font that has no box
        // drawing and no arrows: the whole point of the plain vocabulary.
        let mut fence = false;
        let samples = [
            render_line("### H", &mut fence, false),
            render_line("| a | b |", &mut fence, false),
            render_line("- item", &mut fence, false),
            format!("{} cargo test", TOOL),
            format!("{} [exit 0]", OUT),
            echo_line("hi"),
            paint(GREEN, &format!("ok {s}", s = "saved")),
            paint(YELLOW, &format!("! {s}", s = "careful")),
            paint(RED, &format!("error: {s}", s = "bad")),
        ];
        for s in samples {
            // The two marks the user kept are the only non-ASCII cells allowed.
            let stripped = s
                .replace(PROMPT, "")
                .replace(TOOL, "")
                .replace(OUT, "");
            assert!(stripped.is_ascii(), "{s:?} carries a character outside ASCII");
        }
    }

    #[test]
    fn the_spinner_is_a_single_cell_that_a_monospace_font_has() {
        // Every frame must be exactly one char, or the row's column arithmetic
        // is wrong; and the UTF-8 set must avoid the blocks monospace fonts
        // are known to be missing, braille U+28xx above all.
        for set in [&FRAMES[..], &FRAMES_ASCII[..]] {
            for f in set {
                assert_eq!(f.chars().count(), 1, "{f:?} is not one cell");
                assert!(!f.trim().is_empty(), "a blank frame reads as a stalled spinner");
            }
        }
        for f in FRAMES {
            let c = f.chars().next().unwrap();
            assert!(
                !('\u{2800}'..='\u{28ff}').contains(&c),
                "{f:?} is braille, which falls back to a proportional font"
            );
        }
        // The ASCII set is the fallback and must be genuinely ASCII.
        assert!(FRAMES_ASCII.iter().all(|f| f.is_ascii()));
    }

    #[test]
    fn the_frames_turn_in_one_direction_without_repeating() {
        // A rotation reads as motion only if no frame is drawn twice in a turn.
        for set in [&FRAMES[..], &FRAMES_ASCII[..]] {
            let mut seen: Vec<&str> = Vec::new();
            for i in 0..set.len() {
                let f = frame_in(set, i);
                assert!(!seen.contains(&f), "{f:?} repeats inside one turn");
                seen.push(f);
            }
            // Wrapping is what lets the caller count forever.
            assert_eq!(frame_in(set, set.len()), frame_in(set, 0));
        }
        // The rotation's period is the frame count times the hold time.
        assert_eq!(FRAMES.len() as u64 * FRAME_MS, 640);
    }

    #[test]
    fn a_non_utf8_terminal_gets_the_ascii_frames() {
        // Both answers, without touching the process environment: the tests in
        // this binary share one process and run in parallel.
        assert!(locale_is_utf8("en_US.UTF-8"));
        assert!(locale_is_utf8("C.utf8"));
        assert!(locale_is_utf8("utf-8"));
        assert!(!locale_is_utf8("C"), "the C locale is not UTF-8");
        assert!(!locale_is_utf8("en_US.ISO-8859-1"));
        assert!(!locale_is_utf8(""));
        // The set follows the answer.
        assert_eq!(frames_for(true), &FRAMES);
        assert_eq!(frames_for(false), &FRAMES_ASCII);
        assert!(frames_for(false).iter().all(|f| f.is_ascii()));
    }

    #[test]
    fn the_colour_answer_does_not_depend_on_the_terminal_that_ran_the_test() {
        assert!(!color(), "a test run must default to plain text");
        {
            let _on = color_fixed(true);
            assert!(color(), "the fix was ignored");
            assert!(paint(DIM, "x").starts_with('\u{1b}'), "paint ignored the fix");
        }
        assert!(!color(), "the fix outlived its scope");
    }

    #[test]
    fn only_the_dot_carries_the_outcome() {
        let _on = color_fixed(true);
        // Green dot on success, red on failure; the command text is unstyled.
        assert_eq!(
            tool_echo("cargo test", true),
            "\x1b[32m\u{25cf}\x1b[0m cargo test"
        );
        assert_eq!(
            tool_echo("cargo test", false),
            "\x1b[31m\u{25cf}\x1b[0m cargo test"
        );
        // Exactly one escape sequence: the dot, and nothing else on the line.
        assert_eq!(tool_echo("cargo test", true).matches('\u{1b}').count(), 2);
        // With colour off, the line is plain text.
        let _off = color_fixed(false);
        assert_eq!(tool_echo("cargo test", true), "\u{25cf} cargo test");
    }

    #[test]
    fn md_colors_when_tty() {
        let mut fence = false;
        // Nothing inline is styled: the words are the output, the markers are
        // not decoration worth a second weight.
        assert_eq!(render_line("**b** and `c`", &mut fence, true), "b and c");
        // A heading is the one role that carries a weight, and it is dim.
        assert_eq!(render_line("## H", &mut fence, true), "\x1b[2mH\x1b[0m");
    }

    #[test]
    fn md_fence_passes_code_through_untouched() {
        let mut fence = false;
        // The fence line itself is scaffolding and is dropped.
        assert_eq!(render_line("```rust", &mut fence, false), "");
        assert!(fence);
        // Inside the fence nothing is stripped, not even markdown-looking text.
        assert_eq!(render_line("**not bold**", &mut fence, false), "**not bold**");
        assert_eq!(render_line("  indented", &mut fence, false), "  indented");
        assert_eq!(render_line("```", &mut fence, false), "");
        assert!(!fence);
        // Outside it again, markers are stripped.
        assert_eq!(render_line("**bold**", &mut fence, false), "bold");
    }

    #[test]
    fn md_push_renders_split_markers() {
        let mut md = Md::new();
        // A bold marker split across stream chunks still strips cleanly.
        assert_eq!(md.push("**ri"), "");
        assert_eq!(md.push("ft** ok\nnext"), "rift ok\n");
        assert_eq!(md.flush(), "next");
        assert_eq!(md.flush(), "");
    }
}
