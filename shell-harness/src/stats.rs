//! Session telemetry for `/stats`: what this process used while it ran, and
//! what the model billed for the tokens it wrote. Numbers only; the report
//! formatting is separate so it can be asserted exactly.

use std::sync::Mutex;
use std::time::{Duration, Instant};

/// One model call's token accounting. `input` is the *uncached* prompt only,
/// so `cache_read` is never counted twice inside `prompt()`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Usage {
    pub input: u64,
    pub cache_read: u64,
    pub cache_write: u64,
    pub output: u64,
}

impl Usage {
    /// Every token the provider had to read: uncached, newly written, re-read.
    pub fn prompt(&self) -> u64 {
        self.input + self.cache_read + self.cache_write
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct Inner {
    requests: u64,
    usage: Usage,
    /// Summed seconds to first token, and summed seconds spent generating.
    ttft: f64,
    gen: f64,
    cpu_sum: f64,
    cpu_samples: u64,
    mem_kb_sum: u64,
    mem_samples: u64,
}

impl Inner {
    const fn new() -> Self {
        Inner {
            requests: 0,
            usage: Usage { input: 0, cache_read: 0, cache_write: 0, output: 0 },
            ttft: 0.0,
            gen: 0.0,
            cpu_sum: 0.0,
            cpu_samples: 0,
            mem_kb_sum: 0,
            mem_samples: 0,
        }
    }
}

static INNER: Mutex<Inner> = Mutex::new(Inner::new());

/// A plain snapshot, so the report can be formatted and tested without locks.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct View {
    pub requests: u64,
    pub usage: Usage,
    /// Mean seconds to the first streamed token.
    pub avg_ttft_secs: f64,
    /// Output tokens per second of generating time. `None` when the stream was
    /// too quick to measure, because a denominator near zero reports a number
    /// in the tens of thousands that means nothing.
    pub avg_speed: Option<f64>,
    /// Mean CPU used, as a percentage of one core. `None` before any sample.
    pub cpu_percent: Option<f64>,
    /// Mean resident memory in MiB. `None` before any sample.
    pub mem_mib: Option<f64>,
    pub cores: usize,
    /// Cached share of all prompt tokens. `None` when nothing was sent yet.
    pub cache_hit_percent: Option<f64>,
}

/// Output tokens per second, or `None` when the measurement is too short to
/// mean anything. A local model on loopback finishes in milliseconds and would
/// otherwise be reported as tens of thousands of tokens per second.
fn measure_speed(output: u64, gen_secs: f64, requests: u64) -> Option<f64> {
    if requests == 0 || gen_secs <= 0.0 {
        return None;
    }
    // Under 50ms of generating time per call, the number is scheduling noise.
    if gen_secs / (requests as f64) < 0.05 {
        return None;
    }
    Some(output as f64 / gen_secs)
}

/// Record one finished model call.
pub fn record(usage: Usage, ttft: Duration, gen: Duration) {
    if let Ok(mut s) = INNER.lock() {
        s.requests += 1;
        s.usage.input += usage.input;
        s.usage.cache_read += usage.cache_read;
        s.usage.cache_write += usage.cache_write;
        s.usage.output += usage.output;
        s.ttft += ttft.as_secs_f64();
        s.gen += gen.as_secs_f64();
    }
}

/// The current snapshot.
pub fn view() -> View {
    let s = *INNER.lock().unwrap_or_else(|e| e.into_inner());
    let prompt = s.usage.prompt();
    View {
        requests: s.requests,
        usage: s.usage,
        avg_ttft_secs: if s.requests > 0 { s.ttft / s.requests as f64 } else { 0.0 },
        avg_speed: measure_speed(s.usage.output, s.gen, s.requests),
        cpu_percent: (s.cpu_samples > 0).then(|| s.cpu_sum / s.cpu_samples as f64 * 100.0),
        mem_mib: (s.mem_samples > 0)
            .then(|| s.mem_kb_sum as f64 / s.mem_samples as f64 / 1024.0),
        cores: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
        cache_hit_percent: (prompt > 0).then(|| s.usage.cache_read as f64 / prompt as f64 * 100.0),
    }
}

/// Sample this process once a second for as long as it lives. Cheap, and it
/// makes the CPU and memory figures means over the session rather than a
/// reading taken at the moment `/stats` was typed.
pub fn spawn_sampler() {
    // Not a terminal-driven tool: a headless one-shot run has no one to read it.
    std::thread::spawn(|| {
        let hz = clock_ticks_per_sec();
        let mut prev_ticks = cpu_ticks().unwrap_or(0);
        let mut prev_at = Instant::now();
        loop {
            std::thread::sleep(Duration::from_secs(1));
            let now = Instant::now();
            let ticks = cpu_ticks().unwrap_or(prev_ticks);
            let wall = now.duration_since(prev_at).as_secs_f64();
            let cpu = ticks.saturating_sub(prev_ticks) as f64 / hz;
            prev_ticks = ticks;
            prev_at = now;
            let mem = rss_kb().unwrap_or(0);
            if let Ok(mut s) = INNER.lock() {
                if wall > 0.0 {
                    // A crashed sample would otherwise poison the mean with a
                    // wild ratio; one core's worth is a sane ceiling per tick.
                    s.cpu_sum += (cpu / wall).min(1.0);
                    s.cpu_samples += 1;
                }
                if mem > 0 {
                    s.mem_kb_sum += mem;
                    s.mem_samples += 1;
                }
            }
        }
    });
}

/// One line of the report: what it is, what was measured, and the context that
/// makes the figure meaningful.
pub struct Line {
    pub label: &'static str,
    pub value: String,
    pub note: String,
}

impl Line {
    fn new(label: &'static str, value: impl Into<String>, note: impl Into<String>) -> Self {
        Line { label, value: value.into(), note: note.into() }
    }
}

/// The `/stats` report as data.
///
/// Layout and colour belong to the UI, which owns every other visual decision;
/// this returns what was measured and nothing about how it looks. The sections
/// are ordered so the model's own cost comes first and this process's resource
/// use second: that is the order the two are usually wondered about.
/// `(used prompt tokens, context window)` as of the last request, when there
/// has been one.
pub type ContextFigures = Option<(u64, u64)>;

pub fn lines(ctx: ContextFigures) -> Vec<(&'static str, Vec<Line>)> {
    sections_of(&view(), ctx)
}

/// The sections for an explicit snapshot. Separate from `lines` so the report
/// can be asserted against fixed figures rather than the live process.
fn sections_of(v: &View, ctx: ContextFigures) -> Vec<(&'static str, Vec<Line>)> {
    let mut usage = vec![Line::new("requests", v.requests.to_string(), "")];
    if v.requests == 0 {
        usage.push(Line::new("tokens", "none yet", "no model call made"));
        usage.push(Line::new("speed", "none yet", ""));
        usage.push(Line::new("cache hit", "none yet", ""));
        usage.push(Line::new("context", "none yet", "no model call made"));
    } else {
        // How much of the window the last request filled.
        usage.push(match ctx {
            Some((used, window)) => Line::new(
                "context",
                format!("{} of ~{} ({}%)", num(used), num(window), used * 100 / window.max(1)),
                "last request, as the model saw it",
            ),
            None => Line::new("context", "n/a", "no measurement yet"),
        });
        usage.push(Line::new(
            "tokens",
            format!("{} in, {} out", num(v.usage.prompt()), num(v.usage.output)),
            String::new(),
        ));
        usage.push(match v.avg_speed {
            Some(sp) => Line::new(
                "speed",
                format!("{sp:.1} tok/s"),
                format!("first token {:.2}s", v.avg_ttft_secs),
            ),
            // A loopback model finishes too fast to time; saying so is better
            // than printing a number in the tens of thousands.
            None => Line::new("speed", "n/a", "finished too fast to time"),
        });
        usage.push(match v.cache_hit_percent {
            Some(h) => Line::new(
                "cache hit",
                format!("{h:.1}%"),
                format!("{} of {} prompt tokens reused", num(v.usage.cache_read), num(v.usage.prompt())),
            ),
            None => Line::new("cache hit", "n/a", "no prompt tokens sent"),
        });
    }
    let process = vec![
        match v.cpu_percent {
            Some(c) => Line::new("cpu", format!("{c:.1}% of one core"), format!("{} cores", v.cores)),
            None => Line::new("cpu", "n/a", "nothing sampled yet"),
        },
        match v.mem_mib {
            Some(m) => Line::new("memory", format!("{m:.1} MB"), "average while running"),
            None => Line::new("memory", "n/a", "nothing sampled yet"),
        },
    ];
    vec![("model usage", usage), ("this process", process)]
}

/// The plain-text rendering of a prepared section list. Labels and notes are
/// passed through `dim` first, so the UI can colour them while plain-text
/// callers and tests see bare text; the values are always left unpainted,
/// because the numbers are the point.
pub fn render(sections: &[(&'static str, Vec<Line>)], dim: &dyn Fn(&str) -> String) -> String {
    let width = sections
        .iter()
        .flat_map(|(_, rows)| rows.iter())
        .map(|r| r.label.chars().count())
        .max()
        .unwrap_or(0);
    let mut out = String::new();
    for (i, (heading, rows)) in sections.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        out.push_str(&dim(heading));
        out.push('\n');
        for r in rows {
            let pad = " ".repeat(width - r.label.chars().count());
            let mut line = format!("  {}{}  {}", dim(r.label), pad, r.value);
            if !r.note.is_empty() {
                line.push_str(&format!("  {}", dim(&r.note)));
            }
            out.push_str(&line);
            out.push('\n');
        }
    }
    out.trim_end().to_string()
}

/// Thousands separators: token counts get long enough to misread without them.
pub(crate) fn num(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// Units of `cpu_ticks` per second: clock ticks from `sysconf` on Linux.
#[cfg(target_os = "linux")]
fn clock_ticks_per_sec() -> f64 {
    let hz = unsafe { libc::sysconf(libc::_SC_CLK_TCK) };
    if hz > 0 { hz as f64 } else { 100.0 }
}

/// `cpu_ticks` reports microseconds of CPU time on the other Unixes.
#[cfg(all(unix, not(target_os = "linux")))]
fn clock_ticks_per_sec() -> f64 {
    1_000_000.0
}

/// `cpu_ticks` reports 100 ns units of CPU time on Windows.
#[cfg(windows)]
fn clock_ticks_per_sec() -> f64 {
    10_000_000.0
}

/// This process's own CPU time, in clock ticks, from `/proc/self/stat`.
#[cfg(target_os = "linux")]
fn cpu_ticks() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/stat").ok()?;
    // The comm field may hold spaces and parens, so fields begin after the
    // last ')' — one bug this avoids is a process named "(foo) bar".
    let rest = s.rsplit_once(')')?.1;
    let f: Vec<&str> = rest.split_whitespace().collect();
    let utime: u64 = f.get(11)?.parse().ok()?;
    let stime: u64 = f.get(12)?.parse().ok()?;
    Some(utime + stime)
}

/// This process's own CPU time in microseconds, from `getrusage`.
#[cfg(all(unix, not(target_os = "linux")))]
fn cpu_ticks() -> Option<u64> {
    let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) } != 0 {
        return None;
    }
    let secs = ru.ru_utime.tv_sec + ru.ru_stime.tv_sec;
    let usecs = ru.ru_utime.tv_usec + ru.ru_stime.tv_usec;
    Some(secs as u64 * 1_000_000 + usecs as u64)
}

/// This process's own CPU time in 100 ns units, from `GetProcessTimes`.
#[cfg(windows)]
fn cpu_ticks() -> Option<u64> {
    use windows_sys::Win32::Foundation::FILETIME;
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetProcessTimes};

    fn units(t: FILETIME) -> u64 {
        (t.dwHighDateTime as u64) << 32 | t.dwLowDateTime as u64
    }

    let mut created: FILETIME = unsafe { std::mem::zeroed() };
    let mut exited: FILETIME = unsafe { std::mem::zeroed() };
    let mut kernel: FILETIME = unsafe { std::mem::zeroed() };
    let mut user: FILETIME = unsafe { std::mem::zeroed() };
    let ok = unsafe {
        GetProcessTimes(GetCurrentProcess(), &mut created, &mut exited, &mut kernel, &mut user)
    };
    if ok == 0 {
        return None;
    }
    Some(units(kernel) + units(user))
}

/// Resident set size in KiB, from `/proc/self/statm`.
#[cfg(target_os = "linux")]
fn rss_kb() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/statm").ok()?;
    let pages: u64 = s.split_whitespace().nth(1)?.parse().ok()?;
    Some(pages * (page_size() / 1024))
}

/// Resident set size in KiB, from `getrusage`. This is the peak high-water
/// mark on macOS, not the current size.
#[cfg(all(unix, not(target_os = "linux")))]
fn rss_kb() -> Option<u64> {
    let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) } != 0 {
        return None;
    }
    Some(ru.ru_maxrss as u64 / 1024)
}

/// Resident set size in KiB, from `GetProcessMemoryInfo`.
#[cfg(windows)]
fn rss_kb() -> Option<u64> {
    use windows_sys::Win32::System::ProcessStatus::{
        GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    let mut c: PROCESS_MEMORY_COUNTERS = unsafe { std::mem::zeroed() };
    c.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
    if unsafe { GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) } == 0 {
        return None;
    }
    Some(c.WorkingSetSize as u64 / 1024)
}

/// Page size in bytes; only the `/proc/self/statm` reader needs it.
#[cfg(target_os = "linux")]
fn page_size() -> u64 {
    let ps = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if ps > 0 { ps as u64 } else { 4096 }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fixed figures pushed through the real section builder and renderer.
    fn report_of(v: &View) -> String {
        let sections = sections_of(v, None);
        render(&sections, &|s| s.to_string())
    }

    fn sample(requests: u64, input: u64, cached: u64, output: u64) -> View {
        let usage = Usage { input, cache_read: cached, cache_write: 0, output };
        View {
            requests,
            usage,
            avg_ttft_secs: 1.25,
            avg_speed: Some(42.5),
            cpu_percent: Some(12.34),
            mem_mib: Some(48.2),
            cores: 8,
            cache_hit_percent: {
                let p = usage.prompt();
                (p > 0).then(|| cached as f64 / p as f64 * 100.0)
            },
        }
    }

    #[test]
    fn usage_prompt_counts_every_kind_of_input() {
        let u = Usage { input: 100, cache_read: 250, cache_write: 50, output: 9 };
        assert_eq!(u.prompt(), 400);
    }

    #[test]
    fn report_shows_every_figure_asked_for() {
        let out = report_of(&sample(3, 12000, 9000, 2331));
        assert!(out.contains("model usage"), "{out}");
        assert!(out.contains("this process"), "{out}");
        assert!(out.contains("21,000 in, 2,331 out"), "{out}");
        assert!(out.contains("42.5 tok/s"), "{out}");
        assert!(out.contains("first token 1.25s"), "{out}");
        // 9000 of 21000 = 42.9%
        assert!(out.contains("42.9%"), "{out}");
        assert!(out.contains("9,000 of 21,000 prompt tokens reused"), "{out}");
        assert!(out.contains("12.3% of one core"), "{out}");
        assert!(out.contains("8 cores"), "{out}");
        assert!(out.contains("48.2 MB"), "{out}");
        assert!(out.contains("requests"), "{out}");
        assert!(out.contains("3"), "{out}");
    }

    #[test]
    fn report_says_so_before_anything_happened() {
        let out = report_of(&View { cores: 8, ..View::default() });
        assert!(out.contains("requests"), "{out}");
        assert!(out.contains("none yet"), "{out}");
        // Nothing sampled and nothing sent both say so rather than showing 0:
        // the four usage figures say "none yet", the two process ones "n/a".
        assert_eq!(out.matches("none yet").count(), 4, "{out}");
        assert_eq!(out.matches("n/a").count(), 2, "{out}");
    }

    #[test]
    fn a_report_reads_back_and_stays_inside_a_80_column_terminal() {
        let long = report_of(&sample(12, 3_800_000, 900_000, 1_250_000));
        for line in long.lines() {
            assert!(line.chars().count() <= 80, "{line:?} is {} cells", line.chars().count());
            // Plain text only: no escape codes, no box drawing.
            assert!(line.is_ascii(), "{line:?}");
        }
    }

    #[test]
    fn the_labels_line_up_so_the_values_read_as_a_column() {
        // The property that makes the report scannable: every value begins in
        // the same column, whatever the label lengths are.
        let sections = sections_of(&sample(3, 12000, 9000, 2331), Some((150_000, 200_000)));
        let width = sections
            .iter()
            .flat_map(|(_, rows)| rows.iter())
            .map(|r| r.label.chars().count())
            .max()
            .unwrap();
        let out = render(&sections, &|s| s.to_string());
        let value_col = 2 + width + 2;
        let rows: Vec<&str> = out.lines().filter(|l| l.starts_with("  ")).collect();
        assert_eq!(rows.len(), 7, "{out}");
        for line in rows {
            let rest = &line[value_col..];
            assert!(!rest.starts_with(' '), "ragged value column in {line:?}");
            // The label itself is what was padded, so it never reaches the
            // value column.
            assert!(line[2..value_col].trim().len() < value_col - 2, "{line:?}");
        }
    }

    #[test]
    fn the_context_line_states_the_window_and_the_share_used() {
        // The window and the share used: both must be readable at a glance.
        let sections = sections_of(&sample(3, 12000, 9000, 2331), Some((150_000, 200_000)));
        let out = render(&sections, &|s| s.to_string());
        assert!(out.contains("150,000 of ~200,000 (75%)"), "{out}");
        // Without a measurement the line says so rather than inventing a pair.
        let none = render(&sections_of(&sample(1, 10, 0, 0), None), &|s| s.to_string());
        assert!(none.contains("context    n/a"), "{none}");
        let empty = render(&sections_of(&View { cores: 8, ..View::default() }, None), &|s| s.to_string());
        assert!(empty.contains("context    none yet"), "{empty}");
    }

    #[test]
    fn a_stream_too_quick_to_measure_reports_no_speed() {
        // 30 tokens in 0.4ms is not 75000 tok/s, it is an unmeasurable number.
        assert_eq!(measure_speed(30, 0.0004, 1), None);
        assert_eq!(measure_speed(0, 1.0, 1), Some(0.0));
        assert_eq!(measure_speed(100, 0.0, 1), None);
        assert_eq!(measure_speed(100, 1.0, 0), None, "no request, no speed");
        // 100 tokens over 2s of generating time is 50 tok/s.
        assert_eq!(measure_speed(100, 2.0, 1), Some(50.0));
        // The threshold is per request, not on the session total.
        assert_eq!(measure_speed(200, 0.09, 2), None);
        assert_eq!(measure_speed(200, 0.11, 2), Some(200.0 / 0.11));
    }

    #[test]
    fn num_groups_thousands() {
        assert_eq!(num(0), "0");
        assert_eq!(num(999), "999");
        assert_eq!(num(1000), "1,000");
        assert_eq!(num(1234567), "1,234,567");
    }

    #[test]
    fn the_process_metrics_are_readable_on_this_machine() {
        assert!(cpu_ticks().is_some(), "cpu time must be readable");
        assert!(rss_kb().is_some_and(|k| k > 0), "resident memory must be readable");
        assert!(clock_ticks_per_sec() > 0.0);
    }

    #[test]
    fn the_sampler_fills_cpu_and_memory_over_time() {
        // The figures only exist if the background thread actually runs, so
        // this is the one assertion that a real sample lands rather than a
        // parsed /proc file being correct in isolation.
        spawn_sampler();
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while std::time::Instant::now() < deadline {
            let v = view();
            if v.cpu_percent.is_some() && v.mem_mib.is_some() {
                assert!(v.mem_mib.unwrap() > 0.0);
                assert!(v.cpu_percent.unwrap() >= 0.0);
                return;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        panic!("the sampler never produced a sample");
    }

    #[test]
    fn recording_accumulates_and_the_mean_is_per_request() {
        let before = view();
        record(Usage { input: 10, cache_read: 0, cache_write: 0, output: 4 },
               Duration::from_secs(1), Duration::from_secs(2));
        record(Usage { input: 0, cache_read: 90, cache_write: 0, output: 6 },
               Duration::from_secs(3), Duration::from_secs(2));
        let after = view();
        assert_eq!(after.requests, before.requests + 2);
        assert_eq!(after.usage.input, before.usage.input + 10);
        assert_eq!(after.usage.cache_read, before.usage.cache_read + 90);
        assert_eq!(after.usage.output, before.usage.output + 10);
        // 10 output tokens over 4 seconds of generating time.
        assert!((after.avg_speed.unwrap() - 2.5).abs() < 1e-9, "{:?}", after.avg_speed);
    }
}
