//! Unix terminal layer: termios raw mode, SIGWINCH resize, poll/read on fd 0.

/// Raw mode for the line editor and the mid-turn input watcher.
pub struct RawGuard {
    _private: (),
}

impl RawGuard {
    pub fn new() -> Option<Self> {
        if !super::is_tty() {
            return Some(RawGuard { _private: () });
        }
        unsafe {
            libc::tcgetattr(0, &raw mut RAW_OLD as *mut _);
            let mut new = RAW_OLD;
            new.c_lflag &= !(libc::ICANON | libc::ECHO | libc::ISIG | libc::IEXTEN);
            new.c_iflag &= !(libc::BRKINT | libc::ICRNL | libc::INPCK | libc::ISTRIP | libc::IXON);
            new.c_cc[libc::VMIN] = 1;
            new.c_cc[libc::VTIME] = 0;
            libc::tcsetattr(0, libc::TCSAFLUSH, &new);
        }
        Some(RawGuard { _private: () })
    }
}

impl Drop for RawGuard {
    fn drop(&mut self) {
        if super::is_tty() {
            unsafe {
                libc::tcsetattr(0, libc::TCSAFLUSH, &raw const RAW_OLD);
            }
        }
    }
}

static mut RAW_OLD: libc::termios = unsafe { std::mem::zeroed() };

/// Resize is process-wide. The handler only records that it happened; the
/// reader loop turns the flag into a repaint. A handler may not allocate.
static RESIZED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

extern "C" fn on_resize(_: libc::c_int) {
    RESIZED.store(true, std::sync::atomic::Ordering::Relaxed);
}

/// Subscribe to window-size changes. Call once, before the first frame.
pub fn watch_resize() {
    unsafe {
        libc::signal(
            libc::SIGWINCH,
            on_resize as extern "C" fn(libc::c_int) as libc::sighandler_t,
        );
    }
}

pub fn took_resize() -> bool {
    RESIZED.swap(false, std::sync::atomic::Ordering::Relaxed)
}

/// Window size in cells; `None` when the terminal does not answer.
pub fn size() -> Option<(usize, usize)> {
    let mut ws: libc::winsize = unsafe { std::mem::zeroed() };
    if unsafe { libc::ioctl(1, libc::TIOCGWINSZ, &mut ws) } != 0 || ws.ws_col == 0 {
        return None;
    }
    Some((ws.ws_col as usize, ws.ws_row as usize))
}

/// Input available on stdin within `ms`, without consuming it.
pub fn poll_in(ms: i32) -> bool {
    let mut pfd: libc::pollfd = libc::pollfd { fd: 0, events: libc::POLLIN, revents: 0 };
    unsafe { libc::poll(&mut pfd, 1, ms) > 0 && (pfd.revents & libc::POLLIN) != 0 }
}

/// Read up to `buf.len()` bytes of input. `Interrupted` is handed back to the
/// caller, so a resize that landed on a blocked read stays visible.
pub fn read_stdin(buf: &mut [u8]) -> std::io::Result<usize> {
    let n = unsafe { libc::read(0, buf.as_mut_ptr() as *mut _, buf.len()) };
    if n < 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(n as usize)
    }
}

/// No-op: every watcher read is preceded by `poll_in`, so a finished turn
/// never leaves a read blocked on this platform.
pub fn unblock_stdin(_: &std::thread::JoinHandle<()>) {}

/// Unix terminals need no setup: the ANSI the renderer writes is native there.
/// Present so `main` can call it on every platform.
pub fn platform_init() {}
