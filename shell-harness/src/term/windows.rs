//! Windows terminal layer: console modes for raw input and VT output, window
//! size, resize (polled — there is no SIGWINCH) and reads from the console
//! handle. The console is driven with `ENABLE_VIRTUAL_TERMINAL_INPUT`, so
//! stdin delivers the same VT byte stream the shared parser already reads on
//! Unix; the editor and renderer above need no Windows branch.
//!
//! Windows 10 1809+ (ConPTY / VT console) is assumed.

use std::sync::atomic::{AtomicU64, Ordering};

use windows_sys::Win32::Foundation::{HANDLE, WAIT_OBJECT_0};
use windows_sys::Win32::Storage::FileSystem::ReadFile;
use windows_sys::Win32::System::Console::{
    GetConsoleMode, GetConsoleScreenBufferInfo, GetStdHandle, SetConsoleMode, SetConsoleOutputCP,
    CONSOLE_MODE, CONSOLE_SCREEN_BUFFER_INFO, ENABLE_ECHO_INPUT, ENABLE_EXTENDED_FLAGS,
    ENABLE_LINE_INPUT, ENABLE_PROCESSED_INPUT, ENABLE_PROCESSED_OUTPUT, ENABLE_QUICK_EDIT_MODE,
    ENABLE_VIRTUAL_TERMINAL_INPUT, ENABLE_VIRTUAL_TERMINAL_PROCESSING, STD_ERROR_HANDLE,
    STD_HANDLE, STD_INPUT_HANDLE, STD_OUTPUT_HANDLE,
};
use windows_sys::Win32::System::Threading::WaitForSingleObject;

/// The three standard handles; the console API answers with null when the
/// stream is redirected, which every function below treats as "not a console".
fn handle(which: STD_HANDLE) -> HANDLE {
    unsafe { GetStdHandle(which) }
}

/// Windows 10+ console setup: UTF-8 output (the prompt glyphs) and VT parsing
/// on stdout and stderr, so the shared ANSI renderer draws instead of printing
/// literal escapes. Redirection makes the calls fail and is not an error.
pub fn platform_init() {
    unsafe {
        SetConsoleOutputCP(65001);
        for h in [handle(STD_OUTPUT_HANDLE), handle(STD_ERROR_HANDLE)] {
            let mut mode: CONSOLE_MODE = 0;
            if GetConsoleMode(h, &mut mode) != 0 {
                SetConsoleMode(
                    h,
                    mode | ENABLE_PROCESSED_OUTPUT | ENABLE_VIRTUAL_TERMINAL_PROCESSING,
                );
            }
        }
    }
}

/// Raw mode: keys arrive one at a time as VT byte sequences, Enter as a bare
/// CR and Ctrl-C as 0x03 (processed input off, matching ISIG-off on Unix).
pub struct RawGuard {
    old: Option<(HANDLE, CONSOLE_MODE)>,
}

impl RawGuard {
    pub fn new() -> Option<Self> {
        if !super::is_tty() {
            return Some(RawGuard { old: None });
        }
        unsafe {
            let h = handle(STD_INPUT_HANDLE);
            let mut mode: CONSOLE_MODE = 0;
            if GetConsoleMode(h, &mut mode) == 0 {
                return Some(RawGuard { old: None });
            }
            let raw = (mode
                & !(ENABLE_ECHO_INPUT
                    | ENABLE_LINE_INPUT
                    | ENABLE_PROCESSED_INPUT
                    | ENABLE_QUICK_EDIT_MODE))
                | ENABLE_EXTENDED_FLAGS
                | ENABLE_VIRTUAL_TERMINAL_INPUT;
            SetConsoleMode(h, raw);
            Some(RawGuard { old: Some((h, mode)) })
        }
    }
}

impl Drop for RawGuard {
    fn drop(&mut self) {
        if let Some((h, old)) = self.old {
            unsafe {
                SetConsoleMode(h, old);
            }
        }
    }
}

/// Echo off while a secret (API key) is typed. `None` when stdin is not a
/// console, so the caller knows not to expect a swallowed newline.
pub struct EchoGuard {
    old: (HANDLE, CONSOLE_MODE),
}

impl EchoGuard {
    pub fn new() -> Option<Self> {
        if !super::is_tty() {
            return None;
        }
        unsafe {
            let h = handle(STD_INPUT_HANDLE);
            let mut mode: CONSOLE_MODE = 0;
            if GetConsoleMode(h, &mut mode) == 0 {
                return None;
            }
            SetConsoleMode(h, mode & !ENABLE_ECHO_INPUT);
            Some(EchoGuard { old: (h, mode) })
        }
    }
}

impl Drop for EchoGuard {
    fn drop(&mut self) {
        let (h, old) = self.old;
        unsafe {
            SetConsoleMode(h, old);
        }
    }
}

/// The last window size `took_resize` saw. No signal exists to tell us the
/// window changed, so the size itself is the signal.
static LAST_SIZE: AtomicU64 = AtomicU64::new(u64::MAX);

/// There is nothing to subscribe to on Windows; resize is detected by
/// `took_resize` polling the window size.
pub fn watch_resize() {}

pub fn took_resize() -> bool {
    let Some((cols, rows)) = size() else {
        return false;
    };
    let now = ((cols as u64) << 32) | rows as u64;
    let prev = LAST_SIZE.swap(now, Ordering::Relaxed);
    prev != u64::MAX && prev != now
}

/// Visible window size in cells, from the console screen buffer.
pub fn size() -> Option<(usize, usize)> {
    let mut info: CONSOLE_SCREEN_BUFFER_INFO = unsafe { std::mem::zeroed() };
    if unsafe { GetConsoleScreenBufferInfo(handle(STD_OUTPUT_HANDLE), &mut info) } == 0 {
        return None;
    }
    let cols = i32::from(info.srWindow.Right) - i32::from(info.srWindow.Left) + 1;
    let rows = i32::from(info.srWindow.Bottom) - i32::from(info.srWindow.Top) + 1;
    if cols <= 0 || rows <= 0 {
        return None;
    }
    Some((cols as usize, rows as usize))
}

/// Input available on stdin within `ms`, without consuming it. A console
/// handle is waitable; `WSAPoll` does not cover it.
pub fn poll_in(ms: i32) -> bool {
    let h = handle(STD_INPUT_HANDLE);
    unsafe { WaitForSingleObject(h, ms.max(0) as u32) == WAIT_OBJECT_0 }
}

/// Read up to `buf.len()` bytes from the console. The handle is read directly
/// rather than through `std::io::stdin`, whose buffering would desync the
/// availability checks above.
pub fn read_stdin(buf: &mut [u8]) -> std::io::Result<usize> {
    let h = handle(STD_INPUT_HANDLE);
    let mut n: u32 = 0;
    let ok = unsafe {
        ReadFile(
            h,
            buf.as_mut_ptr(),
            buf.len() as u32,
            &mut n,
            std::ptr::null_mut(),
        )
    };
    if ok == 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(super::collapse_crlf(&mut buf[..n as usize]))
}
