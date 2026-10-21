//! Windows terminal layer: console modes for raw input and VT output, window
//! size, resize (polled — there is no SIGWINCH) and reads from the console
//! handle. The console is driven with `ENABLE_VIRTUAL_TERMINAL_INPUT`, so
//! stdin delivers the same VT byte stream the shared parser already reads on
//! Unix; the editor and renderer above need no Windows branch.
//!
//! Windows 10 1809+ (ConPTY / VT console) is assumed.

use std::os::windows::io::AsRawHandle;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use windows_sys::Win32::Foundation::{HANDLE, WAIT_OBJECT_0};
use windows_sys::Win32::Storage::FileSystem::ReadFile;
use windows_sys::Win32::System::Console::{
    GetConsoleMode, GetConsoleScreenBufferInfo, GetStdHandle, SetConsoleMode, SetConsoleOutputCP,
    CONSOLE_MODE, CONSOLE_SCREEN_BUFFER_INFO, ENABLE_ECHO_INPUT, ENABLE_EXTENDED_FLAGS,
    ENABLE_LINE_INPUT, ENABLE_MOUSE_INPUT, ENABLE_PROCESSED_INPUT, ENABLE_PROCESSED_OUTPUT,
    ENABLE_QUICK_EDIT_MODE, ENABLE_VIRTUAL_TERMINAL_INPUT, ENABLE_VIRTUAL_TERMINAL_PROCESSING,
    ENABLE_WINDOW_INPUT, STD_ERROR_HANDLE, STD_HANDLE, STD_INPUT_HANDLE, STD_OUTPUT_HANDLE,
};
use windows_sys::Win32::System::IO::CancelSynchronousIo;
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
            // Mouse and window events are console input records, not keys.
            // With VT input enabled the console turns them into escape
            // sequences (`ESC[<35;25;24M` for a mouse move), which the line
            // editor would otherwise have to parse and discard. Leaving the
            // flags on also means every mouse move wakes the watcher and
            // repaints the prompt row. Resize is polled from the screen
            // buffer, so no window events are needed either.
            let raw = (mode
                & !(ENABLE_ECHO_INPUT
                    | ENABLE_LINE_INPUT
                    | ENABLE_PROCESSED_INPUT
                    | ENABLE_QUICK_EDIT_MODE
                    | ENABLE_MOUSE_INPUT
                    | ENABLE_WINDOW_INPUT))
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

/// A CR that ended the previous read, waiting to see whether the next byte is
/// the LF of the same Enter. Only one input reader is active at a time.
static PENDING_CR: AtomicBool = AtomicBool::new(false);

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
    let mut pending = PENDING_CR.load(Ordering::Relaxed);
    let n = super::collapse_crlf(&mut pending, &mut buf[..n as usize]);
    PENDING_CR.store(pending, Ordering::Relaxed);
    Ok(n)
}

/// Cancel the console read the input watcher may still have in flight. A turn
/// can end while `ReadFile` is waiting for a key that produces no bytes; the
/// next prompt must not wait for the user to press something first.
///
/// Retried briefly: the watcher can see the finished flag just before it
/// starts the read, so the first cancel has nothing to cancel yet.
pub fn unblock_stdin(handle: &std::thread::JoinHandle<()>) {
    for _ in 0..3 {
        unsafe {
            CancelSynchronousIo(handle.as_raw_handle());
        }
        if handle.is_finished() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
}
