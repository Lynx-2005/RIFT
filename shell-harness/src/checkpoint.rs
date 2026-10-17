//! Snapshots of the working directory, one per turn, so `/rewind` can undo the
//! work and not only the conversation.
//!
//! A snapshot is a plain copy of the tree with build output and VCS internals
//! left out. Restoring it puts changed files back and deletes the files that
//! appeared since, because "undo that turn" has to include the file the turn
//! created. Files too large to copy are named in a manifest and left alone on
//! restore: deleting a user's asset to satisfy a snapshot that never held it
//! would be worse than not undoing it.

use std::collections::HashSet;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// Never worth copying: build output, caches, VCS internals.
const SKIP_DIRS: &[&str] = &[
    ".git",
    "target",
    "node_modules",
    "dist",
    "build",
    "out",
    "__pycache__",
    ".venv",
    "venv",
    ".cache",
    ".next",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".gradle",
    ".idea",
    ".terraform",
];

/// Past this size it is an asset or build output, not source.
const MAX_FILE: u64 = 2 * 1024 * 1024;

/// Budget for one snapshot. A turn needs to undo an edit, not copy a disk: a
/// working directory that was a projects root held 68,000 files and 20 GB, and
/// copying that before every turn made a typed prompt look like a hang. When a
/// budget trips the snapshot stops early, says it is partial, and a restore
/// from it never deletes anything.
const MAX_FILES: usize = 8_000;
const MAX_BYTES: u64 = 64 * 1024 * 1024;
const MAX_SECS: f64 = 2.0;

/// How many turn snapshots one session keeps; older ones are dropped.
pub const KEEP: usize = 10;

/// Records the paths a snapshot deliberately did not copy.
const MANIFEST: &str = ".rift-snapshot-skipped";

/// Every companion file a snapshot writes for itself shares this prefix. They
/// are the snapshot's own bookkeeping, never part of the project, and a walk
/// that mistook one for a captured file would have a rewind put it back into
/// the user's working directory.
const MARKER_PREFIX: &str = ".rift-snapshot-";

/// Present when a snapshot stopped early. Nothing is deleted on a restore from
/// one: it cannot know what came after the point where it gave up.
const PARTIAL: &str = ".rift-snapshot-partial";

/// What the turn actually changed, written when the turn ends: one line per
/// path, `A|M|D <hash> <path>`. Rewind follows this and nothing else, so it
/// undoes the session's own work instead of every difference in the tree.
///
/// Nothing was recording the turn's own edits because a shell tool does not
/// report which files it touched, and pi, which was the model to copy, has no
/// file undo at all. Comparing at both ends is how a bash-only agent learns it.
const TOUCHED: &str = ".rift-snapshot-touched";

#[derive(Debug, Default, PartialEq)]
pub struct Snap {
    pub files: usize,
    pub skipped: usize,
    /// The tree did not fit in the budget, so the copy stops where it is.
    pub partial: bool,
}

#[derive(Debug, Default, PartialEq)]
pub struct Restore {
    pub restored: usize,
    pub removed: usize,
    pub left_alone: usize,
    /// The snapshot was partial, so files that appeared since were kept.
    pub partial: bool,
    /// Paths the turn changed that something else has changed since, left as
    /// they are rather than overwritten.
    pub changed_since: usize,
    /// No record of what the turn touched: only the conversation was rewound.
    pub unattributed: bool,
}

/// FNV-1a. Fingerprints are compared, never relied on as proof of identity, so
/// a cheap hash with no dependency is the right size of tool.
fn fingerprint(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn file_fingerprint(path: &Path) -> Option<u64> {
    fs::read(path).ok().map(|b| fingerprint(&b))
}

/// The turn's own edits, or none if this session never recorded them.
fn touched(src: &Path) -> Vec<(char, u64, String)> {
    let text = match fs::read_to_string(src.join(TOUCHED)) {
        Ok(t) => t,
        Err(_) => return Vec::new(),
    };
    text.lines()
        .filter_map(|line| {
            let mut parts = line.splitn(3, ' ');
            let kind = parts.next()?.chars().next()?;
            let hash = parts.next()?.parse::<u64>().ok()?;
            let path = parts.next()?.to_string();
            Some((kind, hash, path))
        })
        .collect()
}

/// Record what this turn changed, by comparing the tree against the snapshot
/// taken before it ran. Called when the turn ends, including a cancelled one:
/// a half-finished turn is exactly when someone wants the undo.
pub fn mark(session: i64, turn: usize, work: &Path) -> io::Result<usize> {
    let src = dir_for(session, turn);
    if !src.is_dir() {
        return Ok(0);
    }
    let captured = listing(&src)?;
    let partial = src.join(PARTIAL).exists();
    // Files the snapshot deliberately passed over, for size. They existed
    // before the turn, so they must never be mistaken for the turn's own
    // creations and deleted, and their contents were never kept, so they
    // cannot be put back either.
    let uncaptured: HashSet<PathBuf> = fs::read_to_string(src.join(MANIFEST))
        .map(|t| t.lines().filter(|l| !l.trim().is_empty()).map(PathBuf::from).collect())
        .unwrap_or_default();
    let spend = Spend::new();
    let mut lines: Vec<String> = Vec::new();
    for rel in &captured {
        if spend.over() {
            // Out of room to look: the rest stays unrecorded, and a rewind
            // simply leaves unrecorded paths alone.
            return write_touched(&src, &lines);
        }
        let live = work.join(rel);
        match file_fingerprint(&live) {
            None => lines.push(format!("D 0 {}", rel.display())),
            Some(h) => {
                let before = file_fingerprint(&src.join(rel)).unwrap_or(0);
                if h != before {
                    lines.push(format!("M {h} {}", rel.display()));
                }
            }
        }
    }
    if !partial {
        // Files the turn created. Only a snapshot that saw the whole tree can
        // know what was not there before, and only then may a rewind delete.
        let exclude = excluded_dirs();
        let cap = budget().0;
        let mut made: Vec<PathBuf> = Vec::new();
        walk(work, work, &exclude, &mut |rel, _| {
            if made.len() < cap && !captured.contains(rel) && !uncaptured.contains(rel) {
                made.push(rel.to_path_buf());
            }
            Ok(())
        })?;
        for rel in made {
            if let Some(h) = file_fingerprint(&work.join(&rel)) {
                lines.push(format!("A {h} {}", rel.display()));
            }
        }
    }
    write_touched(&src, &lines)
}

fn write_touched(src: &Path, lines: &[String]) -> io::Result<usize> {
    fs::write(src.join(TOUCHED), lines.join("\n"))?;
    Ok(lines.len())
}

/// Files and bytes one snapshot may spend, overridable so tests can trip the
/// budget without writing eight thousand files.
fn budget() -> (usize, u64, f64) {
    #[cfg(test)]
    if let Some(b) = TEST_BUDGET.with(|b| *b.borrow()) {
        return b;
    }
    (MAX_FILES, MAX_BYTES, MAX_SECS)
}

#[cfg(test)]
thread_local! {
    static TEST_BUDGET: std::cell::RefCell<Option<(usize, u64, f64)>> =
        const { std::cell::RefCell::new(None) };
}

/// What a copy has spent so far, and whether it has run out.
struct Spend {
    files: usize,
    bytes: u64,
    started: std::time::Instant,
    files_cap: usize,
    bytes_cap: u64,
    secs_cap: f64,
}

impl Spend {
    fn new() -> Self {
        let (files_cap, bytes_cap, secs_cap) = budget();
        Self {
            files: 0,
            bytes: 0,
            started: std::time::Instant::now(),
            files_cap,
            bytes_cap,
            secs_cap,
        }
    }

    fn over(&self) -> bool {
        self.files >= self.files_cap
            || self.bytes >= self.bytes_cap
            || self.started.elapsed().as_secs_f64() >= self.secs_cap
    }

    fn charge(&mut self, bytes: u64) {
        self.files += 1;
        self.bytes += bytes;
    }
}

/// Root of all snapshots: inside the data dir, never inside the tree we copy.
pub fn root() -> PathBuf {
    #[cfg(test)]
    if let Some(dir) = TEST_ROOT.with(|r| r.borrow().clone()) {
        return dir;
    }
    crate::store::data_dir().join("snapshots")
}

// Tests point the snapshots at their own scratch dir. Thread-local, so
// parallel tests cannot redirect each other, and absent from real builds.
#[cfg(test)]
thread_local! {
    static TEST_ROOT: std::cell::RefCell<Option<PathBuf>> =
        const { std::cell::RefCell::new(None) };
}

/// Where one turn's snapshot lives.
fn dir_for(session: i64, turn: usize) -> PathBuf {
    root().join(format!("s{session}")).join(format!("turn-{turn}"))
}

/// Snapshot the tree before a turn starts. Returns what it copied, and whether
/// it ran out of budget before finishing.
pub fn take(session: i64, turn: usize, work: &Path) -> io::Result<Snap> {
    let dest = dir_for(session, turn);
    let exclude = excluded_dirs();
    let _ = fs::remove_dir_all(&dest);
    let mut snap = Snap::default();
    let mut skipped: Vec<String> = Vec::new();
    let mut spend = Spend::new();
    copy_tree(work, &dest, work, &exclude, &mut snap, &mut skipped, &mut spend)?;
    if snap.partial {
        fs::write(dest.join(PARTIAL), "stopped early\n")?;
    }
    if !skipped.is_empty() {
        fs::write(dest.join(MANIFEST), skipped.join("\n"))?;
    }
    prune(session);
    Ok(snap)
}

/// Directories a snapshot must never enter: the snapshot root, and the data dir
/// it lives in.
///
/// Learned the hard way. With the working directory holding the data dir, the
/// first snapshot copied itself, then that copy, and so on until the path ran
/// out of room. The guard used to compare canonical paths, which cannot work
/// before the root exists. Identity is the reliable question: same device, same
/// inode, same directory, whatever the path says or hides behind a symlink.
fn excluded_dirs() -> Vec<(u64, u64)> {
    let root = root();
    let mut out = Vec::new();
    for dir in [root.as_path(), root.parent().unwrap_or(root.as_path())] {
        let _ = fs::create_dir_all(dir);
        if let Some(id) = dir_id(dir) {
            out.push(id);
        }
    }
    out
}

#[cfg(unix)]
fn dir_id(dir: &Path) -> Option<(u64, u64)> {
    use std::os::unix::fs::MetadataExt;
    fs::metadata(dir).ok().map(|m| (m.dev(), m.ino()))
}

/// Windows identity: volume serial number plus file index — what the file
/// system uses in place of (device, inode). `MetadataExt` cannot answer this on
/// stable Rust, so the directory is opened for its handle information.
#[cfg(windows)]
fn dir_id(dir: &Path) -> Option<(u64, u64)> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::Storage::FileSystem::{
        CreateFileW, GetFileInformationByHandle, BY_HANDLE_FILE_INFORMATION,
        FILE_FLAG_BACKUP_SEMANTICS, FILE_SHARE_DELETE, FILE_SHARE_READ, FILE_SHARE_WRITE,
        OPEN_EXISTING,
    };

    let mut wide: Vec<u16> = dir.as_os_str().encode_wide().collect();
    wide.push(0);
    unsafe {
        // No access rights, share everything: this only asks who the directory
        // is, and it must not block another reader while a snapshot runs.
        let h = CreateFileW(
            wide.as_ptr(),
            0,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
            std::ptr::null(),
            OPEN_EXISTING,
            FILE_FLAG_BACKUP_SEMANTICS,
            std::ptr::null_mut(),
        );
        if h == INVALID_HANDLE_VALUE {
            return None;
        }
        let mut info: BY_HANDLE_FILE_INFORMATION = std::mem::zeroed();
        let ok = GetFileInformationByHandle(h, &mut info);
        CloseHandle(h);
        if ok == 0 {
            return None;
        }
        let index = (info.nFileIndexHigh as u64) << 32 | info.nFileIndexLow as u64;
        Some((info.dwVolumeSerialNumber as u64, index))
    }
}

/// Is this directory one the walk must not enter?
fn is_excluded(dir: &Path, exclude: &[(u64, u64)]) -> bool {
    dir_id(dir).is_some_and(|id| exclude.contains(&id))
}

/// Put the tree back the way it was before `turn`. `None` when this session
/// has no snapshot for that turn, which is different from a failed restore.
pub fn put_back(session: i64, turn: usize, work: &Path) -> Option<io::Result<Restore>> {
    let src = dir_for(session, turn);
    if !src.is_dir() {
        return None;
    }
    Some(restore(work, &src))
}

/// Forget the snapshots for turns after `turn`: after a rewind they describe a
/// tree that no longer exists.
pub fn drop_after(session: i64, turn: usize) {
    for n in turns(session) {
        if n > turn {
            let _ = fs::remove_dir_all(dir_for(session, n));
        }
    }
}

/// Keep only the newest `KEEP` snapshots for a session.
fn prune(session: i64) {
    let mut have = turns(session);
    while have.len() > KEEP {
        let oldest = have.remove(0);
        let _ = fs::remove_dir_all(dir_for(session, oldest));
    }
}

/// Turn numbers this session has snapshots for, ascending.
fn turns(session: i64) -> Vec<usize> {
    let dir = root().join(format!("s{session}"));
    let mut out: Vec<usize> = fs::read_dir(&dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .filter_map(|e| {
                    e.file_name()
                        .to_string_lossy()
                        .strip_prefix("turn-")
                        .and_then(|n| n.parse::<usize>().ok())
                })
                .collect()
        })
        .unwrap_or_default();
    out.sort_unstable();
    out
}

/// Relative paths of every file in the snapshot.
fn listing(src: &Path) -> io::Result<HashSet<PathBuf>> {
    let mut files = HashSet::new();
    walk(src, src, &[], &mut |rel, _| {
        files.insert(rel.to_path_buf());
        Ok(())
    })?;
    Ok(files)
}

/// Put back everything the turns from `first` to `last` changed, unwinding the
/// newest first. Newest first matters: if a later turn edited a file an earlier
/// one had already touched, its record compares against the state the later
/// turn left, so undoing in order keeps every check meaningful. `None` when
/// none of those turns has a snapshot at all.
pub fn put_back_all(session: i64, first: usize, last: usize, work: &Path) -> Option<io::Result<Restore>> {
    let mut total: Option<Restore> = None;
    for turn in (first..=last).rev() {
        match put_back(session, turn, work) {
            Some(Ok(r)) => {
                let t = total.get_or_insert_with(Restore::default);
                t.restored += r.restored;
                t.removed += r.removed;
                t.changed_since += r.changed_since;
                t.partial |= r.partial;
                t.unattributed |= r.unattributed;
            }
            Some(Err(e)) => return Some(Err(e)),
            None => {}
        }
    }
    total.map(Ok)
}

/// Put back what this turn changed, and nothing else. A path the turn touched
/// but something else has changed since is left as it is: this undoes one
/// session's work, not the world's.
fn restore(work: &Path, src: &Path) -> io::Result<Restore> {
    let partial = src.join(PARTIAL).exists();
    let edits = touched(src);
    if edits.is_empty() {
        // Older snapshots, or a turn that never got to record its work. Doing
        // nothing beats guessing: every difference in the tree is not ours.
        return Ok(Restore { partial, unattributed: true, ..Restore::default() });
    }
    let mut r = Restore { partial, ..Restore::default() };
    for (kind, post, rel) in &edits {
        let rel = PathBuf::from(rel);
        let live = work.join(&rel);
        let now = file_fingerprint(&live);
        let unchanged_since = now == Some(*post);
        match kind {
            'A' => {
                if !unchanged_since {
                    r.changed_since += 1;
                    continue;
                }
                if fs::remove_file(&live).is_ok() {
                    r.removed += 1;
                }
            }
            // A file the turn deleted is untouched since when it is still gone:
            // one that came back was put there by something else.
            'D' => {
                if now.is_some() {
                    r.changed_since += 1;
                    continue;
                }
                if let Some(parent) = live.parent() {
                    fs::create_dir_all(parent)?;
                }
                if fs::copy(src.join(&rel), &live).is_ok() {
                    r.restored += 1;
                }
            }
            'M' => {
                if !unchanged_since {
                    r.changed_since += 1;
                    continue;
                }
                if let Some(parent) = live.parent() {
                    fs::create_dir_all(parent)?;
                }
                if fs::copy(src.join(&rel), &live).is_ok() {
                    r.restored += 1;
                }
            }
            _ => {}
        }
    }
    Ok(r)
}

/// Copy a tree, skipping the noise and never descending into an excluded
/// directory: those are the snapshot's own storage, and a snapshot that walks
/// into its storage copies itself.
fn copy_tree(
    from: &Path,
    to: &Path,
    root: &Path,
    exclude: &[(u64, u64)],
    snap: &mut Snap,
    skipped: &mut Vec<String>,
    spend: &mut Spend,
) -> io::Result<()> {
    if spend.over() {
        snap.partial = true;
        return Ok(());
    }
    let entries = match fs::read_dir(from) {
        Ok(e) => e,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e),
    };
    fs::create_dir_all(to)?;
    for entry in entries {
        if spend.over() {
            snap.partial = true;
            return Ok(());
        }
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_string();
        let src = entry.path();
        let dst = to.join(&name);
        let meta = entry.metadata()?;
        if meta.is_dir() {
            if SKIP_DIRS.contains(&name.as_str()) || is_excluded(&src, exclude) {
                continue;
            }
            copy_tree(&src, &dst, root, exclude, snap, skipped, spend)?;
        } else if meta.file_type().is_symlink() {
            link(&src, &dst);
        } else if meta.is_file() {
            if meta.len() > MAX_FILE {
                snap.skipped += 1;
                if let Ok(rel) = src.strip_prefix(root) {
                    skipped.push(rel.to_string_lossy().to_string());
                }
            } else {
                fs::copy(&src, &dst)?;
                spend.charge(meta.len());
                snap.files += 1;
            }
        }
    }
    Ok(())
}

#[cfg(unix)]
fn link(from: &Path, to: &Path) {
    if let Ok(target) = fs::read_link(from) {
        let _ = std::os::unix::fs::symlink(target, to);
    }
}

/// Best effort: creating a symlink on Windows needs developer mode or the
/// symlink privilege, and a failure only drops the link from the snapshot.
#[cfg(windows)]
fn link(from: &Path, to: &Path) {
    if let Ok(target) = fs::read_link(from) {
        let is_dir = fs::metadata(from).map(|m| m.is_dir()).unwrap_or(false);
        let r = if is_dir {
            std::os::windows::fs::symlink_dir(target, to)
        } else {
            std::os::windows::fs::symlink_file(target, to)
        };
        let _ = r;
    }
}

/// Visit every file under `at`, giving paths relative to `root`.
fn walk<F>(root: &Path, at: &Path, exclude: &[(u64, u64)], f: &mut F) -> io::Result<()>
where
    F: FnMut(&Path, &Path) -> io::Result<()>,
{
    let entries = match fs::read_dir(at) {
        Ok(e) => e,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e),
    };
    for entry in entries {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_string();
        let path = entry.path();
        let meta = entry.metadata()?;
        if meta.is_dir() {
            if SKIP_DIRS.contains(&name.as_str()) || is_excluded(&path, exclude) {
                continue;
            }
            walk(root, &path, exclude, f)?;
        } else if meta.is_file() && !name.starts_with(MARKER_PREFIX) {
            if let Ok(rel) = path.strip_prefix(root) {
                f(rel, &path)?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A private scratch tree per test, wiped on the way in. Snapshots go to a
    /// sibling directory, so no test can reach the real data dir.
    fn scratch(name: &str) -> PathBuf {
        let base = std::env::temp_dir().join(format!("rift-snap-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        let work = base.join("work");
        fs::create_dir_all(&work).unwrap();
        TEST_ROOT.with(|r| *r.borrow_mut() = Some(base.join("snapshots")));
        TEST_BUDGET.with(|b| *b.borrow_mut() = None);
        work
    }

    /// Both halves of the scratch tree go away together.
    fn done(work: &Path) {
        if let Some(base) = work.parent() {
            let _ = fs::remove_dir_all(base);
        }
    }

    fn put(dir: &Path, rel: &str, body: &str) {
        let p = dir.join(rel);
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(p, body).unwrap();
    }

    fn get(dir: &Path, rel: &str) -> String {
        fs::read_to_string(dir.join(rel)).unwrap_or_default()
    }

    #[test]
    fn a_rewind_puts_changed_files_back_and_removes_new_ones() {
        let work = scratch("undo-work");
        put(&work, "src/main.rs", "one\n");
        put(&work, "notes.md", "keep me\n");
        put(&work, "src/deep/mod.rs", "old\n");

        let snap = take(1, 1, &work).unwrap();
        assert_eq!(snap.files, 3);

        // The turn: edits a file, creates one, deletes one, makes a directory.
        put(&work, "src/main.rs", "two\n");
        put(&work, "src/new.rs", "new\n");
        fs::remove_file(work.join("notes.md")).unwrap();
        mark(1, 1, &work).unwrap();

        let r = put_back(1, 1, &work).unwrap().unwrap();
        assert_eq!(get(&work, "src/main.rs"), "one\n", "edit not undone");
        assert_eq!(get(&work, "notes.md"), "keep me\n", "delete not undone");
        assert!(!work.join("src/new.rs").exists(), "created file left behind");
        assert_eq!(r.restored, 2);
        assert_eq!(r.removed, 1);
        assert_eq!(r.changed_since, 0);
        done(&work);
    }

    #[test]
    fn a_file_something_else_changed_since_the_turn_is_left_alone() {
        // The point of attributing: rewind undoes this session's turn, not
        // every difference in the tree. An editor, a build, another agent, or
        // a second rift session may have written here afterwards.
        let work = scratch("changed-since");
        put(&work, "src/mine.rs", "before\n");
        put(&work, "src/theirs.rs", "before\n");
        take(21, 1, &work).unwrap();

        put(&work, "src/mine.rs", "rift wrote this\n");
        put(&work, "src/theirs.rs", "rift wrote this\n");
        mark(21, 1, &work).unwrap();
        // Somebody else edits one of them after the turn.
        put(&work, "src/theirs.rs", "someone else wrote this\n");

        let r = put_back(21, 1, &work).unwrap().unwrap();
        assert_eq!(get(&work, "src/mine.rs"), "before\n", "our own edit was not undone");
        assert_eq!(
            get(&work, "src/theirs.rs"),
            "someone else wrote this\n",
            "an edit made after the turn was clobbered"
        );
        assert_eq!(r.restored, 1);
        assert_eq!(r.changed_since, 1);
        done(&work);
    }

    #[test]
    fn a_created_file_something_else_has_since_written_to_is_kept() {
        let work = scratch("created-then-touched");
        put(&work, "keep.txt", "keep\n");
        take(22, 1, &work).unwrap();
        put(&work, "made.rs", "rift made this\n");
        mark(22, 1, &work).unwrap();
        put(&work, "made.rs", "and then changed\n");

        let r = put_back(22, 1, &work).unwrap().unwrap();
        assert!(work.join("made.rs").exists(), "a file changed since was deleted");
        assert_eq!(r.removed, 0);
        assert_eq!(r.changed_since, 1);
        done(&work);
    }

    #[test]
    fn without_a_record_of_the_turn_nothing_is_touched() {
        let work = scratch("unattributed");
        put(&work, "a.txt", "one\n");
        take(23, 1, &work).unwrap();
        put(&work, "a.txt", "two\n");
        put(&work, "b.txt", "new\n");
        // No mark(): the turn never recorded what it did.
        let r = put_back(23, 1, &work).unwrap().unwrap();
        assert!(r.unattributed);
        assert_eq!(get(&work, "a.txt"), "two\n");
        assert!(work.join("b.txt").exists());
        assert_eq!(r.restored + r.removed + r.changed_since, 0);
        done(&work);
    }

    #[test]
    fn build_output_is_neither_copied_nor_deleted() {
        let work = scratch("skip-dirs");
        put(&work, "src/a.rs", "a\n");
        put(&work, "target/debug/junk", "junk\n");
        put(&work, ".git/config", "cfg\n");
        let snap = take(2, 1, &work).unwrap();
        assert_eq!(snap.files, 1, "build output was copied");

        // Work inside target/ after the snapshot must survive the restore.
        put(&work, "target/debug/later", "later\n");
        mark(2, 1, &work).unwrap();
        let r = put_back(2, 1, &work).unwrap().unwrap();
        assert_eq!(r.removed, 0);
        assert!(work.join("target/debug/later").exists(), "build output was deleted");
        done(&work);
    }

    #[test]
    fn a_file_too_large_to_copy_is_left_alone_and_named() {
        let work = scratch("big-file");
        let big = vec![b'x'; (MAX_FILE + 1) as usize];
        fs::write(work.join("asset.bin"), &big).unwrap();
        put(&work, "src/a.rs", "a\n");
        let snap = take(3, 1, &work).unwrap();
        assert_eq!(snap.files, 1);
        assert_eq!(snap.skipped, 1);

        put(&work, "asset.bin", "changed\n");
        mark(3, 1, &work).unwrap();
        let r = put_back(3, 1, &work).unwrap().unwrap();
        assert_eq!(r.restored, 0, "a file the snapshot never held was written to");
        assert_eq!(get(&work, "asset.bin"), "changed\n", "asset was touched");
        done(&work);
    }

    #[test]
    fn the_snapshot_storage_is_never_copied_into_itself() {
        // The real layout: the data dir holds the database and the snapshots.
        // A snapshot that walks in there copies itself, then that copy, until
        // the growing path hits the limit. This is that bug, pinned.
        let work = scratch("self-nesting");
        let data = work.join("data/rift");
        fs::create_dir_all(&data).unwrap();
        TEST_ROOT.with(|r| *r.borrow_mut() = Some(data.join("snapshots")));
        put(&work, "src/a.rs", "a\n");
        put(&work, "data/rift/rift.db", "the session database\n");

        let snap = take(9, 1, &work).unwrap();
        assert_eq!(snap.files, 1, "the snapshot copied files from its own storage");
        let dest = data.join("snapshots/s9/turn-1");
        assert!(
            !dest.join("data/rift/snapshots").exists(),
            "the snapshot nested a copy of its own storage"
        );
        assert!(
            !dest.join("data/rift/rift.db").exists(),
            "the session database was copied into a snapshot"
        );

        // A rewind must leave the data dir alone: rolling the database back to
        // an earlier turn would take the sessions with it.
        put(&work, "src/a.rs", "changed\n");
        put(&work, "data/rift/rift.db", "later session data\n");
        mark(9, 1, &work).unwrap();
        let r = put_back(9, 1, &work).unwrap().unwrap();
        assert_eq!(get(&work, "src/a.rs"), "a\n");
        assert_eq!(
            get(&work, "data/rift/rift.db"),
            "later session data\n",
            "the database was rolled back"
        );
        assert_eq!(r.removed, 0, "files in the data dir were deleted");
        done(&work);
    }

    #[test]
    fn restoring_leaves_files_that_never_changed_alone() {
        let work = scratch("untouched");
        put(&work, "src/a.rs", "stable\n");
        put(&work, "src/b.rs", "before\n");
        take(13, 1, &work).unwrap();
        let stamp = fs::metadata(work.join("src/a.rs")).unwrap().modified().unwrap();

        put(&work, "src/b.rs", "after\n");
        mark(13, 1, &work).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let r = put_back(13, 1, &work).unwrap().unwrap();
        assert_eq!(r.restored, 1, "rewrote a file the turn never touched");
        assert_eq!(get(&work, "src/b.rs"), "before\n");
        // A build tool looks at this: an untouched file keeps its timestamp.
        assert_eq!(
            fs::metadata(work.join("src/a.rs")).unwrap().modified().unwrap(),
            stamp,
            "an unchanged file was rewritten"
        );
        done(&work);
    }

    #[test]
    fn a_tree_too_big_for_the_budget_keeps_additions_and_undoes_edits() {
        // The reason budgets exist: a working directory that was a projects
        // root, 68,000 files and 20 GB, copied before every single turn.
        let work = scratch("budget");
        TEST_BUDGET.with(|b| *b.borrow_mut() = Some((2, u64::MAX, f64::MAX)));
        for i in 0..6 {
            put(&work, &format!("src/f{i}.rs"), "one\n");
        }
        let snap = take(11, 1, &work).unwrap();
        assert!(snap.partial, "the budget did not stop the copy");
        assert!(snap.files <= 3, "copied past the budget: {}", snap.files);

        // A turn edits every file and creates one that was never captured.
        for i in 0..6 {
            put(&work, &format!("src/f{i}.rs"), "changed\n");
        }
        put(&work, "src/made-later.rs", "new\n");
        mark(11, 1, &work).unwrap();

        let r = put_back(11, 1, &work).unwrap().unwrap();
        assert!(r.partial);
        assert_eq!(r.removed, 0, "a partial snapshot deleted files");
        assert!(
            work.join("src/made-later.rs").exists(),
            "a file created since was deleted by a snapshot that never saw it"
        );
        // Whichever files the budget got to are the ones it must be able to
        // undo. Which ones depends on directory order, so ask the snapshot.
        let captured = listing(&dir_for(11, 1)).unwrap();
        assert!(!captured.is_empty(), "the budget captured nothing at all");
        for rel in &captured {
            assert_eq!(
                fs::read_to_string(work.join(rel)).unwrap_or_default(),
                "one\n",
                "a captured edit was not undone: {}",
                rel.display()
            );
        }
        done(&work);
    }

    #[test]
    fn a_two_turn_rewind_unwinds_both_turns_newest_first() {
        let work = scratch("two-turns");
        put(&work, "shared.txt", "A\n");
        put(&work, "first-only.txt", "A\n");

        take(31, 1, &work).unwrap();
        put(&work, "shared.txt", "B\n");
        put(&work, "first-only.txt", "B\n");
        mark(31, 1, &work).unwrap();

        take(31, 2, &work).unwrap();
        put(&work, "shared.txt", "C\n");
        put(&work, "second-only.txt", "new\n");
        mark(31, 2, &work).unwrap();

        let r = put_back_all(31, 1, 2, &work).unwrap().unwrap();
        assert_eq!(get(&work, "shared.txt"), "A\n", "the first turn's edit was left in place");
        assert_eq!(get(&work, "first-only.txt"), "A\n");
        assert!(!work.join("second-only.txt").exists(), "the second turn's file survived");
        assert_eq!(r.restored, 3);
        assert_eq!(r.removed, 1);
        assert_eq!(r.changed_since, 0);
        done(&work);
    }

    #[test]
    fn an_edit_from_outside_stops_that_path_but_not_the_earlier_turns() {
        let work = scratch("two-turns-guard");
        put(&work, "shared.txt", "A\n");
        put(&work, "mine.txt", "A\n");
        take(32, 1, &work).unwrap();
        put(&work, "shared.txt", "B\n");
        mark(32, 1, &work).unwrap();

        take(32, 2, &work).unwrap();
        put(&work, "shared.txt", "C\n");
        put(&work, "mine.txt", "B\n");
        mark(32, 2, &work).unwrap();
        // Somebody else writes the shared file after the second turn.
        put(&work, "shared.txt", "D from elsewhere\n");

        let r = put_back_all(32, 1, 2, &work).unwrap().unwrap();
        assert_eq!(
            (r.restored, r.removed, r.changed_since),
            (1, 0, 2),
            "unexpected restore tally: {r:?}"
        );
        assert_eq!(
            get(&work, "shared.txt"),
            "D from elsewhere\n",
            "a file edited since was rolled back"
        );
        assert_eq!(get(&work, "mine.txt"), "A\n", "the earlier turn was not unwound");
        done(&work);
    }

    #[test]
    fn a_rewind_of_turns_with_no_snapshots_touches_nothing() {
        let work = scratch("no-turns");
        put(&work, "a.txt", "one\n");
        assert!(put_back_all(33, 1, 3, &work).is_none());
        done(&work);
    }

    #[test]
    fn a_snapshot_that_fits_still_deletes_what_the_turn_added() {
        // The budget must not turn every restore into the timid kind.
        let work = scratch("budget-fits");
        put(&work, "src/a.rs", "a\n");
        let snap = take(12, 1, &work).unwrap();
        assert!(!snap.partial, "a two-file tree cannot be partial");
        put(&work, "src/new.rs", "new\n");
        mark(12, 1, &work).unwrap();
        let r = put_back(12, 1, &work).unwrap().unwrap();
        assert!(!r.partial);
        assert_eq!(r.removed, 1);
        done(&work);
    }

    #[test]
    fn no_snapshot_is_reported_as_none_and_pruning_keeps_the_newest() {
        let work = scratch("prune");
        put(&work, "a.txt", "a\n");
        assert!(put_back(4, 1, &work).is_none(), "missing snapshot must read as none");

        for turn in 1..=(KEEP + 3) {
            put(&work, "a.txt", &format!("turn {turn}\n"));
            take(4, turn, &work).unwrap();
        }
        let have = turns(4);
        assert_eq!(have.len(), KEEP, "pruning did not hold the cap");
        assert_eq!(*have.last().unwrap(), KEEP + 3, "newest snapshot was pruned");
        assert!(!have.contains(&1), "oldest snapshot survived");

        // Dropping back to a turn forgets only what came after it.
        drop_after(4, KEEP + 1);
        let after = turns(4);
        assert_eq!(*after.last().unwrap(), KEEP + 1);
        assert_eq!(after.len(), KEEP - 2);
        assert!(!after.contains(&(KEEP + 2)) && !after.contains(&(KEEP + 3)));
        done(&work);
    }
}
