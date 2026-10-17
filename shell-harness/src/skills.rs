//! Skills: instructions a project or the user keeps on disk, loaded on demand.
//!
//! A skill is a directory holding a `SKILL.md`: a small YAML frontmatter
//! (`name`, `description`) then a body of instructions. Only the one-line
//! description ever reaches the model unasked; the body arrives when the model
//! asks for it by name through the `skill` tool. A collection of twenty skills
//! therefore costs twenty lines of prompt instead of twenty documents nobody
//! needed, which is the whole reason the body is deferred.

use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};

/// One skill: what it is called, what it is for, and where its file lives.
/// The body is deliberately not kept: it is read when it is asked for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Skill {
    pub name: String,
    pub description: String,
    pub path: PathBuf,
}

/// Where skills are looked for, in precedence order (earlier wins): the
/// project's own, then the user's rift directory, then `~/.claude/skills` so a
/// collection already written for another harness is not stranded.
/// `RIFT_SKILLS_DIR` replaces the list entirely, which is what tests use.
fn search_dirs(cwd: &Path) -> Vec<PathBuf> {
    if let Ok(list) = std::env::var("RIFT_SKILLS_DIR") {
        return std::env::split_paths(&list)
            .filter(|s| !s.as_os_str().is_empty())
            .collect();
    }
    let mut dirs = vec![cwd.join(".rift").join("skills")];
    if let Some(home) = crate::config::home_dir() {
        dirs.push(home.join(".config").join("rift").join("skills"));
        dirs.push(home.join(".claude").join("skills"));
    }
    dirs
}

/// Split a `SKILL.md` into (name, description, body).
///
/// Frontmatter is a fence-delimited block of `key: value` lines. Anything
/// missing falls back rather than failing: a skill someone wrote by hand with
/// no frontmatter still works, with its directory name as its name and no
/// description. That is a whole skill better than a hard error at startup.
pub fn parse(text: &str, fallback_name: &str) -> (String, String, String) {
    let mut name = String::new();
    let mut description = String::new();
    let mut lines = text.lines();
    let fenced = lines.next().is_some_and(|l| l.trim() == "---");
    let body = if fenced {
        for line in lines.by_ref() {
            if line.trim() == "---" {
                break;
            }
            let Some((k, v)) = line.split_once(':') else { continue };
            let v = unquote(v.trim());
            match k.trim() {
                "name" => name = v,
                "description" => description = v,
                _ => {}
            }
        }
        lines.collect::<Vec<_>>().join("\n")
    } else {
        text.to_string()
    };
    let name = if name.is_empty() { fallback_name.to_string() } else { name };
    (name, description, body.trim_matches('\n').to_string())
}

/// Strip the one layer of matching quotes a hand-written value may carry.
fn unquote(v: &str) -> String {
    let b = v.as_bytes();
    if b.len() >= 2 && (b[0] == b'"' || b[0] == b'\'') && b[b.len() - 1] == b[0] {
        return v[1..v.len() - 1].to_string();
    }
    v.to_string()
}

/// Every skill on disk for this working directory, project first, so a project
/// skill shadows a user one with the same name.
pub fn discover(cwd: &Path) -> Vec<Skill> {
    let mut out: Vec<Skill> = Vec::new();
    for dir in search_dirs(cwd) {
        for skill in scan(&dir) {
            if !out.iter().any(|s| s.name == skill.name) {
                out.push(skill);
            }
        }
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

/// The skills in one directory: `<dir>/<name>/SKILL.md` and the flat
/// `<dir>/<name>.md` convenience. A directory with no `SKILL.md` is not a
/// skill, so an unrelated folder in the path cannot invent one.
fn scan(dir: &Path) -> Vec<Skill> {
    let Ok(entries) = std::fs::read_dir(dir) else { return Vec::new() };
    let mut out = Vec::new();
    let mut found: Vec<(PathBuf, String)> = Vec::new();
    for e in entries.flatten() {
        let path = e.path();
        let stem = path.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
        if path.is_dir() {
            let doc = path.join("SKILL.md");
            if doc.is_file() {
                found.push((doc, stem));
            }
        } else if path.extension().is_some_and(|x| x == "md") {
            found.push((path, stem));
        }
    }
    found.sort();
    for (path, stem) in found {
        let Ok(text) = std::fs::read_to_string(&path) else { continue };
        let (name, description, _) = parse(&text, &stem);
        out.push(Skill { name, description, path });
    }
    out
}

/// The skill's own name, or the closest thing we can offer instead of a shrug.
pub fn load(cwd: &Path, name: &str) -> Result<String> {
    let skills = discover(cwd);
    if skills.is_empty() {
        bail!("no skills are installed. Put one at .rift/skills/<name>/SKILL.md");
    }
    let Some(skill) = skills.iter().find(|s| s.name == name) else {
        let names: Vec<&str> = skills.iter().map(|s| s.name.as_str()).collect();
        bail!("no skill named {name}. Available: {}", names.join(", "));
    };
    let text = std::fs::read_to_string(&skill.path)
        .with_context(|| format!("reading {}", skill.path.display()))?;
    let (_, _, body) = parse(&text, name);
    if body.trim().is_empty() {
        bail!("skill {name} has no instructions in it ({})", skill.path.display());
    }
    // The directory is named because a skill usually carries files beside its
    // SKILL.md, and the model reaches those with the shell.
    let dir = skill.path.parent().unwrap_or(Path::new("."));
    Ok(format!(
        "# skill: {name}\n{}\n\n(The rest of this skill's files, if any, are in {}.)",
        body,
        dir.display()
    ))
}

/// The one-line-each catalog that goes in front of the model. Empty when there
/// are no skills, so an unconfigured run carries no heading about nothing.
pub fn catalog(cwd: &Path) -> String {
    let skills = discover(cwd);
    if skills.is_empty() {
        return String::new();
    }
    let mut out = String::from("Skills available (call the `skill` tool with the name to load one):\n");
    for s in &skills {
        let desc = if s.description.is_empty() { "(no description)" } else { s.description.as_str() };
        out.push_str(&format!("- {}: {}\n", s.name, desc));
    }
    out.trim_end().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    static ENV_GUARD: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Run `f` with `RIFT_SKILLS_DIR` pointed at a fresh temp directory.
    fn with_skills(f: impl FnOnce(&Path)) {
        let _g = ENV_GUARD.lock().unwrap_or_else(|e| e.into_inner());
        let dir = std::env::temp_dir().join(format!(
            "rift-skills-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let saved = std::env::var("RIFT_SKILLS_DIR").ok();
        std::env::set_var("RIFT_SKILLS_DIR", &dir);
        f(&dir);
        match saved {
            Some(v) => std::env::set_var("RIFT_SKILLS_DIR", v),
            None => std::env::remove_var("RIFT_SKILLS_DIR"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn write(dir: &Path, rel: &str, body: &str) {
        let path = dir.join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, body).unwrap();
    }

    #[test]
    fn frontmatter_supplies_the_name_and_the_description() {
        let text = "---\nname: deploy\ndescription: ship it\nother: ignored\n---\n\nDo the thing.\n";
        let (name, desc, body) = parse(text, "fallback");
        assert_eq!(name, "deploy");
        assert_eq!(desc, "ship it");
        assert_eq!(body, "Do the thing.");
        // Quoted values are unquoted, once.
        let (_, desc, _) = parse("---\ndescription: \"two words\"\n---\nx", "f");
        assert_eq!(desc, "two words");
    }

    #[test]
    fn a_file_without_frontmatter_is_still_a_skill() {
        let (name, desc, body) = parse("Just instructions.\n", "handwritten");
        assert_eq!(name, "handwritten");
        assert_eq!(desc, "");
        assert_eq!(body, "Just instructions.");
    }

    #[test]
    fn discovery_reads_directories_and_flat_files() {
        with_skills(|dir| {
            write(dir, "alpha/SKILL.md", "---\ndescription: first\n---\nbody a\n");
            write(dir, "beta.md", "---\nname: beta\ndescription: second\n---\nbody b\n");
            // A folder with no SKILL.md is not a skill.
            write(dir, "gamma/notes.txt", "not a skill\n");
            let found = discover(Path::new("/tmp"));
            let names: Vec<&str> = found.iter().map(|s| s.name.as_str()).collect();
            assert_eq!(names, vec!["alpha", "beta"]);
            assert_eq!(found[0].description, "first");
        });
    }

    #[test]
    fn loading_returns_the_body_without_the_frontmatter() {
        with_skills(|dir| {
            write(dir, "deploy/SKILL.md", "---\ndescription: ship it\n---\n\nRun make deploy.\n");
            let text = load(Path::new("/tmp"), "deploy").unwrap();
            assert!(text.starts_with("# skill: deploy\nRun make deploy."), "{text}");
            assert!(!text.contains("description:"), "{text}");
            assert!(text.contains("/deploy"), "the directory is named: {text}");
        });
    }

    #[test]
    fn an_unknown_name_lists_what_is_available() {
        with_skills(|dir| {
            write(dir, "alpha/SKILL.md", "---\ndescription: first\n---\nbody\n");
            let err = load(Path::new("/tmp"), "nope").unwrap_err().to_string();
            assert!(err.contains("no skill named nope"), "{err}");
            assert!(err.contains("alpha"), "{err}");
        });
    }

    #[test]
    fn no_skills_means_no_catalog_section() {
        with_skills(|_dir| {
            assert_eq!(catalog(Path::new("/tmp")), "");
            assert!(load(Path::new("/tmp"), "x").is_err());
        });
    }

    #[test]
    fn the_catalog_is_one_line_per_skill() {
        with_skills(|dir| {
            write(dir, "alpha/SKILL.md", "---\ndescription: first thing\n---\nbody\n");
            write(dir, "beta/SKILL.md", "body only\n");
            let c = catalog(Path::new("/tmp"));
            assert!(c.contains("- alpha: first thing"), "{c}");
            assert!(c.contains("- beta: (no description)"), "{c}");
            assert_eq!(c.lines().count(), 3, "{c}");
        });
    }
}
