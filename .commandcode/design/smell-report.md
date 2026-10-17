# RIFT · shell-harness — Smell Report

Mode: `/design smell`
Date: 2026-10-16
Artifact: a Rust terminal UI (`shell-harness/src/*.rs`), not a web surface. The interface is ANSI escape output: banner, prompt line editor, live `/` menu, streamed markdown, tool echo. Odors that require a rendering vector (gradients, blur, icon tiles) are structurally absent here; where a web tell has a terminal analogue, the mapping is named in the row.

---

## Score: 8 / 10 — FAINT

Two of the ten tracked tells are present. The interface is not generic; it is under-designed at the edges.

## TL;DR

The transcript vocabulary is specific and deliberate: `❯` for tool commands, dim for output, reverse-video for the live menu, a 10-row window with `↑/↓ N more` hints, ghost rows wiped on backspace and Esc. The copy has a voice ("gone: set it inline now", "stopped (Esc)", "… 12 more lines (full output sent to model)"). None of that reads as template output.

What is missing is a model of the window it runs in, a role system for color, and a type scale for model output. One of those is a verified failure, not a wart: below roughly 80 columns the frame comes apart, and it does not recover on resize.

**Primary recommendation:** `/design interaction` (or `relayout`) to give the prompt and menu a size model, then `/design recolor` for color roles and `/design typeset` for the markdown scale.

---

## Heuristic Scores

Each row is `1` when the odor is absent, `0` when it is present.

| # | Odor | Score | Key finding |
|---|---|---|---|
| 1 | Tech gradient | 1 | No gradients. Sixteen-color ANSI only; no truecolor, no 256-color theatre |
| 2 | Generic tech hue | 1 | Brand and semantics ride the terminal's named colors. Note: flagging terminal-green as "default" would be inventing odor — named ANSI colors inherit the user's theme, which is the medium-correct choice. The real problem is role assignment (finding 2), not the hues |
| 3 | Feature tile grid | 1 | Single-column transcript. No card grids, no equal-weight panels |
| 4 | Accent rail | 0 | No rails in a TUI. The analogue — decoration simulating structure — is the table rule in `render_line`: rules sized from the markdown dashes, not from the columns above them (finding 4) |
| 5 | Unearned blur | 1 | No depth system to fake. `DIM` is a text style, not decoration |
| 6 | Stat monument | 1 | One-line banner. No number clusters, no metric tiles |
| 7 | Icon topper | 1 | No decorative iconography. Every glyph is functional: `❯` command, `>` prompt, `↑/↓` menu, `⠋` spinner, `…` truncation |
| 8 | Bounce everywhere | 1 | The braille spinner is the only motion, single-cell, stops on the first streamed token. Standard affordance for the medium |
| 9 | Default type | 0 | No type scale. Six markdown heading levels render identically (finding 3), and tables fake alignment (finding 4) |
| 10 | Center stack | 1 | Everything is left-aligned and anchored to column 0. No centering |

**Tells found: 2 → 8/10 — FAINT.**

---

## Findings

Ordered by severity, then by reach and leverage. Discipline names the reference that owns the rule.

| # | Severity | Discipline | Location | Before | After | Why |
|---|---|---|---|---|---|---|
| 1 | HIGH | Interaction | `src/ui.rs:517` (`MENU_ROWS`), `src/ui.rs:533` (`frame`), `src/term.rs` (no size listener) | Menu rows are emitted at their natural length and `frame` moves the cursor up by its logical row count. Nothing ever asks the terminal how wide it is: `TIOCGWINSZ`, `winsize`, `SIGWINCH` have zero hits in `src/` | Read the window size before the first frame and on resize; clamp each row to `width − 1` with an ellipsis; re-render the frame on `SIGWINCH`; scroll the input line instead of letting it wrap | Verified at 40 columns: the banner and the `> /` prompt leave the visible pane, rows split mid-word, and after a resize one keypress paints text into stale rows |
| 2 | MEDIUM | Color | `src/ui.rs:5-12`, `:22-34`, `:47`, `:232-266`, `:884`; `src/agent.rs:128` | Eight codes carry twelve jobs: `YELLOW` is both the missing-key warning and the tool echo, `GREEN` is both the wordmark and the prompt mark, `CYAN` is both `info` and inline code, `DIM` is tool output, thinking, fences and the truncation notice. The failure class renders three ways: plain `println!` (`Action::Message`), cyan (`ui::info`), red (`ui::error`) | One role per code. Give error and warning a glyph or a word (`✗`, `!`) so severity survives without hue. Route every user-facing message through a role instead of the bare `println!` | Severity is decided per call site: "unknown command" prints plain, "usage: /rewind [n]" prints cyan, "no API key" prints red — the same class of news at three volumes |
| 3 | MEDIUM | Type | `src/ui.rs:113-116` | Every heading level 1–6 collapses to one treatment (`hashes` 1..=6 → `style(BOLD, …)`), so `#`, `##` and `###` are indistinguishable | Map the levels onto the media the terminal has: H1 bold + reverse, H2 bold, H3 bold dim, H4–6 dim | Model answers arrive with three-level heading trees and lose all of it. The medium offers weight, reverse and dim; the renderer uses one |
| 4 | MEDIUM | Layout | `src/ui.rs:101-108` | Table rules are built from the source dashes (`\|---\|---\|` → `───  ───`) and cells are joined with a fixed two spaces, so the rule never matches the column above it and no two rows align | Buffer a table block, size columns to their content, pad cells, span each rule across the column width; or drop per-column rules for one dim full-width rule | A structure that reads as a table but scans worse than the plain lines it replaced. The in-tree test asserts the fake rule verbatim (`ui.rs:1213`) |
| 5 | MEDIUM | Interaction | `src/ui.rs:238-256`; `src/shell.rs:82`, `:91` | The display shows the first 12 lines and summarizes the rest, but `shell.rs` appends `[stderr]` and `[exit N]` at the end of the payload — so for any command longer than 12 lines, the error block and the exit code are exactly what gets hidden | Show head and tail (e.g. 8 + 4), or always surface the trailer line; keep the "N more lines" notice | The two lines that tell an operator whether a command worked are the ones the cut removes |
| 6 | LOW | Interaction | `src/commands.rs:122-158`; `src/agent.rs:128` | Bare `/model` prints the entire catalog as static text headed "type to search, ↑↓ pick, Enter sets" — copy that describes the live menu, on a list where none of it works | Open the live menu on bare `/model` (prefilled with the current id), or cap the dump and say "refine with /model `<fragment>`" | With a fetched OpenRouter catalog, bare `/model` is hundreds of rows of scrollback whose header promises navigation it does not have |
| 7 | LOW | Writing | `src/agent.rs:465` | Spinner label is `working` | Name the phase or the model: `waiting on qwen3.7-flash`, `thinking (medium)` | `compacting` names its work; `working` names nothing |

---

## Evidence: the 40-column capture

tmux 3.4, fresh build, 40×14. Verbatim rendered grid after typing `/` (the banner and the `> /` prompt are not on screen at all):

```
  /model     show or set the model for t
his session
  /thinking  show or set thinking effort
 (off|minimal|low|medium|high|xhigh|max)
  /compact   summarize history to free c
ontext
  /btw       side question: same context
, history untouched
  /resume    list past sessions or resum
e one by id
  /rewind    drop the last turn(s), e.g.
 /rewind 2
  /reset     clear conversation history
↓ 1 more
```

Same session resized 100×30 → 44×12, then one keypress:

```
  /model  show or set the model for this ses
sionnimal|low|medium|high|xhigh|max)
```

The new frame painted into the stale one. At 100×30 the same interaction is clean: aligned rows, `↓ 1 more` footer, prompt visible.

---

## Considered but Rejected

| Location | Candidate | Rejected because |
|---|---|---|
| `src/ui.rs:5-12` | Replace the ANSI named palette (terminal-green brand) with an authored one | Named ANSI colors inherit the user's terminal theme, which is the accessible, light-theme-safe choice for a TUI. Flagging the medium's correct answer as "default" would be inventing odor; finding 2 is about assignment, not hues |
| `src/ui.rs:27-40` | Spinner needs a no-motion mode | Terminals expose no `prefers-reduced-motion` channel. The spinner is single-cell, standard for the medium, and stops at the first token |
| `src/ui.rs:22-34` | Banner shows provider and model but never the working directory | Vertical space is the scarce resource in a TUI; `-C` is fixed at startup and `/status` names the cwd. Deliberate |
| `src/ui.rs:884`, `:232` | Unify `>` (user line) and `❯` (tool echo) into one glyph | The two glyphs mark different authors — you versus the machine. A working lexical distinction, not drift |

---

## Verification

Ran:

- `cargo test --offline` → **66 passed, 0 failed**. Includes `md_renders_and_strips` (asserts `| a | b |` → `a  b` and `|---|---|` → `───  ───`), `frame_wipes_stale_rows`, `frame_truncates_long_menus_with_hints`, `head_lines_truncates_with_count`.
- `cargo build --offline` → current debug binary. Note: the checked-in `target/release/rift` is stale (built 2026-10-15 23:54 against source edited 2026-10-16 10:15) and renders a different, boxed UI — a first capture against it was discarded. All findings are verified against a current build.
- tmux 3.4 terminal-emulator captures: 100×30 (baseline, clean), 40×14 (finding 1), and a 100×30 → 44×12 resize (finding 1).
- Source trace of every call site for findings 2, 5, 6, 7 (message paths, output assembly, catalog printing, spinner label).

Not verified:

- Live model streaming: thinking dimming, the streaming markdown renderer, the tool echo and the 12-line truncation notice as they appear on screen. No provider key is available in this environment; these were read from source and unit tests only (finding 5's truncation logic is unit-tested; its live appearance is not confirmed).
- Terminals other than tmux (kitty, iTerm2, Windows Terminal, light themes).

---

## Verdict

**Block** — one HIGH is standing (finding 1).

The smell score is FAINT because the interface is not generated-looking; the HIGH is a defect, not a sameness tell. The report takes both.

## Next modes

- `/design interaction` or `/design relayout` — the width model: `TIOCGWINSZ`, `SIGWINCH`, row clamping, input scrolling (finding 1, finding 5)
- `/design recolor` — one role per code; glyphs for error and warning (finding 2)
- `/design typeset` — heading scale and table alignment in the markdown renderer (findings 3, 4)
- `/design finish` — `/model` dump and spinner copy (findings 6, 7)
