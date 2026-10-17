# RIFT · shell-harness — Checkup

Mode: `/design checkup`
Date: 2026-10-16
Artifact: a Rust terminal UI (`shell-harness/src/*.rs`). This read supersedes the pre-deslop checkup: both previous Criticals are resolved, and the six vitals below are re-measured on the current binary.

---

## Score: 55 / 60

| Vital | Status | Score |
|---|---|---|
| Intentionality | Healthy | 10 |
| Readability | Healthy | 10 |
| Usability | Healthy | 10 |
| Responsiveness | Watch | 5 |
| Speed | Healthy | 10 |
| Accessibility | Healthy | 10 |

No Criticals. Two contained width gaps remain, both in printed chrome rather than the interactive frame.

---

## Vital Signs

**Intentionality — Healthy (10).** The surface is chosen at every level now: one color role per code, documented in the source (`ui.rs:5-13`); a glyph vocabulary where every mark has a job (`❯` machine, `>` you, `!` warning, `[thinking]`, `↑/↓`, `⠋`); a heading ladder; a table renderer that computes its own columns; and an arrival line that states the category (`rift · shell agent · openai · mock-1`). Nothing here reads as an assembled default.

**Readability — Healthy (10).** Hierarchy verified in raw SGR output: `[32m` prompt, `[36m` tool echo, `[2m` output and reasoning, `[1m` headings and table header, `[4m` inline code, `[7m` selection. Tables align at both 100 and 40 columns; long output clamps per line and keeps its tail; reasoning is labeled instead of masquerading as the answer. One residual risk, not a defect: `DIM` is rendered by the user's terminal theme, so in a low-contrast theme the secondary content is faint by the theme's choice, not by the app's.

**Usability — Healthy (10).** Primary flow verified end to end at 100×30 and 40×14: type, reasoning, tool echo, output, truncation notice, answer, prompt returns. Esc cancels once (was twice). Bare `/model` opens the live picker, verified through pick → submit → `model set to mock-1`. Failures carry an `error: ` prefix and warnings a `!`. Empty state for zero connected providers is implemented and code-path verified, but not observed live: environment provider keys make that branch unreachable in this environment.

**Responsiveness — Watch (5).** The interactive frame adapts: at 25×8 the menu caps at six rows with `↓ 5 more` and every row clamps; at 40×14 the transcript stays coherent; a 100×30 → 44×12 resize repaints correctly with no keypress. Two gaps sit outside the frame: three chrome lines are still printed at natural length (at 40 columns the truncation notice breaks mid-word, at 30 the banner does), and the printed reports (`/help`, the session list) wrap mid-word and lose their column alignment below ~60 columns. See the prescriptions; both are contained, neither hides content.

**Speed — Healthy (10).** Idle cost of the 200ms prompt poll measured at **0.00% CPU over 4 seconds** (zero scheduler ticks). Streamed tokens print on arrival, the spinner clears on the first delta, and no capture across this pass showed flicker, shift, or jank. Network latency is not measured (local mock only).

**Accessibility — Healthy (10).** No escalation trigger stands. Content is reachable at every width tested down to 25×8; no state is carried by hue alone (`error: `, `!`, `[thinking]`, and distinct glyphs carry the meaning); the keyboard is the only input path and it is complete; the single motion is one cell and stops at the first token. Residual risk: `DIM` contrast is unverifiable here because it belongs to the user's theme.

---

## Prescriptions

| # | Severity | Discipline | Location | Before | After | Why |
|---|---|---|---|---|---|---|
| 1 | MEDIUM | Responsiveness | `src/ui.rs:360` (truncation notice), `src/ui.rs:38` (`banner`), `src/ui.rs:407` (`warn`) | Chrome lines print at natural length: at 40 columns `… 3 more lines (full output sent to mode` / `l)`; at 30 the banner splits `openai ·` / `mock-1` | Clamp these three lines with the existing helper (`ui.rs:391`), as the frame already does for its rows | The frame became width-aware and these three lines did not, so the breakage now reads as a defect in exactly the widths where everything else holds. Prose (answers, reasoning, reports) should keep wrapping, where wrapping preserves meaning |
| 2 | MEDIUM | Responsiveness | `src/commands.rs:125` (`help`), `src/agent.rs:187` (`resume_list`) | At 40 columns `/help` wraps mid-word and its aligned columns collapse (`show or switch provider (ke` / `y=/url= inline)`) | Reflow below ~60 columns: name on its own line, description indented under it. Alternatively route `/?` to the live menu, which already adapts | Reports are a scanning surface; a wrapped alignment is worse than no alignment |

Fix modes: `/design finish` for both (writing the chrome clamp is mechanical; the report reflow is a small layout change).

---

## Considered but Rejected

| Location | Candidate | Rejected because |
|---|---|---|
| `src/agent.rs:123-128` (`/rewind`) | Irreversible with no confirmation (dropped turns are deleted from the session store) | The command is typed with an explicit count, so the number is the confirmation; it is the undo mechanism for a bad turn, and it reports what it removed. Not a control that looks safe |
| `src/ui.rs:8` (`DIM` for tool output) | Contrast failure | Single role now, and the alternative (default weight for output) would flatten the transcript's core hierarchy. The actual color is the theme's decision, so this is logged as a risk, not a defect |
| `src/ui.rs:38` (banner) | Add cwd to the arrival line | Vertical space is the scarce resource; `-C` is startup-fixed and `/status` names it |
| `src/ui.rs:681` (200ms poll) | Wakeups cost battery | Measured at 0.00% CPU over 4s; a 5Hz poll on stdin is below measurement noise |
| Streaming text and reasoning | Clamp them like chrome | Prose must wrap; clamping would silently destroy model output |

---

## Verification

Ran:

- `cargo test --offline` → 71 passed, 0 failed.
- tmux 3.4 captures on the current binary: 100×30 full turn; 40×14 full turn plus 40-line scrollback; 25×8 menu; 100×30 → 44×12 resize with no keypress; 30-column arrival.
- Raw `capture-pane -e` SGR read for the color roles (codes listed under Readability).
- Idle cost: `/proc/<pid>/stat` ticks sampled 4s apart → 0 ticks.
- States: warning (`!`), failure (`error: `), cancellation (`stopped (Esc)`, once), model picker pick → submit.

Not verified:

- The zero-provider empty state. Provider keys exist in this environment, so the `connected_catalog` branch could not be reached; it is code-verified only.
- Real provider streaming over a network, terminals other than tmux, light themes (`DIM` contrast).

---

## Verdict

**Needs changes** — nothing blocks, and two contained Responsiveness findings are outstanding. `checkup-report.html` carries the same read for scanning.
