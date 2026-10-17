# RIFT · shell-harness — Review

Mode: `/design review`
Date: 2026-10-16
Artifact: a Rust terminal UI (`shell-harness/src/*.rs`), walked live in tmux against a local mock provider.

---

## Score: 31 / 50

| Lens | Score |
|---|---|
| First impression | 6 / 10 |
| Hierarchy | 7 / 10 |
| Color voice | 5 / 10 |
| Type voice | 6 / 10 |
| Interaction feel | 7 / 10 |

The work is honest and specific; it is held back by three things — no width model, an unlabeled reasoning stream, and a palette handed out per call site.

---

## Gut Reaction

Within ten seconds of a turn this feels like a well-behaved CLI: quiet, fast, no ceremony, no mascot, no gradient. The tool echo with its `❯` spine and the dim output under it are the best idea on the surface — you can always see what the machine actually ran, in order, without leaving the transcript.

At the same time, nothing on screen says what this is. `rift  openai · gpt-4.1` names a binary and a model. The one interaction a new user needs first (`/` opens the command menu) is invisible until discovered. And the first thing written after your prompt is the model thinking out loud, in dim text, unlabeled — which a reader will take as the answer.

---

## Primary Flow Walked

Type a request → spinner → reasoning streams → answer streams → tool call echoes (`❯ ls -la src | head -40`) → output prints dim → `… 3 more lines (full output sent to model)` → answer resumes → prompt returns with history intact. Esc cancels mid-turn and the turn ends cleanly. Empty Enter does not spam the scrollback. Resize: the interface does not survive it.

---

## The Five Lenses

**First impression (6/10).** Quiet and competent, with no claim about itself. The banner is one dim line: name, provider, model. Category is not visible; the arrival screen would fit any model-agnostic REPL. The menu, once found, is the most memorable thing here and it is hidden behind a character the user has to guess.

**Hierarchy (7/10).** Inside a turn the ordering is real: your line bold, the machine's command in yellow with a glyph, output dim, the answer at full weight. Two breaks: reasoning and answer share a weight and a shape, and six heading levels collapse to one bold treatment, so a three-level document arrives flat.

**Color voice (5/10).** Eight ANSI codes carry twelve jobs. Yellow is both the missing-key warning and the tool echo; green is both the wordmark and the prompt mark; cyan is both a system notice and an inline code span; dim is tool output, reasoning, fences and the truncation notice. The messages a failure can print are three different colors depending on which code path produced them: plain (`Action::Message`), cyan (`ui::info`), red (`ui::error`). Using ANSI named colors is the right call for a terminal — it inherits the user's theme — but the assignment is per call site, so there is no state color to learn.

**Type voice (6/10).** The medium hands over the font and keeps the weights: bold, dim, reverse, underline. Bold for headings and emphasis, dim for secondary content, reverse for the menu selection are all correct. The scale stops there — `#`, `##`, `###` are one treatment — and tables fabricate an alignment they do not compute.

**Interaction feel (7/10).** The strongest lens. The live menu is genuinely good: a 10-row window that scrolls around the selection with `↑/↓ N more` hints, Tab completing without skipping ahead, Enter accepting then submitting on the second press, Esc hiding the menu then clearing the line, and surplus rows always wiped. Cancellation is instant and honest ("stopped (Esc)"). Deductions: reasoning is unlabeled, bare `/model` prints a static catalog under a header that describes the live menu, and the spinner says "working".

**Smell lens.** Two tracked tells present (fabricated table structure, flat heading scale) → 8/10 FAINT on `/design smell`. This does not look generated. It looks like a good CLI that has not designed its edges.

---

## Top Improvements, In Order

| # | Severity | Location | Finding | Mode |
|---|---|---|---|---|
| 1 | HIGH | `src/ui.rs:517`, `src/ui.rs:533`, `src/term.rs` | No width model: at 40 columns the prompt and banner leave the pane; resize leaves scrambled rows (verified). Escalation trigger — outranks every aesthetic finding here | `interaction` / `relayout` |
| 2 | MEDIUM | `src/agent.rs:465-500`, `src/ui.rs:47` | Reasoning streams unlabeled and reads as the answer; the spinner says "working" | `writing` |
| 3 | MEDIUM | `src/ui.rs:5-12`, `:22-34`, `:232-266`, `:884`; `src/agent.rs:128` | No color role system; state is hue-first; the failure class prints in three different treatments | `recolor` |
| 4 | MEDIUM | `src/ui.rs:101-116` | Heading collapse and unconputed table alignment in the markdown renderer | `typeset` |
| 5 | MEDIUM | `src/ui.rs:238-256`; `src/shell.rs:82`, `:91` | The 12-line output cut hides `[stderr]` and `[exit N]`, which `shell.rs` appends at the end | `interaction` |
| 6 | LOW | `src/ui.rs:22-34` | The arrival screen does not state the category: add the product's own word for itself to the banner | `voice` |
| 7 | LOW | `src/commands.rs:122-158` | Bare `/model` prints a static dump under menu copy | `finish` |

---

## Considered but Rejected

| Location | Candidate | Rejected because |
|---|---|---|
| `src/ui.rs:5-12` | Authored palette to escape "developer tool = terminal green" | ANSI named colors inherit the user's theme; the domain default here is the medium itself, not a reflex to break. The real work is role assignment |
| `src/ui.rs:22-34` | Persistent status line with cwd | Vertical space is the scarce resource; `-C` is startup-fixed and `/status` names it |
| `src/ui.rs:27-40` | Reduced-motion option for the spinner | No such channel exists in a terminal; single-cell spinner, stops on first token |

---

## Verdict

**Block** — escalation trigger standing (content unreachable on a narrow terminal). Scores are otherwise mid-band: a focused three-mode pass (interaction, recolor, typeset) moves this to a 45+/50 surface without touching its character.
