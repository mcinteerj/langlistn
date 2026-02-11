# Handoff Notes

## 1. The Objective

**Goal:** Test `langlistn` end-to-end using tmux against a Korean YouTube video playing in Google Chrome. Evaluate the terminal UX — formatting, readability, visual glitches, timing of locked vs speculative text, status bar, paragraph breaks, header/footer. Identify and fix issues.

**Definition of Done:** The tool runs smoothly against live Korean audio, output is readable and well-formatted, no visual glitches (flickering, ghost lines, misaligned text), and the UX feels polished for a non-technical user watching subtitles.

## 2. State & Resources

### Current Status
- **Working:** Full pipeline — audio capture → Silero VAD → mlx-whisper large-v2 → LocalAgreement-2 → Claude translation (Bedrock) → terminal display
- **Working:** Interactive setup wizard, type-to-filter app picker, quick rerun, loading spinner, AWS auth error handling
- **Untested since recent changes:** The three UX branches (polish, status, loading) were built by subagents and merged but have NOT been manually tested yet. This is the first real test.
- **Known cosmetic issues from prior testing:**
  - Paragraph breaks are mechanical (every 3 sentences) — may look odd with short sentences
  - Display had prior flicker bugs that were supposedly fixed but need verification
  - Characters occasionally missing at start of speculative text (was fixed, needs re-verification)

### Key Assets
```
Project root:     /Users/jakemc/projects/langlistn
Venv:             .venv/bin/python
Entry point:      .venv/bin/langlistn
Swift binary:     swift/.build/release/AudioCaptureHelper
Config saved at:  ~/.config/langlistn/last.json
```

**Key source files:**
| File | Role | Lines |
|------|------|-------|
| `src/langlistn/__main__.py` | CLI + interactive wizard | 501 |
| `src/langlistn/pipeline.py` | Two-loop orchestrator | 556 |
| `src/langlistn/display.py` | Terminal renderer (two-zone) | 276 |
| `src/langlistn/translate.py` | Bedrock Claude translation | 400 |
| `src/langlistn/streaming_asr.py` | Whisper + LocalAgreement-2 | 246 |
| `src/langlistn/config.py` | Constants, language map | 56 |

### Tooling — Test Commands

```bash
# Install
cd ~/projects/langlistn && .venv/bin/pip install -e . -q

# Basic test (have a Korean YouTube video playing in Chrome)
.venv/bin/langlistn --app "Google Chrome" --source ko --debug-log /tmp/langlistn-debug.log

# With sonnet for better quality
.venv/bin/langlistn --app "Google Chrome" --source ko --translate-model sonnet --debug-log /tmp/langlistn-debug.log

# Interactive mode (tests wizard + quick rerun)
.venv/bin/langlistn

# Transcribe only (no AWS needed)
.venv/bin/langlistn --app "Google Chrome" --source ko --no-translate

# Check debug log for multi-hypothesis and LLM calls
tail -f /tmp/langlistn-debug.log
```

**For tmux testing:**
```bash
# Start a tmux session
tmux new -s langlistn-test

# Run the tool
cd ~/projects/langlistn && .venv/bin/langlistn --app "Google Chrome" --source ko --debug-log /tmp/langlistn-debug.log

# In another pane (Ctrl-b %), watch debug log
tail -f /tmp/langlistn-debug.log
```

### Environment
- macOS Apple Silicon (M1+)
- AWS credentials required for translation (run `aws sso login` first)
- Bedrock model IDs: haiku=`global.anthropic.claude-haiku-4-5-20251001-v1:0`, sonnet=`global.anthropic.claude-sonnet-4-5-20250929-v1:0`
- Terminal must have Screen Recording AND System Audio Recording permissions
- First run downloads whisper model (~3GB), subsequent runs start in seconds

## 3. Knowledge Transfer

### What was tried (this session)
1. Fixed display bug where "starting audio capture" left stale erasable line count → subsequent display updates erased wrong lines. **Fix:** use raw `\r` stdout instead of `display.update()` for pre-load messages.
2. Fixed app picker number selection using original indices instead of filtered indices. **Fix:** digits now select from current filtered list.
3. Fixed misleading `→` arrows in language/model picker implying keyboard navigation. **Fix:** replaced with `*`.
4. Added AWS SSO auth error handling — catches expired tokens, shows red error with `aws sso login` hint once, then skips all LLM calls (transcription continues).
5. Three UX branches built by subagents and merged:
   - **ux/polish:** Quick rerun prompt, exit summary box, redesigned header
   - **ux/status:** Simplified status bar (`🎧 3:45 · $0.13`), silence feedback (`⏸ waiting for speech...`), speaker colour support
   - **ux/loading:** Animated braille spinner during model load, raw-mode app picker with in-place redraw

### Dead Ends
- **Don't use `display.update()` for one-off status messages during startup.** It sets `_erasable_lines` which persists and corrupts the erase logic when the spinner or other output follows.
- **Don't ask the LLM to insert paragraph breaks.** Tried adding "Insert a blank line between distinct topics" to the prompt — LLM ignores it because the reference translation (fed back) has no breaks, so it stays consistent with that. Paragraph breaks are now done client-side in `display.py` via `_add_paragraph_breaks()` (every 3 sentences).
- **Whisper v3 hallucinates 4x more on Korean than v2.** Don't switch to v3.

### Unverified Hypotheses
- The three subagent-built UX branches were merged and parse correctly, but **have not been visually tested**. There may be rendering issues from the merge (especially `pipeline.py` which had a conflict resolved).
- Paragraph breaks every 3 sentences may feel too frequent or too mechanical with real Korean content. The interval (`PARAGRAPH_INTERVAL = 3` in display.py) may need tuning.
- The speculative text rendering fix (word-by-word fitting on the locked line) hasn't been tested with the new status bar changes.

## 4. Recommendations

### Next Steps
1. **Run the tool via tmux** against a Korean YouTube video. Watch for:
   - Does the header render correctly? (should show app name, language, Ctrl+C hint)
   - Does the loading spinner animate and then show ✓?
   - Does the status bar show `🎧 3:45 · $0.13` (simplified) or the old verbose format?
   - Does `⏸ waiting for speech...` appear during silence?
   - Are paragraph breaks in the right places?
   - Is locked (bold) vs speculative (dim italic) visually clear?
   - Any ghost lines, flickering, or misaligned text?
   - Does Ctrl+C show a summary box?
2. **Check the debug log** in a second pane — verify `Alt 1:` / `Alt 2:` hypotheses appear in LLM prompts (multi-hypothesis feature).
3. **Fix anything broken**, focusing on display/formatting.

### Watch Outs
- **`pipeline.py` is 556 lines** — slightly over the 500 LOC preference. If modifying it, consider extracting the `_build_status()` function or the hallucination detection into a small utility module.
- **Display `_erasable_lines` tracking is fragile.** Any code that writes to stdout outside of `display.update()` will desync the line counter. If you need to print anything, either go through the display or reset `_erasable_lines` after.
- **The `_pick_app` raw-mode code** (`tty.setraw`) can leave the terminal in a bad state if an exception occurs between `setraw` and the `finally` block. If terminal gets stuck, `reset` or `stty sane` fixes it.
- **Whisper timeout is 8s.** If the buffer grows large (approaching 30s), whisper can take 3-5s. The hard cap is `buffer_trimming_sec + 10s = 35s`. If you see timeouts in the log, the buffer is too large.
- The `Spinner` class uses `asyncio.create_task` — it must be started from within an async context. Don't try to use it from synchronous code.
