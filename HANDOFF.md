# Handoff Notes

## 1. The Objective

**Goal:** The display is broken — user sees only a growing stack of status bar lines (`🎧 0:01`, `🎧 0:04`, etc.) with no transcription or translation text. Diagnose and fix so that locked (bold) and speculative (dim italic) text renders correctly above the status bar.

**Definition of Done:**
- Running `langlistn --app "Google Chrome" --source ko --debug-log /tmp/langlistn-debug.log` against a Korean YouTube video shows Korean transcription AND English translation flowing in the terminal
- Status bar updates in-place (single line, not stacking)
- Bold locked text stays permanent, dim italic speculative text overwrites in-place
- Exit summary renders cleanly on Ctrl+C

## 2. State & Resources

### Current Status

**Working:**
- Whisper transcription: confirmed working. Debug logs show Korean text being transcribed with ~1.5s latency, LocalAgreement-2 confirming text correctly
- Silero VAD: working, filtering silence
- Audio capture: working (ScreenCaptureKit Swift helper)
- App picker with type-to-filter: working
- Loading spinner: working

**Broken:**
- **Display rendering**: status bar lines stack vertically instead of overwriting. No transcription/translation text visible to user. The `display.update()` erase-and-redraw mechanism is not functioning — `_erasable_lines` tracking is likely out of sync with actual terminal state
- **Translation**: was failing with `Token has expired and refresh failed` (AWS SSO). This is an auth issue, not a code bug. User needs to run `aws sso login` before testing translation. **However**, even without translation, Korean transcription text should still appear (it doesn't)

**Pending (not broken, just not tested yet):**
- Speaker diarization (`--diarize`)
- Exit summary box
- Quick rerun (`_quick_rerun_prompt`)
- Multi-hypothesis LLM feeding (needs working translation to test)

### Key Assets

| Asset | Path |
|-------|------|
| Project root | `/Users/jakemc/projects/langlistn/` |
| Venv | `.venv/bin/python` |
| Display renderer | `src/langlistn/display.py` (180 lines) |
| Pipeline (builds status string, calls display) | `src/langlistn/pipeline.py` (553 lines) |
| CLI entry | `src/langlistn/__main__.py` (487 lines) |
| Translation | `src/langlistn/translate.py` |
| Whisper ASR | `src/langlistn/streaming_asr.py` |
| Debug log from test | `/tmp/langlistn-debug.log` |
| Raw output from test | `/tmp/langlistn-output.log` |
| Swift helper binary | `swift/.build/release/AudioCaptureHelper` |

### Test Commands

```bash
# Ensure AWS auth (needed for translation)
aws sso login

# Install
cd ~/projects/langlistn && .venv/bin/pip install -e .

# Run with debug logging (Korean YouTube video must be playing in Chrome)
.venv/bin/langlistn --app "Google Chrome" --source ko --debug-log /tmp/langlistn-debug.log

# Run via tmux for non-interactive testing
tmux -S /tmp/openclaw-tmux-sockets/openclaw.sock new-session -d -s main 2>/dev/null || true
tmux -S /tmp/openclaw-tmux-sockets/openclaw.sock new-window -d -n ll -- \
  bash -c 'cd ~/projects/langlistn && .venv/bin/langlistn --app "Google Chrome" --source ko --debug-log /tmp/langlistn-debug.log 2>&1 | tee /tmp/langlistn-output.log'

# Check after ~15s
tail -50 /tmp/langlistn-debug.log   # whisper + translation activity
tail -30 /tmp/langlistn-output.log  # raw terminal output with ANSI codes

# Kill test
tmux -S /tmp/openclaw-tmux-sockets/openclaw.sock send-keys -t ll C-c

# Transcribe-only mode (no AWS needed — good for isolating display bugs)
.venv/bin/langlistn --app "Google Chrome" --source ko --no-translate --debug-log /tmp/langlistn-debug.log
```

## 3. Knowledge Transfer

### What was tried (this session)
1. Rewrote entire project from scratch — two-loop pipeline (whisper + claude translation)
2. Built `display.py` with two-zone approach: permanent lines (bold, never erased) + erasable zone (last locked line + speculative + status)
3. Fixed display flicker bug where combined text re-wrapping lost characters
4. Added paragraph breaks (every 3 sentences), multi-hypothesis LLM feeding, 25s buffer, speaker diarization
5. Three parallel Bob builds for UX polish (header, status bar, spinner, app picker)
6. Fixed "starting audio capture" stuck issue — changed from `display.update()` to raw stdout `\r`
7. Fixed app picker number selection against filtered list

### Dead Ends
- **Using `display.update()` for pre-loading messages**: it sets `_erasable_lines` which gets out of sync with spinner. Fixed to use raw stdout instead.
- **Combined text re-wrapping for locked+speculative on same line**: caused char loss. Replaced with word-level splitting that keeps locked text exact.
- **`→` arrow marker in `_pick()`**: users expected arrow key navigation. Changed to `*`.

### Hypotheses (UNVERIFIED)

The stacking status bar issue is likely one of:

1. **`_erasable_lines` count mismatch**: The Bob UX builds may have introduced extra writes (speaker colours, silence feedback) that print lines without incrementing `_erasable_lines`. The `_erase()` method then doesn't move up enough lines, so each update appends below instead of overwriting.

2. **`_build_status()` return value contains newlines or extra content**: The status string might be multi-line or the speaker colour logic might inject extra `\n`s.

3. **Pipeline not passing text to display**: Looking at the debug log, whisper IS producing text (`confirmed` and `speculative` are populated), but the pipeline's translation failure path may be returning empty strings for `locked`/`spec`, so `display.update("", "", status)` is called with empty text every cycle — meaning only the status bar renders.

4. **The three Bob merges conflicted subtly**: `pipeline.py` had a merge conflict that was resolved. The `_build_status()` function, speaker colour handling, and display calls may not be integrated correctly after the three-way merge.

## 4. Recommendations

### Next Steps (options)

**Option A — Quick diagnosis via debug log enhancement:**
Add a `logger.debug("DISPLAY locked=%r spec=%r status=%r", locked, spec, status)` right before `display.update()` in pipeline.py. Re-run. This tells you immediately whether the pipeline is passing empty text or the display is eating it.

**Option B — Focus on the no-translate path first:**
Run with `--no-translate` to remove the AWS auth variable. In this mode, `locked` = confirmed Korean source, `spec` = speculative Korean. If text still doesn't show, the bug is in display.py. If it does show, the bug is in the translation failure handling path.

**Option C — Audit the merged pipeline.py:**
The three Bob builds all modified `pipeline.py`. Check that the section building `locked`/`spec` variables and calling `display.update(locked, spec, status)` is correct after the merges. Pay special attention to the `_build_status()` integration and where `display.update` is called vs where `status` is built.

### Watch Outs
- **AWS SSO auth**: User needs `aws sso login` before translation works. Without it, every LLM call fails silently and returns the last confirmed/speculative (which starts empty)
- **`pipeline.py` is 553 lines** — over the 500 line guideline. Consider extracting `_build_status()` and display logic
- **Three Bob builds modified overlapping files** — the merge was mechanical. A careful read of the merged `pipeline.py` and `display.py` is warranted
- **Don't force push or reset --hard** without explicit approval
- **Test with `--no-translate` first** to isolate display vs translation issues
