# Group B: Status Bar + Silence Feedback + Speaker Colours

Branch: `ux/status` · Worktree: `langlistn-ux-status`

---

## 1. Simplified Status Bar

**Goal**: `🎧 3:45 · $0.13` in normal mode; verbose stats → debug log only.

### pipeline.py — `transcribe_and_translate()` status block (~line 270-280)

Replace the status string construction:

```python
# BEFORE
status = (
    f"{speaker_label}listening · windows: {windows_processed} · "
    f"avg: {avg_ms:.0f}ms · buf: {buf_seconds:.0f}s · "
    f"{runtime // 60}:{runtime % 60:02d} · {cost_str}"
)

# AFTER
logger.debug(
    "stats | windows=%d avg=%.0fms buf=%.0fs",
    windows_processed, avg_ms, buf_seconds,
)
runtime_str = f"{runtime // 60}:{runtime % 60:02d}"
cost_display = f"${translator.estimated_cost():.2f}" if translator else ""
status = f"🎧 {runtime_str} · {cost_display}".rstrip(" ·")
```

- Cost rounds to 2 decimal places (not 3) — cleaner
- `speaker_label` removed from status (moved to inline text — see §3)
- Verbose stats go to `logger.debug` so `--debug` flag still shows them

---

## 2. Silence / Listening Feedback

**Goal**: When silence > 2s, status shows `⏸ waiting for speech...` instead of frozen `🎧`. Speech resumption flips back.

### pipeline.py changes

**A) Add silence state to status block** (same area, after building `runtime_str`/`cost_display`):

```python
silence_duration = time.time() - last_speech_time  # already computed above

if silence_duration > 2.0:
    status = f"⏸  waiting for speech... · {runtime_str} · {cost_display}".rstrip(" ·")
else:
    status = f"🎧 {runtime_str} · {cost_display}".rstrip(" ·")
```

Note: `silence_duration` is already calculated earlier in the loop (~line 207). Reuse that value — don't recompute. Move or hoist the variable so it's available at status-build time. Currently it's inside an `if` block that only fires when `silence_duration > cfg.silence_reset_seconds`. Fix:

- Move `silence_duration = time.time() - last_speech_time` to just after `await asyncio.sleep(cfg.process_interval)` (before the `buf_seconds` line), so it's always available.
- Keep the existing `> cfg.silence_reset_seconds` reset logic where it is — it still references the same variable.

**B) Early-cycle status update**: Currently display only updates after whisper processes. During long silence, the `continue` at line ~220 skips the display update entirely. Fix: before every `continue` in the silence/buffer-too-small early exits, update the display with the silence status:

```python
if silence_duration > cfg.silence_reset_seconds:
    # ... existing reset logic ...
    if buf_seconds < cfg.min_audio_seconds:
        runtime = int(time.time() - start_time)
        runtime_str = f"{runtime // 60}:{runtime % 60:02d}"
        cost_display = f"${translator.estimated_cost():.2f}" if translator else ""
        display.update(locked_text, "", f"⏸  waiting for speech... · {runtime_str} · {cost_display}".rstrip(" ·"))
        continue
```

Where `locked_text` = `translator.confirmed_translation` if translator else `confirmed_source`.

Extract a helper to avoid duplication:

```python
def _build_status() -> str:
    runtime = int(time.time() - start_time)
    runtime_str = f"{runtime // 60}:{runtime % 60:02d}"
    cost_display = f"${translator.estimated_cost():.2f}" if translator else ""
    silence_dur = time.time() - last_speech_time
    if silence_dur > 2.0:
        return f"⏸  waiting for speech... · {runtime_str} · {cost_display}".rstrip(" ·")
    return f"🎧 {runtime_str} · {cost_display}".rstrip(" ·")
```

Define this as a closure inside `transcribe_and_translate()` (it captures `start_time`, `last_speech_time`, `translator`). Use it everywhere status is needed.

---

## 3. Speaker Colours in Text

**Goal**: When `--diarize`, insert coloured `\n[Speaker A] ` into locked text on speaker change. Up to 6 colours.

### display.py changes

**A) Add speaker colour constants:**

```python
SPEAKER_COLOURS = [
    "\033[36m",  # cyan
    "\033[33m",  # yellow
    "\033[32m",  # green
    "\033[35m",  # magenta
    "\033[34m",  # blue
    "\033[91m",  # bright red
]
```

No display.py logic changes needed beyond this — the speaker tags will be embedded in the locked text string itself (with ANSI codes), and `_wrap_lines` / the renderer already pass them through. The `BOLD` wrapper on locked lines will combine with the colour codes fine.

**B) However**: `_wrap_lines` uses `len()` for width calc, which counts ANSI escape chars. Fix `_wrap_lines` to strip ANSI when measuring width:

```python
_ANSI_RE = re.compile(r'\033\[[0-9;]*m')

def _visible_len(s: str) -> int:
    return len(_ANSI_RE.sub('', s))
```

Update `_wrap_lines` to use `_visible_len()` instead of `len()` for width checks.

### pipeline.py changes

**A) Track previous speaker** — add state var:

```python
prev_speaker: str | None = None
```

**B) Build a speaker colour map** — add near speaker_tracker init:

```python
from .display import SPEAKER_COLOURS
speaker_colour_map: dict[str, str] = {}  # speaker name → ANSI colour
```

**C) Inject speaker tags into locked text** — in the diarization block (~line 262):

```python
# ── diarization ──
if speaker_tracker and speaker_tracker.available:
    speaker_tracker.process()
    cur = speaker_tracker.current_speaker()
    if cur and cur != prev_speaker and confirmed:
        # Assign colour
        if cur not in speaker_colour_map:
            idx = len(speaker_colour_map) % len(SPEAKER_COLOURS)
            speaker_colour_map[cur] = SPEAKER_COLOURS[idx]
        colour = speaker_colour_map[cur]
        # Inject speaker tag into confirmed_source (the locked text source)
        tag = f"\n{colour}[{cur}]\033[0m "
        # Insert before the latest confirmed chunk
        # confirmed_source already has the new confirmed appended above
        # We need to insert the tag before the last addition
        # Approach: track where we appended and insert tag there
        confirmed_source = confirmed_source.rstrip()
        confirmed_source += tag
        prev_speaker = cur
    elif cur:
        prev_speaker = cur  # track even if no confirmed text yet
```

Wait — this has a timing issue. The confirmed text is appended *before* the diarization block. Restructure:

**Better approach**: Insert the tag *when appending confirmed text*, not after. Move diarization check to just before the `confirmed_source += confirmed` line:

```python
if confirmed:
    # Speaker change detection (before appending)
    if speaker_tracker and speaker_tracker.available:
        speaker_tracker.process()
        cur = speaker_tracker.current_speaker()
        if cur and cur != prev_speaker:
            if cur not in speaker_colour_map:
                idx = len(speaker_colour_map) % len(SPEAKER_COLOURS)
                speaker_colour_map[cur] = SPEAKER_COLOURS[idx]
            colour = speaker_colour_map[cur]
            tag = f"\n{colour}[{cur}]\033[0m "
            confirmed_source += tag
            prev_speaker = cur

    confirmed_source += (" " + confirmed if confirmed_source else confirmed)
```

Move the existing `speaker_tracker.process()` call from its current location (line ~262) into this block. Keep a second `speaker_tracker.process()` call in the original spot for the case when there's no confirmed text but we still want to keep diarization warm:

```python
# ── diarization (keep model warm even without confirmed text) ──
if speaker_tracker and speaker_tracker.available and not confirmed:
    speaker_tracker.process()
```

**D) Remove old speaker_label from status string** — it's now inline. Delete the block at ~line 260-264 that builds `speaker_label`.

### diarize.py

No changes needed. `current_speaker()` and `_map_speaker()` already produce stable `Speaker A/B/C` labels.

---

## File change summary

| File | Changes |
|------|---------|
| `display.py` | Add `SPEAKER_COLOURS`, `_ANSI_RE`, `_visible_len()`. Fix `_wrap_lines` width calc. |
| `pipeline.py` | Extract `_build_status()` closure. Simplify status string. Add silence feedback. Move diarization to pre-append. Inject coloured speaker tags. Remove `speaker_label` from status. |
| `diarize.py` | No changes. |

## Testing notes

- Test without `--diarize`: should show simplified status, silence feedback
- Test with `--diarize`: speaker tags in text, colours cycle
- Test `--plain` mode: ANSI codes won't render — consider stripping speaker colour codes in plain mode (add `if not self.plain` guard, or strip in `update()` when `self.plain`)
- Test long sessions: verify `_visible_len` doesn't regress wrap performance
