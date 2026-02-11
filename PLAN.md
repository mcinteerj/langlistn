# PLAN — Group C: Loading Spinner + Cleaner App Picker

## 1. Loading Spinner (`display.py` + `pipeline.py`)

### display.py — New `Spinner` class

```python
class Spinner:
    """Animated braille spinner with elapsed time. Thread-safe start/stop."""

    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    INTERVAL = 0.1  # 100ms

    def __init__(self, message: str, hint: str = ""):
        # message: e.g. "loading whisper-large-v2-mlx + VAD"
        # hint: e.g. "(downloading model ~3GB)" — shown on first run only
        self._message = message
        self._hint = hint
        self._task: asyncio.Task | None = None
        self._start_time: float = 0

    async def start(self):
        """Call from async context. Spawns display task."""
        self._start_time = time.time()
        self._task = asyncio.create_task(self._animate())

    async def _animate(self):
        idx = 0
        try:
            while True:
                elapsed = time.time() - self._start_time
                frame = self.FRAMES[idx % len(self.FRAMES)]
                parts = f"  {frame} {self._message} ({elapsed:.1f}s)"
                if self._hint:
                    parts += f"  {DIM}{self._hint}{RESET}"
                sys.stdout.write(f"\r{CLEAR_LINE}{parts}")
                sys.stdout.flush()
                idx += 1
                await asyncio.sleep(self.INTERVAL)
        except asyncio.CancelledError:
            # Final line: replace spinner with ✓
            elapsed = time.time() - self._start_time
            sys.stdout.write(f"\r{CLEAR_LINE}  ✓ {self._message} ({elapsed:.1f}s)\n")
            sys.stdout.flush()

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
```

**Imports needed in display.py:** `asyncio`, `time`

### pipeline.py — Use Spinner during model load

Replace the static `display.update("", "", loading_msg + "...")` block (lines ~184-204) with:

```python
# Detect first-run: check if model files exist locally
from .config import recommend_model
import huggingface_hub
model_id = model_name  # e.g. "mlx-community/whisper-large-v2-mlx"
hint = ""
try:
    # If snapshot_download would need to fetch, it's a first run
    local_path = huggingface_hub.try_to_load_from_cache(model_id, "config.json")
    if local_path is None:
        hint = "(downloading model ~3GB — first run only)"
except Exception:
    pass

loading_msg = f"loading {model_name} + VAD"
if speaker_tracker:
    loading_msg += " + diarization"

spinner = Spinner(loading_msg, hint=hint)
await spinner.start()

# ... existing threading.Thread load logic stays the same ...

await load_done.wait()
await spinner.stop()

if load_error:
    sys.stderr.write(f"\nModel load failed: {load_error}\n")
    return
```

**Import:** `from .display import Spinner`

### First-run detection detail

Use `huggingface_hub.try_to_load_from_cache(repo_id, "config.json")`. Returns `None` if not cached → show hint. If it raises or returns a path → no hint. Keep in try/except; hint is best-effort.

---

## 2. Cleaner App Picker (`__main__.py`)

### Replace `_pick_app` with ANSI in-place rewrite

Key behaviors:
- Print header line: `"  Choose app (type to filter, Enter to select):"`
- Below: up to **10** filtered items + optional `"  ... N more"` line
- Below: input prompt `"  Filter: {query}_"`
- On each keystroke: erase all printed lines (MOVE_UP + CLEAR_LINE), reprint
- Uses `sys.stdin` raw mode to capture single keystrokes (no Enter needed for filtering)

### Implementation

```python
import tty, termios

def _pick_app(apps: list[str]) -> str:
    last = _load_last_session()
    last_app = last.get("app")
    if last_app and last_app in apps:
        apps = [last_app] + [a for a in apps if a != last_app]

    query = ""
    filtered = apps
    MAX_VISIBLE = 10
    selected_idx = 0  # cursor position in filtered list

    # Save terminal state for raw mode
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    MOVE_UP = "\033[A"
    CLEAR_LINE = "\033[2K"

    def _draw(lines_to_erase: int) -> int:
        """Erase previous output, draw current state. Returns lines drawn."""
        # Erase
        for _ in range(lines_to_erase):
            sys.stdout.write(f"{MOVE_UP}{CLEAR_LINE}\r")

        shown = filtered[:MAX_VISIBLE]
        lines = 0

        # Items
        for i, app in enumerate(shown):
            marker = "→" if i == selected_idx else " "
            tag = " (last)" if app == last_app and not query else ""
            sys.stdout.write(f"  {marker} [{i+1}] {app}{tag}\n")
            lines += 1

        if not shown:
            sys.stdout.write("    No matches\n")
            lines += 1

        if len(filtered) > MAX_VISIBLE:
            sys.stdout.write(f"    ... {len(filtered) - MAX_VISIBLE} more\n")
            lines += 1

        # Prompt
        sys.stdout.write(f"  Filter: {query}█")
        sys.stdout.flush()
        lines += 1  # prompt line (no \n — cursor stays on it)

        return lines

    try:
        # Initial draw
        sys.stdout.write("\n  Choose app (type to filter, Enter to select):\n")
        drawn = _draw(0)

        tty.setraw(fd)
        while True:
            ch = sys.stdin.read(1)

            if ch == '\r' or ch == '\n':  # Enter
                # Need to move past drawn content cleanly
                break
            elif ch == '\x03':  # Ctrl-C
                raise KeyboardInterrupt
            elif ch == '\x7f' or ch == '\x08':  # Backspace
                query = query[:-1]
            elif ch == '\x1b':  # Escape sequence (arrows)
                seq = sys.stdin.read(2)
                if seq == '[A':  # Up
                    selected_idx = max(0, selected_idx - 1)
                elif seq == '[B':  # Down
                    selected_idx = min(len(filtered[:MAX_VISIBLE]) - 1, selected_idx + 1)
                continue  # redraw below
            elif ch.isprintable():
                query += ch
            else:
                continue

            # Refilter
            if ch != '\x1b':
                filtered = [a for a in apps if query.lower() in a.lower()] if query else apps
                selected_idx = 0
                if len(filtered) == 1:
                    break

            drawn = _draw(drawn)

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        # Clean exit: erase the picker, print selection
        for _ in range(drawn):
            sys.stdout.write(f"{MOVE_UP}{CLEAR_LINE}\r")
        sys.stdout.flush()

    if filtered:
        choice = filtered[min(selected_idx, len(filtered) - 1)]
        sys.stdout.write(f"  ✓ {choice}\n")
        sys.stdout.flush()
        return choice

    # Fallback — shouldn't reach
    return apps[0]
```

### Edge cases

- **`drawn` tracking**: `_draw` returns count of lines written (items + overflow + prompt). The prompt line counts as 1 even though no trailing `\n` — the next `_draw` call's MOVE_UP will handle it because cursor is on that line.
- **Raw mode restore**: `finally` block ensures terminal is always restored, even on Ctrl-C.
- **No stdin.fileno()** (piped input): wrap the raw-mode block in try/except `OSError` and fall back to the current simple picker. This keeps `--app` CLI path working.
- **Arrow keys**: Up/Down move `selected_idx` within visible items. Enter selects at cursor.

### Fallback guard

```python
try:
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
except (OSError, termios.error):
    # Non-interactive — fall back to simple input() picker
    return _pick_app_simple(apps)  # extract current logic into this
```

---

## File change summary

| File | Changes |
|---|---|
| `display.py` | Add `Spinner` class (~40 LOC). Add `import asyncio, time` |
| `pipeline.py` | Replace static loading msg with `Spinner` usage. Add first-run detection via `huggingface_hub.try_to_load_from_cache`. Import `Spinner` |
| `__main__.py` | Rewrite `_pick_app` with raw-mode ANSI picker. Add `import tty, termios`. Extract old logic to `_pick_app_simple` as fallback |

## Testing

- `langlistn` — interactive: verify spinner animates during load, shows elapsed, hint on first run
- `langlistn` — interactive: verify app picker redraws in-place, arrow keys work, backspace filters, Enter selects, Ctrl-C exits clean
- `langlistn --app "Chrome" --source ko` — verify no regression (bypasses picker entirely)
- Pipe test: `echo "" | langlistn` — verify graceful fallback (no raw-mode crash)
