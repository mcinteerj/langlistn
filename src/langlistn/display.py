"""Terminal display with locked (bold) and speculative (dim italic) text.

All text flows continuously on the same line. Locked text is permanent,
speculative tail overwrites in-place each cycle.

Approach: we track how many lines are permanently printed. Each update we
erase the erasable zone (last locked line + speculative + status), print
any new permanent locked lines, then redraw the erasable zone.
"""

import asyncio
import re
import shutil
import sys
import time

BOLD = "\033[1m"
DIM = "\033[2m"
DIM_ITALIC = "\033[2;3m"
RESET = "\033[0m"
CLEAR_LINE = "\033[2K"
MOVE_UP = "\033[A"


_ANSI_RE = re.compile(r'\033\[[0-9;]*m')


def _visible_len(s: str) -> int:
    """Length of string excluding ANSI escape sequences."""
    return len(_ANSI_RE.sub('', s))

# Sentence-ending punctuation pattern
_SENTENCE_END = re.compile(r'([.!?。！？…]\s+)')

PARAGRAPH_INTERVAL = 3  # Insert blank line every N sentences


class Spinner:
    """Animated braille spinner with elapsed time. Thread-safe start/stop."""

    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    INTERVAL = 0.1

    def __init__(self, message: str, hint: str = ""):
        self._message = message
        self._hint = hint
        self._task: asyncio.Task | None = None
        self._start_time: float = 0

    async def start(self):
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


def _add_paragraph_breaks(text: str) -> str:
    """Insert paragraph breaks every N sentences for readability."""
    parts = _SENTENCE_END.split(text)
    # parts alternates: text, punctuation, text, punctuation, ...
    result = []
    sentence_count = 0
    for i, part in enumerate(parts):
        result.append(part)
        # Odd indices are the punctuation captures
        if i % 2 == 1:
            sentence_count += 1
            if sentence_count % PARAGRAPH_INTERVAL == 0:
                result.append("\n")
    return "".join(result)


def _wrap_lines(text: str, width: int) -> list[str]:
    """Word-wrap text, preserving newlines."""
    if not text:
        return []
    out: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            out.append("")
            continue
        words = paragraph.split()
        current = ""
        for word in words:
            if current and _visible_len(current) + 1 + _visible_len(word) > width:
                out.append(current)
                current = word
            else:
                current = f"{current} {word}" if current else word
        if current:
            out.append(current)
    return out


class TerminalDisplay:
    """Two-zone terminal display."""

    def __init__(self, plain: bool = False):
        self.width = shutil.get_terminal_size().columns
        self.plain = plain
        self._permanent_lines = 0     # lines we've permanently printed
        self._permanent_content: list[str] = []  # content of each permanent line
        self._erasable_lines = 0      # lines in the erasable zone
        self._last_locked = ""
        self._lock = __import__('threading').Lock()
        self._stopped = False

    def _erase(self):
        if self.plain or self._erasable_lines == 0:
            return
        for _ in range(self._erasable_lines):
            sys.stdout.write(f"{MOVE_UP}{CLEAR_LINE}\r")
        self._erasable_lines = 0

    def update(self, locked: str, speculative: str = "", status: str = "", force_commit: bool = False):
        with self._lock:
            self._update_inner(locked, speculative, status, force_commit)

    def _update_inner(self, locked: str, speculative: str = "", status: str = "", force_commit: bool = False):
        if self._stopped:
            return
        self._erase()

        if self.plain:
            new = locked[len(self._last_locked):]
            if new.strip():
                sys.stdout.write(new.strip() + "\n")
            sys.stdout.flush()
            self._last_locked = locked
            return

        w = self.width - 1
        locked_formatted = _add_paragraph_breaks(locked) if locked else ""
        locked_lines = _wrap_lines(locked_formatted, w) if locked_formatted else []
        total_locked = len(locked_lines)

        # Permanent zone: locked lines that are safe to commit (can't be erased).
        # Keep a buffer of recent locked lines erasable to absorb word-wrap
        # changes when the LLM revises text. Only commit lines whose content
        # matches what we previously rendered — this prevents duplicates when
        # re-wrapping shifts line boundaries.
        ERASABLE_LOCKED_BUFFER = 0 if force_commit else 3  # keep last N locked lines erasable
        new_permanent_target = max(0, total_locked - ERASABLE_LOCKED_BUFFER)

        # Only advance permanent lines if locked text actually grew.
        # This prevents re-committing the same lines when different callers
        # (whisper_loop vs translate_loop) pass slightly different locked text.
        safe_target = self._permanent_lines
        if total_locked > len(self._permanent_content):
            for i in range(self._permanent_lines, new_permanent_target):
                if i < len(self._permanent_content) and \
                   self._permanent_content[i] == locked_lines[i]:
                    safe_target = i + 1
                elif i >= len(self._permanent_content):
                    safe_target = i + 1
                else:
                    break

        for i in range(self._permanent_lines, safe_target):
            sys.stdout.write(f"{BOLD}{locked_lines[i]}{RESET}\n")
        self._permanent_lines = safe_target
        # Record what locked lines look like for next comparison
        self._permanent_content = list(locked_lines[:total_locked])

        # Build erasable zone: buffered locked lines + speculative + status
        erasable_count = 0

        # Render non-permanent locked lines (erasable buffer)
        erasable_locked = locked_lines[self._permanent_lines:]
        if erasable_locked:
            # All but last erasable locked line
            for line in erasable_locked[:-1]:
                sys.stdout.write(f"{BOLD}{line}{RESET}\n")
                erasable_count += 1

            last_locked_line = erasable_locked[-1]
            if speculative.strip():
                # Render last locked line bold, then speculative continues
                # on the same line in dim italic
                remaining_width = w - _visible_len(last_locked_line) - 1
                spec_text = speculative.strip()

                if remaining_width > 5:
                    # Split speculative: first chunk fits on the locked line
                    spec_words = spec_text.split()
                    same_line = ""
                    rest_words = []
                    for i, word in enumerate(spec_words):
                        if same_line and _visible_len(same_line) + 1 + _visible_len(word) > remaining_width:
                            rest_words = spec_words[i:]
                            break
                        same_line = f"{same_line} {word}" if same_line else word
                    else:
                        rest_words = []

                    sys.stdout.write(f"{BOLD}{last_locked_line}{RESET}{DIM_ITALIC} {same_line}{RESET}\n")
                    erasable_count += 1

                    if rest_words:
                        overflow = " ".join(rest_words)
                        for line in _wrap_lines(overflow, w)[:5]:
                            sys.stdout.write(f"{DIM_ITALIC}{line}{RESET}\n")
                            erasable_count += 1
                else:
                    # Not enough room — locked line on its own, spec on next
                    sys.stdout.write(f"{BOLD}{last_locked_line}{RESET}\n")
                    erasable_count += 1
                    for line in _wrap_lines(spec_text, w)[:5]:
                        sys.stdout.write(f"{DIM_ITALIC}{line}{RESET}\n")
                        erasable_count += 1
            else:
                sys.stdout.write(f"{BOLD}{last_locked_line}{RESET}\n")
                erasable_count += 1
        elif speculative.strip():
            for line in _wrap_lines(speculative.strip(), w)[:6]:
                sys.stdout.write(f"{DIM_ITALIC}{line}{RESET}\n")
                erasable_count += 1

        if status:
            sys.stdout.write(f"{DIM}{status[:w]}{RESET}\n")
            erasable_count += 1

        self._erasable_lines = erasable_count
        self._last_locked = locked
        sys.stdout.flush()

    def print_header(self, mode: str, lang: str | None, model: str, translate_model: str | None = None):
        if self.plain:
            return
        w = self.width
        lang_display = f"{lang} > English" if lang else "auto > English"
        model_short = model.split("/")[-1] if "/" in model else model
        t_tag = f" / {translate_model}" if translate_model else ""
        info = f"{lang_display} / {mode} / {model_short}{t_tag}"
        hint = "Ctrl+C to stop"

        # Top border
        title = " langlistn "
        pad = w - len(title) - 3
        sys.stdout.write(f"\n {DIM}┌{RESET}{BOLD}{title}{RESET}{DIM}{'─' * max(pad, 0)}┐{RESET}\n")
        # Info line: " │ {info padded} · {hint} │"
        # Must be same visible width as borders: w chars total
        # " │" = 2, "│" = 1 → inner = w - 3 chars between the │ bars
        inner_w = w - 3
        if w >= 50:
            # " {info...} · {hint} " — with leading and trailing space
            right = f" | {hint} "
            left_w = inner_w - len(right)
            left = f" {info[:left_w - 1].ljust(left_w - 1)}"
            sys.stdout.write(f" {DIM}│{RESET}{left}{DIM}{right}│{RESET}\n")
        else:
            sys.stdout.write(f" {DIM}│{RESET} {info[:inner_w - 2].ljust(inner_w - 2)} {DIM}│{RESET}\n")
        # Bottom border
        sys.stdout.write(f" {DIM}└{'─' * (w - 3)}┘{RESET}\n\n")
        sys.stdout.flush()

    def finish(self, *, duration: float = 0, words: int = 0, sentences: int = 0,
               llm_calls: int = 0, cost: float = 0.0, has_content: bool = False):
        with self._lock:
            # Commit any remaining buffered lines before shutting down
            if self._last_locked:
                self._update_inner(self._last_locked, "", "", force_commit=True)
            self._stopped = True
            self._erase()
        if self.plain:
            return
        inner = min(36, self.width - 6)  # content width inside box
        # Box total visible width = inner + 6 (│ + 2 padding each side + │)

        def _row(label: str, value: str) -> str:
            content = f"{label}{value}"
            return f" {DIM}│{RESET}  {content}{' ' * max(0, inner - len(content))}  {DIM}│{RESET}\n"

        mins, secs = int(duration) // 60, int(duration) % 60
        dur_str = f"{mins}:{secs:02d}"
        cost_str = f"${cost:.3f}" if cost > 0 else ""
        calls_val = f"{llm_calls} · {cost_str}" if cost_str else str(llm_calls)

        title = " session summary "
        top_dashes = inner + 4 - len(title) - 1  # +4 for padding, -1 for leading ─
        sys.stdout.write(f"\n {DIM}╭─{RESET}{BOLD}{title}{RESET}{DIM}{'─' * max(0, top_dashes)}╮{RESET}\n")
        sys.stdout.write(_row("Duration     ", dur_str))
        if words:
            sys.stdout.write(_row("Words        ", str(words)))
        if sentences:
            sys.stdout.write(_row("Sentences    ", str(sentences)))
        if llm_calls:
            sys.stdout.write(_row("LLM calls    ", calls_val))
        sys.stdout.write(f" {DIM}╰{'─' * (inner + 4)}╯{RESET}\n\n")
        sys.stdout.flush()
