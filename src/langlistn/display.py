"""Terminal display with locked (bold) and speculative (dim italic) text.

All text flows continuously on the same line. Locked text is permanent,
speculative tail overwrites in-place each cycle.

Approach: we track how many lines are permanently printed. Each update we
erase the erasable zone (last locked line + speculative + status), print
any new permanent locked lines, then redraw the erasable zone.
"""

import re
import shutil
import sys

BOLD = "\033[1m"
DIM = "\033[2m"
DIM_ITALIC = "\033[2;3m"
RESET = "\033[0m"
CLEAR_LINE = "\033[2K"
MOVE_UP = "\033[A"

# Sentence-ending punctuation pattern
_SENTENCE_END = re.compile(r'([.!?。！？…])\s+')

PARAGRAPH_INTERVAL = 3  # Insert blank line every N sentences


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
            if current and len(current) + 1 + len(word) > width:
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
        self._erasable_lines = 0      # lines in the erasable zone
        self._last_locked = ""

    def _erase(self):
        if self.plain or self._erasable_lines == 0:
            return
        for _ in range(self._erasable_lines):
            sys.stdout.write(f"{MOVE_UP}{CLEAR_LINE}\r")
        self._erasable_lines = 0

    def update(self, locked: str, speculative: str = "", status: str = ""):
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

        # Permanent zone: all locked lines except the last (which is erasable
        # so speculative can continue on the same line)
        new_permanent_target = max(0, total_locked - 1)

        # Print any new permanent lines
        for i in range(self._permanent_lines, new_permanent_target):
            sys.stdout.write(f"{BOLD}{locked_lines[i]}{RESET}\n")
        self._permanent_lines = new_permanent_target

        # Build erasable zone: last locked line + speculative + status
        erasable_count = 0

        if locked_lines:
            last_locked_line = locked_lines[-1]
            if speculative.strip():
                # Render last locked line bold, then speculative continues
                # on the same line in dim italic
                remaining_width = w - len(last_locked_line) - 1
                spec_text = speculative.strip()

                if remaining_width > 5:
                    # Split speculative: first chunk fits on the locked line
                    spec_words = spec_text.split()
                    same_line = ""
                    rest_words = []
                    for i, word in enumerate(spec_words):
                        if same_line and len(same_line) + 1 + len(word) > remaining_width:
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
        lang_display = f"{lang} → English" if lang else "auto-detect → English"
        sep = "─" * self.width
        model_short = model.split("/")[-1] if "/" in model else model
        t_info = f" + Claude {translate_model}" if translate_model else ""
        sys.stdout.write(
            f"\n{BOLD}langlistn{RESET} — {lang_display} — {mode} — {model_short}{t_info}\n"
        )
        sys.stdout.write(f"{DIM}{sep}{RESET}\n\n")
        sys.stdout.flush()

    def finish(self, cost: float = 0.0):
        self._erase()
        if not self.plain:
            sys.stdout.write(f"\n{DIM}done. · ${cost:.3f}{RESET}\n\n")
            sys.stdout.flush()
