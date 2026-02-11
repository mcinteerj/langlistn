"""Plain terminal output with ANSI overwriting for live translation."""

import asyncio
import os
import shutil
import signal
import sys
import time
from dataclasses import dataclass, field

from .audio import AppCapture, AudioSource
from .audio.mic_capture import MicCapture
from .config import resolve_language_name
from .whisper_local import LocalWhisperSession, _is_hallucination, _diff_suffix

# ANSI escape helpers
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
CLEAR_LINE = "\033[2K"
MOVE_UP = "\033[A"
CYAN = "\033[36m"
YELLOW = "\033[33m"
GRAY = "\033[90m"


@dataclass
class DisplayState:
    """Tracks what's on screen for overwriting."""
    confirmed_lines: list[str] = field(default_factory=list)
    speculative_lines: list[str] = field(default_factory=list)
    speculative_line_count: int = 0  # how many speculative lines currently printed
    status: str = ""
    total_confirmed_text: str = ""


def _wrap_text(text: str, width: int) -> list[str]:
    """Word-wrap text to terminal width."""
    if not text:
        return []
    words = text.split()
    lines = []
    current = ""
    for word in words:
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        lines.append(current)
    return lines


class TerminalDisplay:
    """Manages live terminal output with overwriting speculative text."""

    def __init__(self, term_width: int | None = None):
        self.width = term_width or shutil.get_terminal_size().columns
        self.state = DisplayState()
        self._lock = asyncio.Lock()

    def _erase_speculative(self):
        """Move cursor up and clear speculative lines."""
        for _ in range(self.state.speculative_line_count):
            sys.stdout.write(f"{MOVE_UP}{CLEAR_LINE}\r")
        # Also clear the status line
        sys.stdout.write(f"{MOVE_UP}{CLEAR_LINE}\r")
        self.state.speculative_line_count = 0

    def _print_status(self, status: str):
        """Print status bar at bottom."""
        self.state.status = status
        truncated = status[:self.width]
        sys.stdout.write(f"{GRAY}{truncated}{RESET}\n")

    async def update(self, confirmed: str, speculative: str, status: str):
        """Redraw the display with confirmed and speculative text."""
        async with self._lock:
            # Erase previous speculative + status
            if self.state.speculative_line_count > 0 or self.state.status:
                self._erase_speculative()

            # Check if we have new confirmed text
            new_confirmed = ""
            if confirmed and confirmed != self.state.total_confirmed_text:
                # Find what's new
                if confirmed.startswith(self.state.total_confirmed_text):
                    new_confirmed = confirmed[len(self.state.total_confirmed_text):].strip()
                else:
                    new_confirmed = _diff_suffix(self.state.total_confirmed_text, confirmed)
                self.state.total_confirmed_text = confirmed

            # Print new confirmed lines (permanent, scroll up)
            if new_confirmed:
                conf_lines = _wrap_text(new_confirmed, self.width - 2)
                for line in conf_lines:
                    sys.stdout.write(f"{BOLD}{line}{RESET}\n")

            # Print speculative lines (will be overwritten next update)
            spec_lines = _wrap_text(speculative, self.width - 2)
            for line in spec_lines:
                sys.stdout.write(f"{DIM}{line}{RESET}\n")
            self.state.speculative_line_count = len(spec_lines)

            # Print status
            self._print_status(status)

            sys.stdout.flush()

    async def print_final(self, text: str):
        """Print confirmed text (no overwriting)."""
        async with self._lock:
            if self.state.speculative_line_count > 0 or self.state.status:
                self._erase_speculative()
            lines = _wrap_text(text, self.width - 2)
            for line in lines:
                sys.stdout.write(f"{BOLD}{line}{RESET}\n")
            self.state.speculative_line_count = 0
            sys.stdout.flush()

    def print_header(self, mode: str, lang: str | None):
        """Print startup header."""
        lang_display = f"{lang} → English" if lang else "auto-detect → English"
        sys.stdout.write(f"\n{CYAN}langlistn{RESET} — {lang_display} — {mode}\n")
        sys.stdout.write(f"{GRAY}{'─' * self.width}{RESET}\n\n")
        sys.stdout.write(f"{GRAY}waiting for audio...{RESET}\n")
        sys.stdout.flush()


async def run_cli(
    app_name: str | None = None,
    mic: bool = False,
    device: str | None = None,
    lang: str | None = None,
    model: str = "mlx-community/whisper-large-v3-mlx",
    log_path: str | None = None,
) -> None:
    """Run langlistn with plain terminal output."""

    lang_name = resolve_language_name(lang)

    # Audio source
    source: AudioSource
    if app_name:
        source = AppCapture(app_name)
        mode = f"app: {app_name}"
    elif mic:
        source = MicCapture(device=device)
        mode = "mic"
    else:
        raise ValueError("Must specify --app or --mic")

    display = TerminalDisplay()
    display.print_header(mode, lang_name or lang)

    # Log file
    log_file = None
    if log_path:
        try:
            log_file = open(log_path, "a", encoding="utf-8")
        except OSError as e:
            print(f"{YELLOW}Warning: could not open log file: {e}{RESET}")

    # Start audio capture FIRST (before model load)
    try:
        await source.start()
    except Exception as e:
        print(f"\n{YELLOW}Audio capture failed: {e}{RESET}")
        return

    # Load model
    sys.stdout.write(f"\r{CLEAR_LINE}{GRAY}loading model...{RESET}\n")
    sys.stdout.flush()

    session = LocalWhisperSession(lang=lang, model=model)
    await session.connect()

    # Shutdown handling
    shutdown = asyncio.Event()

    def _signal_handler():
        shutdown.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    # State for confirmed/speculative tracking
    all_confirmed: list[str] = []  # All confirmed segments
    previous_full_text = ""
    prev_prev_text = ""  # Two windows ago, for confirming

    async def audio_loop():
        """Feed audio to session."""
        while not shutdown.is_set():
            chunk = await source.read_chunk()
            if chunk is None:
                break
            await session.send_audio(chunk)

    async def process_loop():
        """Process whisper windows and update display."""
        nonlocal previous_full_text, prev_prev_text

        from .whisper_local import (
            STEP_SECONDS, WINDOW_SECONDS,
            _seconds_to_bytes, SAMPLE_RATE, BYTES_PER_SAMPLE,
        )

        step_bytes = _seconds_to_bytes(STEP_SECONDS)
        window_bytes = _seconds_to_bytes(WINDOW_SECONDS)
        start_time = time.time()

        # Wait for initial buffer fill
        while not shutdown.is_set():
            await asyncio.sleep(0.5)
            if len(session._audio_buffer) >= step_bytes:
                break

        while not shutdown.is_set():
            # Grab window
            async with session._buffer_lock:
                buf_len = len(session._audio_buffer)
                if buf_len < step_bytes:
                    await asyncio.sleep(0.5)
                    continue
                take = min(buf_len, window_bytes)
                window_pcm = bytes(session._audio_buffer[-take:])

            # Run whisper
            t0 = time.time()
            try:
                result = await loop.run_in_executor(
                    None, session._transcribe, window_pcm
                )
            except Exception as e:
                await display.update("", "", f"error: {e}")
                await asyncio.sleep(STEP_SECONDS)
                continue
            elapsed = time.time() - t0

            session.stats.windows_processed += 1
            session.stats.audio_seconds_processed += len(window_pcm) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
            session.stats.processing_time_total += elapsed

            text = result.get("text", "").strip()
            if not text or _is_hallucination(text):
                await display.update(
                    " ".join(all_confirmed), "",
                    session.stats.status_line(),
                )
                await asyncio.sleep(STEP_SECONDS)
                continue

            # Confirm text that appeared in BOTH this and previous window
            # (overlap region = confirmed, new-only region = speculative)
            confirmed_new = ""
            speculative = text

            if previous_full_text:
                # Words that appear in both windows → confirmed
                prev_words = previous_full_text.split()
                curr_words = text.split()

                # Find overlap: suffix of prev matching prefix of curr
                best_overlap = 0
                max_check = min(len(prev_words), len(curr_words))
                for size in range(max_check, 0, -1):
                    if prev_words[-size:] == curr_words[:size]:
                        best_overlap = size
                        break

                if best_overlap > 0:
                    # The overlapping part is now confirmed
                    overlap_text = " ".join(curr_words[:best_overlap])

                    # But only add what's NEW confirmed (not already in all_confirmed)
                    existing = " ".join(all_confirmed)
                    new_conf = _diff_suffix(existing, overlap_text)
                    if new_conf.strip():
                        confirmed_new = new_conf.strip()
                        all_confirmed.append(confirmed_new)
                        if log_file:
                            log_file.write(confirmed_new + "\n")
                            log_file.flush()

                    # Everything after overlap is speculative
                    speculative = " ".join(curr_words[best_overlap:])
                else:
                    # No overlap — diff against previous
                    new_text = _diff_suffix(previous_full_text, text)
                    speculative = new_text if new_text.strip() else text

            prev_prev_text = previous_full_text
            previous_full_text = text

            # Update display
            runtime = int(time.time() - start_time)
            mins, secs = divmod(runtime, 60)
            status = f"{session.stats.status_line()} · {mins}:{secs:02d}"

            await display.update(
                " ".join(all_confirmed),
                speculative,
                status,
            )

            sleep_time = max(0.1, STEP_SECONDS - elapsed)
            await asyncio.sleep(sleep_time)

    # Run everything
    try:
        tasks = [
            asyncio.create_task(audio_loop()),
            asyncio.create_task(process_loop()),
            asyncio.create_task(shutdown.wait()),
        ]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
    except asyncio.CancelledError:
        pass
    finally:
        print(f"\n{GRAY}shutting down...{RESET}")
        await session.shutdown()
        await source.stop()
        if log_file:
            log_file.close()
        print(f"{GRAY}done.{RESET}\n")
