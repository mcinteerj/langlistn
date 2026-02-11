"""Plain terminal output with ANSI overwriting for live translation."""

import asyncio
import os
import shutil
import signal
import sys
import time

from .audio import AppCapture, AudioSource
from .audio.mic_capture import MicCapture
from .config import resolve_language_name
from .whisper_local import (
    LocalWhisperSession, _is_hallucination, _diff_suffix,
    STEP_SECONDS, WINDOW_SECONDS, _seconds_to_bytes,
    SAMPLE_RATE, BYTES_PER_SAMPLE,
)

# ANSI — only use dim/bold/reset so terminal theme colors are respected
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
CLEAR_LINE = "\033[2K"
MOVE_UP = "\033[A"


def _wrap_text(text: str, width: int) -> list[str]:
    """Word-wrap text to terminal width."""
    if not text.strip():
        return []
    words = text.split()
    lines: list[str] = []
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
    """Live terminal output — confirmed text scrolls up, speculative gets overwritten."""

    def __init__(self):
        self.width = shutil.get_terminal_size().columns
        self._spec_lines_on_screen = 0  # includes status line

    def _erase_speculative(self):
        """Erase speculative lines + status line from screen."""
        for _ in range(self._spec_lines_on_screen):
            sys.stdout.write(f"{MOVE_UP}{CLEAR_LINE}\r")
        self._spec_lines_on_screen = 0

    def print_confirmed(self, text: str):
        """Print confirmed text permanently. Erases speculative first."""
        if not text.strip():
            return
        self._erase_speculative()
        lines = _wrap_text(text, self.width - 1)
        for line in lines:
            sys.stdout.write(f"{BOLD}{line}{RESET}\n")
        sys.stdout.flush()

    def print_speculative(self, text: str, status: str):
        """Print speculative text + status (will be overwritten next call)."""
        self._erase_speculative()
        count = 0
        if text.strip():
            lines = _wrap_text(text, self.width - 1)
            for line in lines:
                sys.stdout.write(f"{DIM}{line}{RESET}\n")
            count += len(lines)
        # Status line
        status_trunc = status[:self.width - 1]
        sys.stdout.write(f"{DIM}{status_trunc}{RESET}\n")
        count += 1
        self._spec_lines_on_screen = count
        sys.stdout.flush()

    def print_header(self, mode: str, lang: str | None):
        lang_display = f"{lang} → English" if lang else "auto-detect → English"
        sep = "─" * self.width
        sys.stdout.write(f"\n{BOLD}langlistn{RESET} — {lang_display} — {mode}\n")
        sys.stdout.write(f"{DIM}{sep}{RESET}\n\n")
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
            print(f"Warning: could not open log file: {e}")

    # Start audio
    display.print_speculative("", "starting audio capture...")
    try:
        await source.start()
    except Exception as e:
        print(f"\nAudio capture failed: {e}")
        return

    display.print_speculative("", "loading model...")

    session = LocalWhisperSession(lang=lang, model=model)
    await session.connect()

    # Shutdown handling
    shutdown = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown.set)

    # Translation state
    previous_text = ""      # Full text from previous window
    confirmed_so_far = ""   # All confirmed text concatenated

    async def audio_loop():
        while not shutdown.is_set():
            chunk = await source.read_chunk()
            if chunk is None:
                break
            await session.send_audio(chunk)

    async def process_loop():
        nonlocal previous_text, confirmed_so_far

        step_bytes = _seconds_to_bytes(STEP_SECONDS)
        window_bytes = _seconds_to_bytes(WINDOW_SECONDS)
        start_time = time.time()

        # Wait for buffer
        while not shutdown.is_set():
            await asyncio.sleep(0.5)
            if len(session._audio_buffer) >= step_bytes:
                break

        display.print_speculative("", "listening...")

        while not shutdown.is_set():
            async with session._buffer_lock:
                buf_len = len(session._audio_buffer)
                if buf_len < step_bytes:
                    await asyncio.sleep(0.5)
                    continue
                take = min(buf_len, window_bytes)
                window_pcm = bytes(session._audio_buffer[-take:])

            t0 = time.time()
            try:
                result = await loop.run_in_executor(
                    None, session._transcribe, window_pcm
                )
            except Exception as e:
                display.print_speculative("", f"error: {e}")
                await asyncio.sleep(STEP_SECONDS)
                continue
            elapsed = time.time() - t0

            session.stats.windows_processed += 1
            session.stats.audio_seconds_processed += len(window_pcm) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
            session.stats.processing_time_total += elapsed

            text = result.get("text", "").strip()

            if not text or _is_hallucination(text):
                runtime = int(time.time() - start_time)
                display.print_speculative("", f"{session.stats.status_line()} · {runtime // 60}:{runtime % 60:02d}")
                await asyncio.sleep(STEP_SECONDS)
                continue

            # --- Confirmed / speculative split ---
            # With 20s window and 5s step, consecutive windows share ~15s overlap.
            # Text in the overlap region (appears in both windows) = confirmed.
            # Text only in the new 5s = speculative.

            new_confirmed = ""
            speculative = text

            if previous_text:
                prev_words = previous_text.split()
                curr_words = text.split()

                # Find where previous window's content ends in current window
                # (longest suffix of prev matching prefix of curr)
                best_overlap = 0
                max_check = min(len(prev_words), len(curr_words))
                for size in range(max_check, 0, -1):
                    if prev_words[-size:] == curr_words[:size]:
                        best_overlap = size
                        break

                if best_overlap > 0:
                    # Overlapping text is confirmed
                    overlap_text = " ".join(curr_words[:best_overlap])
                    # Only emit what's genuinely new vs already confirmed
                    new_conf = _diff_suffix(confirmed_so_far, overlap_text)
                    if new_conf.strip():
                        new_confirmed = new_conf.strip()

                    # After-overlap = speculative
                    speculative = " ".join(curr_words[best_overlap:])
                else:
                    # No word-level overlap found — entire diff is speculative
                    diff = _diff_suffix(previous_text, text)
                    if diff.strip():
                        speculative = diff.strip()
                    else:
                        speculative = ""
            else:
                # First window — everything is speculative
                speculative = text

            # Emit confirmed
            if new_confirmed:
                display.print_confirmed(new_confirmed)
                confirmed_so_far = (confirmed_so_far + " " + new_confirmed).strip()
                if log_file:
                    log_file.write(new_confirmed + "\n")
                    log_file.flush()

            # Show speculative + status
            runtime = int(time.time() - start_time)
            status = f"{session.stats.status_line()} · {runtime // 60}:{runtime % 60:02d}"
            display.print_speculative(speculative, status)

            previous_text = text
            sleep_time = max(0.1, STEP_SECONDS - elapsed)
            await asyncio.sleep(sleep_time)

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
        # Clean exit
        display._erase_speculative()
        sys.stdout.write(f"{DIM}shutting down...{RESET}\n")
        sys.stdout.flush()
        await session.shutdown()
        await source.stop()
        if log_file:
            log_file.close()
        sys.stdout.write(f"{DIM}done.{RESET}\n\n")
        sys.stdout.flush()
