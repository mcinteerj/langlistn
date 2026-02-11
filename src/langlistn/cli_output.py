"""Plain terminal output with ANSI overwriting for live translation."""

import asyncio
import shutil
import signal
import sys
import time

from .audio import AppCapture, AudioSource
from .audio.mic_capture import MicCapture
from .config import resolve_language_name
from .whisper_local import (
    LocalWhisperSession, _is_hallucination,
    STEP_SECONDS, WINDOW_SECONDS,
    SAMPLE_RATE, BYTES_PER_SAMPLE, _seconds_to_bytes,
)

# ANSI — only bold/dim/reset to respect terminal theme
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
CLEAR_LINE = "\033[2K"
MOVE_UP = "\033[A"


def _wrap_text(text: str, width: int) -> list[str]:
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


MAX_SPEC_LINES = 6  # Cap speculative display to avoid runaway overwrites


class TerminalDisplay:
    """Live terminal output — confirmed text scrolls up, speculative gets overwritten."""

    def __init__(self, plain: bool = False):
        self.width = shutil.get_terminal_size().columns
        self.plain = plain
        self._spec_lines_on_screen = 0
        self._recent_confirmed: list[str] = []  # dedup buffer

    def _erase_speculative(self):
        if self.plain:
            return
        for _ in range(self._spec_lines_on_screen):
            sys.stdout.write(f"{MOVE_UP}{CLEAR_LINE}\r")
        self._spec_lines_on_screen = 0

    def print_confirmed(self, text: str, original: str | None = None):
        """Print confirmed translation. If original provided, show above."""
        if not text.strip():
            return
        # Dedup: skip if we recently confirmed the same text
        normalized = text.strip().lower()
        if normalized in self._recent_confirmed:
            return
        self._recent_confirmed.append(normalized)
        # Keep last 10 for dedup window
        if len(self._recent_confirmed) > 10:
            self._recent_confirmed = self._recent_confirmed[-10:]
        self._erase_speculative()
        if self.plain:
            if original:
                sys.stdout.write(f"{original}\n")
            sys.stdout.write(f"{text}\n")
        else:
            if original:
                lines = _wrap_text(original, self.width - 1)
                for line in lines:
                    sys.stdout.write(f"{DIM}{line}{RESET}\n")
            lines = _wrap_text(text, self.width - 1)
            for line in lines:
                sys.stdout.write(f"{BOLD}{line}{RESET}\n")
        sys.stdout.flush()

    def print_speculative(self, text: str, status: str, original: str | None = None):
        """Print speculative text + status (will be overwritten next call)."""
        if self.plain:
            # Plain mode: don't print speculative (only confirmed)
            return
        self._erase_speculative()
        count = 0
        if original and original.strip():
            lines = _wrap_text(original, self.width - 1)
            for line in lines:
                sys.stdout.write(f"{DIM}{line}{RESET}\n")
            count += len(lines)
        if text.strip():
            lines = _wrap_text(text, self.width - 1)[:MAX_SPEC_LINES]
            for line in lines:
                sys.stdout.write(f"{DIM}{line}{RESET}\n")
            count += len(lines)
        status_trunc = status[:self.width - 1]
        sys.stdout.write(f"{DIM}{status_trunc}{RESET}\n")
        count += 1
        self._spec_lines_on_screen = count
        sys.stdout.flush()

    def print_header(self, mode: str, lang: str | None, model: str):
        if self.plain:
            return
        lang_display = f"{lang} → English" if lang else "auto-detect → English"
        sep = "─" * self.width
        model_short = model.split("/")[-1] if "/" in model else model
        sys.stdout.write(f"\n{BOLD}langlistn{RESET} — {lang_display} — {mode} — {model_short}\n")
        sys.stdout.write(f"{DIM}{sep}{RESET}\n\n")
        sys.stdout.flush()


async def run_cli(
    app_name: str | None = None,
    mic: bool = False,
    device: str | None = None,
    lang: str | None = None,
    model: str | None = None,
    log_path: str | None = None,
    plain: bool = False,
    dual_lang: bool = False,
) -> None:
    """Run langlistn with plain terminal output."""

    lang_name = resolve_language_name(lang)

    source: AudioSource
    if app_name:
        source = AppCapture(app_name)
        mode = f"app: {app_name}"
    elif mic:
        source = MicCapture(device=device)
        mode = "mic"
    else:
        raise ValueError("Must specify --app or --mic")

    session = LocalWhisperSession(
        lang=lang,
        model=model,  # None = auto-select based on RAM
        dual_lang=dual_lang,
    )

    display = TerminalDisplay(plain=plain)
    display.print_header(mode, lang_name or lang, session.model_name)

    log_file = None
    if log_path:
        try:
            log_file = open(log_path, "a", encoding="utf-8")
        except OSError as e:
            sys.stderr.write(f"Warning: could not open log file: {e}\n")

    display.print_speculative("", "starting audio capture...")
    try:
        await source.start()
    except Exception as e:
        sys.stderr.write(f"\nAudio capture failed: {e}\n")
        return

    display.print_speculative("", f"loading {session.model_name}...")
    await session.connect()

    shutdown = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown.set)

    # --- Timestamp-based confirmed/speculative ---
    # With 20s window / 5s step:
    #   [0, 10s]  = already confirmed → skip
    #   [10s, 15s] = newly confirmed → emit
    #   [15s, 20s] = speculative → overwrite next cycle

    overlap_seconds = WINDOW_SECONDS - STEP_SECONDS  # 15s
    confirm_start = overlap_seconds - STEP_SECONDS    # 10s
    windows_processed = 0
    last_speculative_text = ""  # For flushing on shutdown

    async def audio_loop():
        while not shutdown.is_set():
            chunk = await source.read_chunk()
            if chunk is None:
                break
            await session.send_audio(chunk)

    async def process_loop():
        nonlocal windows_processed, last_speculative_text
        start_time = time.time()
        min_buffer_bytes = _seconds_to_bytes(STEP_SECONDS)

        # Wait for initial buffer
        while not shutdown.is_set():
            await asyncio.sleep(0.5)
            pcm, dur = session._get_window_pcm()
            if len(pcm) >= min_buffer_bytes:
                break

        display.print_speculative("", "listening...")

        while not shutdown.is_set():
            # Wait for either step interval or early speech→silence trigger
            try:
                await asyncio.wait_for(
                    session._early_trigger.wait(),
                    timeout=STEP_SECONDS,
                )
                session._early_trigger.clear()
            except asyncio.TimeoutError:
                pass

            if shutdown.is_set():
                break

            # Grab window
            pcm, duration = session._get_window_pcm()
            if len(pcm) < min_buffer_bytes:
                continue

            # Run Whisper
            t0 = time.time()
            try:
                if session.dual_lang:
                    translated, transcribed = await loop.run_in_executor(
                        None, session._transcribe_dual, pcm
                    )
                    result = translated
                    original_text = transcribed.get("text", "").strip()
                    original_segments = transcribed.get("segments", [])
                else:
                    result = await loop.run_in_executor(
                        None, session._transcribe, pcm, None
                    )
                    original_text = None
                    original_segments = None
            except Exception as e:
                display.print_speculative("", f"error: {e}")
                continue
            elapsed = time.time() - t0

            session.stats.windows_processed += 1
            session.stats.audio_seconds_processed += duration
            session.stats.processing_time_total += elapsed
            windows_processed += 1

            text = result.get("text", "").strip()
            segments = result.get("segments", [])

            if not text or _is_hallucination(text):
                runtime = int(time.time() - start_time)
                display.print_speculative(
                    "", f"{session.stats.status_line()} · {runtime // 60}:{runtime % 60:02d}"
                )
                continue

            # Split by timestamp
            confirmed_parts: list[str] = []
            speculative_parts: list[str] = []
            confirmed_orig_parts: list[str] = []
            speculative_orig_parts: list[str] = []

            if windows_processed <= 1:
                speculative_parts = [
                    s.get("text", "").strip() for s in segments
                    if s.get("text", "").strip()
                ]
                if original_segments:
                    speculative_orig_parts = [
                        s.get("text", "").strip() for s in original_segments
                        if s.get("text", "").strip()
                    ]
            else:
                for seg in segments:
                    seg_text = seg.get("text", "").strip()
                    if not seg_text:
                        continue
                    seg_end = seg.get("end", 0)
                    if seg_end <= confirm_start:
                        continue
                    elif seg_end <= overlap_seconds:
                        confirmed_parts.append(seg_text)
                    else:
                        speculative_parts.append(seg_text)

                if original_segments:
                    for seg in original_segments:
                        seg_text = seg.get("text", "").strip()
                        if not seg_text:
                            continue
                        seg_end = seg.get("end", 0)
                        if seg_end <= confirm_start:
                            continue
                        elif seg_end <= overlap_seconds:
                            confirmed_orig_parts.append(seg_text)
                        else:
                            speculative_orig_parts.append(seg_text)

            confirmed_text = " ".join(confirmed_parts)
            speculative_text = " ".join(speculative_parts)
            confirmed_orig = " ".join(confirmed_orig_parts) if confirmed_orig_parts else None
            speculative_orig = " ".join(speculative_orig_parts) if speculative_orig_parts else None

            last_speculative_text = speculative_text

            if confirmed_text:
                display.print_confirmed(confirmed_text, original=confirmed_orig)
                if log_file:
                    if confirmed_orig:
                        log_file.write(f"[original] {confirmed_orig}\n")
                    log_file.write(f"{confirmed_text}\n")
                    log_file.flush()

            runtime = int(time.time() - start_time)
            status = f"{session.stats.status_line()} · {runtime // 60}:{runtime % 60:02d}"
            display.print_speculative(speculative_text, status, original=speculative_orig)

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
        # Flush last speculative as confirmed (otherwise we lose the tail)
        if last_speculative_text:
            display.print_confirmed(last_speculative_text)
            if log_file:
                log_file.write(f"{last_speculative_text}\n")

        display._erase_speculative()
        if not plain:
            sys.stdout.write(f"{DIM}done.{RESET}\n\n")
            sys.stdout.flush()
        await session.shutdown()
        await source.stop()
        if log_file:
            log_file.close()
