"""Plain terminal output with ANSI overwriting for live translation.

Cascade architecture:
  Audio → mlx-whisper transcribe (LocalAgreement-2) → confirmed source text
    → if English: passthrough
    → if non-English: Bedrock Claude translate → confirmed English
  Speculative: dim raw transcription from Whisper
  Confirmed: bold English
"""

import array
import asyncio
import math
import shutil
import signal
import sys
import time
import threading

import numpy as np

from .audio import AppCapture, AudioSource
from .audio.mic_capture import MicCapture
from .config import resolve_language_name
from .streaming_asr import MLXWhisperASR, OnlineASRProcessor, SAMPLING_RATE
from .translate import BedrockTranslator
from .whisper_local import recommend_model, _is_hallucination, SILENCE_RMS_THRESHOLD

# ANSI — only bold/dim/reset to respect terminal theme
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
CLEAR_LINE = "\033[2K"
MOVE_UP = "\033[A"


def _rms_int16(chunk: bytes) -> float:
    samples = array.array("h")
    samples.frombytes(chunk)
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples))


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


MAX_SPEC_LINES = 6


class TerminalDisplay:
    """Live terminal output — confirmed text scrolls up, speculative gets overwritten."""

    def __init__(self, plain: bool = False):
        self.width = shutil.get_terminal_size().columns
        self.plain = plain
        self._spec_lines_on_screen = 0

    def _erase_speculative(self):
        if self.plain:
            return
        for _ in range(self._spec_lines_on_screen):
            sys.stdout.write(f"{MOVE_UP}{CLEAR_LINE}\r")
        self._spec_lines_on_screen = 0

    def print_confirmed(self, text: str, original: str | None = None):
        if not text.strip():
            return
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
        if self.plain:
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

    def print_header(self, mode: str, lang: str | None, model: str, translate_model: str | None = None):
        if self.plain:
            return
        lang_display = f"{lang} → English" if lang else "auto-detect → English"
        sep = "─" * self.width
        model_short = model.split("/")[-1] if "/" in model else model
        translate_info = f" + Claude {translate_model}" if translate_model else ""
        sys.stdout.write(f"\n{BOLD}langlistn{RESET} — {lang_display} — {mode} — {model_short}{translate_info}\n")
        sys.stdout.write(f"{DIM}{sep}{RESET}\n\n")
        sys.stdout.flush()


def _looks_english(text: str) -> bool:
    """Quick heuristic: if most chars are ASCII letters, it's probably English."""
    if not text.strip():
        return True
    ascii_letters = sum(1 for c in text if c.isascii() and c.isalpha())
    all_letters = sum(1 for c in text if c.isalpha())
    if all_letters == 0:
        return True
    return (ascii_letters / all_letters) > 0.8


async def _translate_and_display(
    translator: BedrockTranslator,
    source_text: str,
    lang: str | None,
    dual_lang: bool,
    display: TerminalDisplay,
    log_file,
    loop: asyncio.AbstractEventLoop,
):
    """Translate and display in background — doesn't block process loop."""
    try:
        translation = await loop.run_in_executor(
            None, translator.translate, source_text, lang
        )
        display.print_confirmed(
            translation,
            original=source_text if dual_lang else None,
        )
        if log_file:
            if dual_lang:
                log_file.write(f"[original] {source_text}\n")
            log_file.write(f"{translation}\n")
            log_file.flush()
    except Exception:
        display.print_confirmed(source_text)


async def run_cli(
    app_name: str | None = None,
    mic: bool = False,
    device: str | None = None,
    lang: str | None = None,
    model: str | None = None,
    log_path: str | None = None,
    plain: bool = False,
    dual_lang: bool = False,
    translate_model: str | None = "haiku",
    no_translate: bool = False,
) -> None:
    """Run langlistn with cascade: local Whisper transcription + Bedrock translation."""

    lang_name = resolve_language_name(lang)
    model_name = model or recommend_model()

    source: AudioSource
    if app_name:
        source = AppCapture(app_name)
        mode = f"app: {app_name}"
    elif mic:
        source = MicCapture(device=device)
        mode = "mic"
    else:
        raise ValueError("Must specify --app or --mic")

    # ASR: transcribe in source language (not translate)
    asr = MLXWhisperASR(model_path=model_name, lang=lang, task="transcribe")
    processor = OnlineASRProcessor(asr, buffer_trimming_sec=15)

    # Translator: Bedrock Claude
    translator = None
    if not no_translate:
        translator = BedrockTranslator(model_tier=translate_model or "haiku")

    display = TerminalDisplay(plain=plain)
    display.print_header(mode, lang_name or lang, model_name, translate_model if translator else None)

    log_file = None
    if log_path:
        try:
            log_file = open(log_path, "a", encoding="utf-8")
        except OSError as e:
            sys.stderr.write(f"Warning: could not open log file: {e}\n")

    # Start audio capture
    display.print_speculative("", "starting audio capture...")
    try:
        await source.start()
    except Exception as e:
        sys.stderr.write(f"\nAudio capture failed: {e}\n")
        return

    # Load Whisper model in background thread
    display.print_speculative("", f"loading {model_name}...")
    loop = asyncio.get_running_loop()
    load_done = asyncio.Event()
    load_error = None

    def _load():
        nonlocal load_error
        try:
            asr.load()
        except Exception as e:
            load_error = e
        finally:
            loop.call_soon_threadsafe(load_done.set)

    threading.Thread(target=_load, daemon=True).start()
    await load_done.wait()
    if load_error:
        sys.stderr.write(f"\nModel load failed: {load_error}\n")
        return

    shutdown = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown.set)

    # Stats
    windows_processed = 0
    processing_time_total = 0.0
    start_time = time.time()

    # Batch buffer: accumulate committed source text until sentence boundary
    pending_source = ""

    MIN_CHUNK_SECONDS = 1.0
    PROCESS_INTERVAL = 0.3  # Process frequently — Whisper run takes ~2s anyway

    def _is_sentence_end(text: str) -> bool:
        """Check if text ends at a natural boundary."""
        text = text.rstrip()
        if not text:
            return False
        # Punctuation that indicates sentence end (including CJK)
        return text[-1] in ".!?。！？…"

    async def audio_loop():
        while not shutdown.is_set():
            chunk = await source.read_chunk()
            if chunk is None:
                break
            rms = _rms_int16(chunk)
            if rms < SILENCE_RMS_THRESHOLD:
                continue
            samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            processor.insert_audio_chunk(samples)

    async def process_loop():
        nonlocal windows_processed, processing_time_total, pending_source

        # Wait for initial audio
        while not shutdown.is_set():
            await asyncio.sleep(0.5)
            if len(processor.audio_buffer) / SAMPLING_RATE >= MIN_CHUNK_SECONDS:
                break

        display.print_speculative("", "listening...")

        while not shutdown.is_set():
            await asyncio.sleep(PROCESS_INTERVAL)
            if shutdown.is_set():
                break

            buf_seconds = len(processor.audio_buffer) / SAMPLING_RATE
            if buf_seconds < MIN_CHUNK_SECONDS:
                continue

            # Run Whisper transcription (in thread)
            t0 = time.time()
            try:
                beg, end, confirmed = await loop.run_in_executor(
                    None, processor.process_iter
                )
            except Exception as e:
                display.print_speculative("", f"error: {e}")
                continue
            elapsed = time.time() - t0
            windows_processed += 1
            processing_time_total += elapsed

            # Accumulate confirmed source text (filter hallucinations)
            if confirmed:
                if _is_hallucination(confirmed):
                    confirmed = ""
                else:
                    # Check for word-level repetition (e.g. "ket ket ket ket")
                    words = confirmed.split()
                    if len(words) > 3:
                        unique_ratio = len(set(w.lower() for w in words)) / len(words)
                        if unique_ratio < 0.3:
                            confirmed = ""

            if confirmed:
                pending_source += (" " + confirmed if pending_source else confirmed)

            # Translate as soon as we have any confirmed text
            if pending_source and (
                _is_sentence_end(pending_source)
                or len(pending_source) > 60
                or (len(pending_source) > 10 and not confirmed)  # any pause → flush
            ):
                source_text = pending_source.strip()
                pending_source = ""

                # Final hallucination check on accumulated text
                if _is_hallucination(source_text):
                    continue

                if _looks_english(source_text) or not translator:
                    display.print_confirmed(source_text)
                    if log_file:
                        log_file.write(f"{source_text}\n")
                        log_file.flush()
                else:
                    # Fire translation without blocking process loop
                    asyncio.create_task(
                        _translate_and_display(
                            translator, source_text, lang, dual_lang,
                            display, log_file, loop,
                        )
                    )

            # Update speculative display
            spec = processor.get_speculative()
            runtime = int(time.time() - start_time)
            avg_ms = (processing_time_total / windows_processed * 1000) if windows_processed else 0
            cost_str = f"${translator.estimated_cost():.3f}" if translator else "$0.00"
            status = (
                f"listening · windows: {windows_processed} · "
                f"avg: {avg_ms:.0f}ms · buf: {buf_seconds:.0f}s · "
                f"{runtime // 60}:{runtime % 60:02d} · {cost_str}"
            )
            if spec and not _is_hallucination(spec):
                display.print_speculative(spec, status)
            else:
                display.print_speculative("", status)

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
        # Flush remaining
        _, _, remaining = processor.finish()
        if remaining:
            pending_source += (" " + remaining if pending_source else remaining)

        if pending_source.strip():
            source_text = pending_source.strip()
            if _looks_english(source_text) or not translator:
                display.print_confirmed(source_text)
            else:
                try:
                    translation = translator.translate(source_text, lang)
                    display.print_confirmed(translation, original=source_text if dual_lang else None)
                except Exception:
                    display.print_confirmed(f"[{source_text}]")

        display._erase_speculative()
        if not plain:
            cost_str = f"${translator.estimated_cost():.3f}" if translator else "$0.00"
            sys.stdout.write(f"{DIM}done. · {cost_str}{RESET}\n\n")
            sys.stdout.flush()
        await source.stop()
        if log_file:
            log_file.close()
