"""Two-loop pipeline: transcription (Whisper) + translation (Claude).

Loop 1 — Transcription (~1-2s cycle):
  - mlx-whisper with LocalAgreement-2
  - Confirmed source text fed back as init_prompt
  - Lock prevents parallel whisper runs

Loop 2 — Translation (on every transcription change):
  - Sends full source transcript + confirmed English to Claude
  - Claude returns full translation (not just continuation)
  - Sentence-level confirmation via consecutive-output diff

Both loops feed a shared display with locked/speculative zones.
"""

import array
import asyncio
import logging
import math
import signal
import sys
import termios
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

from .audio import AudioSource
from .config import SILENCE_RMS_THRESHOLD
from .display import Spinner, TerminalDisplay
from .streaming_asr import MLXWhisperASR, OnlineASRProcessor, SAMPLING_RATE

from .translate import ContinuationTranslator


def _load_vad():
    """Load Silero VAD model. Returns (model, get_speech_timestamps)."""
    from silero_vad import load_silero_vad
    torch.set_num_threads(1)  # VAD is tiny, don't compete with MLX
    model = load_silero_vad()
    return model


_VAD_WINDOW = 512  # Silero requires exactly 512 samples at 16kHz


def _chunk_has_speech(vad_model, samples_f32: np.ndarray, threshold: float = 0.3) -> bool:
    """Run Silero VAD on a chunk. Returns True if any sub-window has speech."""
    for i in range(0, len(samples_f32) - _VAD_WINDOW + 1, _VAD_WINDOW):
        tensor = torch.from_numpy(samples_f32[i:i + _VAD_WINDOW])
        speech_prob = vad_model(tensor, SAMPLING_RATE).item()
        if speech_prob >= threshold:
            return True
    return False


@dataclass
class PipelineConfig:
    """Tunables for the two-loop pipeline."""

    # Transcription
    process_interval: float = 0.3
    min_audio_seconds: float = 1.0
    buffer_trimming_sec: float = 25
    whisper_timeout: float = 8.0        # Kill whisper if it takes longer than this

    # Translation
    translate_on_every_cycle: bool = True
    max_context_chars: int = 2000
    force_confirm_after: int = 3

    # Silence / reset
    silence_reset_seconds: float = 10.0

    # Display
    dual_lang: bool = False

    # Diarization



def _rms_int16(chunk: bytes) -> float:
    samples = array.array("h")
    samples.frombytes(chunk)
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples))


def _is_hallucination(text: str) -> bool:
    """Detect Whisper hallucination loops."""
    if not text.strip():
        return True

    # CJK character repetition
    if len(text) > 10:
        char_counts = Counter(c for c in text if not c.isspace())
        if char_counts:
            most_common_count = char_counts.most_common(1)[0][1]
            total_chars = sum(char_counts.values())
            if most_common_count > 10 and most_common_count / total_chars > 0.5:
                return True

    # N-gram repetition
    words = text.split()
    if len(words) < 6:
        return False
    for ngram_size in range(1, 7):
        if len(words) < ngram_size * 3:
            continue
        ngrams: dict[tuple, int] = {}
        for i in range(len(words) - ngram_size + 1):
            gram = tuple(w.lower().strip(".,!?") for w in words[i : i + ngram_size])
            ngrams[gram] = ngrams.get(gram, 0) + 1
        max_count = max(ngrams.values())
        if max_count >= 3 and ngram_size >= 2:
            # Any 2+ word phrase repeated 3+ times is suspicious
            return True
        possible = len(words) - ngram_size + 1
        if max_count >= max(3, possible * 0.3):
            return True
    return False



def _read_single_key() -> str:
    """Read a single keypress without waiting for Enter."""
    import tty
    import termios
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


def _offer_transcript_save(text: str, start_time: float):
    """Prompt user to save transcript to Desktop."""
    try:
        sys.stdout.write("Save transcript? [Y/n] ")
        sys.stdout.flush()
        ch = _read_single_key().lower()
        sys.stdout.write(ch + "\n")
    except (EOFError, KeyboardInterrupt, OSError):
        sys.stdout.write("\n")
        return
    if ch in ("y", "\r", "\n"):
        ts = time.strftime("%Y%m%d-%H%M%S", time.localtime(start_time))
        path = Path.home() / "Desktop" / f"langlistn-{ts}.txt"
        path.write_text(text, encoding="utf-8")
        print(f"  Saved → {path}")


async def run_pipeline(
    source: AudioSource,
    mode: str,
    lang: str | None = None,
    model: str | None = None,
    translate_model: str | None = "haiku",
    no_translate: bool = False,
    plain: bool = False,
    log_path: str | None = None,
    config: PipelineConfig | None = None,
) -> None:
    """Run the two-loop pipeline."""
    from .config import recommend_model, resolve_language_name

    cfg = config or PipelineConfig()
    lang_name = resolve_language_name(lang)
    model_name = model or recommend_model()

    # ASR
    asr = MLXWhisperASR(model_path=model_name, lang=lang, task="transcribe")
    processor = OnlineASRProcessor(asr, buffer_trimming_sec=cfg.buffer_trimming_sec)

    # Translator
    translator = None
    if not no_translate:
        translator = ContinuationTranslator(
            model_tier=translate_model or "haiku",
            source_lang=lang,
            max_context_chars=cfg.max_context_chars,
            force_confirm_after=cfg.force_confirm_after,
        )

    display = TerminalDisplay(plain=plain)
    display.print_header(
        mode,
        lang_name or lang,
        model_name,
        translate_model if translator else None,
    )

    log_file = None
    if log_path:
        try:
            log_file = open(log_path, "a", encoding="utf-8")
        except OSError as e:
            sys.stderr.write(f"Warning: could not open log file: {e}\n")

    # Start audio
    sys.stdout.write(f"  starting audio capture...\r")
    sys.stdout.flush()
    try:
        await source.start()
    except Exception as e:
        sys.stderr.write(f"\nAudio capture failed: {e}\n")
        return
    sys.stdout.write(f"\033[2K\r")
    sys.stdout.flush()

    # Load whisper model + VAD
    loading_msg = f"loading {model_name} + VAD"

    # Detect first-run: check if model files are cached locally
    hint = ""
    try:
        import huggingface_hub
        local_path = huggingface_hub.try_to_load_from_cache(model_name, "config.json")
        if local_path is None:
            hint = "(downloading model ~3GB — first run only)"
    except Exception:
        pass

    spinner = Spinner(loading_msg, hint=hint)
    await spinner.start()

    loop = asyncio.get_running_loop()
    load_done = asyncio.Event()
    load_error = None
    vad_model = None

    def _load():
        nonlocal load_error, vad_model
        try:
            asr.load()
            vad_model = _load_vad()
            logger.info("VAD model loaded")
        except Exception as e:
            load_error = e
        finally:
            loop.call_soon_threadsafe(load_done.set)

    threading.Thread(target=_load, daemon=True).start()
    await load_done.wait()
    await spinner.stop()

    if load_error:
        sys.stderr.write(f"\nModel load failed: {load_error}\n")
        return

    shutdown = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown.set)

    # ── shared state ──
    confirmed_source = ""
    confirmed_display = ""  # accumulated display text (survives silence resets)
    speculative_source = ""
    last_speech_time = time.time()
    whisper_lock = asyncio.Lock()
    windows_processed = 0
    processing_time_total = 0.0
    start_time = time.time()
    silence_reset_done = False  # prevents repeated resets
    consecutive_timeouts = 0
    consecutive_hallucinations = 0  # track whisper hallucination streaks
    recent_whisper_hypotheses: list[str] = []  # last N raw whisper speculative outputs


    def _build_status() -> str:
        runtime = int(time.time() - start_time)
        runtime_str = f"{runtime // 60}:{runtime % 60:02d}"
        cost_display = f"${translator.estimated_cost():.2f}" if translator else ""
        silence_dur = time.time() - last_speech_time
        if silence_dur > 2.0:
            return f"⏸  waiting for speech... · {runtime_str} · {cost_display}".rstrip(" ·")
        return f"🎧 {runtime_str} · {cost_display}".rstrip(" ·")

    async def audio_loop():
        nonlocal last_speech_time
        while not shutdown.is_set():
            chunk = await source.read_chunk()
            if chunk is None:
                break
            rms = _rms_int16(chunk)
            if rms < SILENCE_RMS_THRESHOLD:
                continue
            samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            # Silero VAD gate — only feed speech to whisper
            if vad_model is not None:
                if not _chunk_has_speech(vad_model, samples):
                    continue
            last_speech_time = time.time()
            processor.insert_audio_chunk(samples)


    async def transcribe_and_translate():
        nonlocal confirmed_source, confirmed_display, speculative_source
        nonlocal windows_processed, processing_time_total
        nonlocal silence_reset_done, consecutive_timeouts, consecutive_hallucinations


        # Wait for initial audio
        while not shutdown.is_set():
            await asyncio.sleep(0.5)
            if len(processor.audio_buffer) / SAMPLING_RATE >= cfg.min_audio_seconds:
                break

        display.update("", "", "listening...")

        while not shutdown.is_set():
            await asyncio.sleep(cfg.process_interval)
            if shutdown.is_set():
                break

            buf_seconds = len(processor.audio_buffer) / SAMPLING_RATE

            # ── silence detection ──
            silence_duration = time.time() - last_speech_time
            if silence_duration > cfg.silence_reset_seconds:
                if not silence_reset_done:
                    if translator:
                        # Promote any remaining speculative to confirmed before reset
                        if translator.speculative_translation:
                            translator.confirmed_translation = (
                                (translator.confirmed_translation + " " + translator.speculative_translation)
                                .strip()
                            )
                            translator.speculative_translation = ""
                            full = translator.confirmed_translation
                            display.update(full, "", "silence — context locked")
                        translator._last_full_output = ""
                    else:
                        # No-translate: promote source text to display baseline
                        confirmed_display += (" " + confirmed_source if confirmed_display else confirmed_source)
                        confirmed_display = confirmed_display.strip()
                    confirmed_source = ""
                    silence_reset_done = True
                if buf_seconds < cfg.min_audio_seconds:
                    locked_text = translator.confirmed_translation if translator else confirmed_display
                    display.update(locked_text, "", _build_status())
                    continue
            else:
                silence_reset_done = False

            if buf_seconds < cfg.min_audio_seconds:
                if translator:
                    locked_text = translator.confirmed_translation
                else:
                    locked_text = (confirmed_display + " " + confirmed_source).strip() if confirmed_source else confirmed_display
                display.update(locked_text, "", _build_status())
                continue

            # Skip if whisper already running
            if whisper_lock.locked():
                continue

            async with whisper_lock:
                t0 = time.time()
                try:
                    fut = loop.run_in_executor(None, processor.process_iter)
                    _beg, _end, confirmed = await asyncio.wait_for(
                        asyncio.shield(fut), timeout=cfg.whisper_timeout,
                    )
                except asyncio.TimeoutError:
                    elapsed = time.time() - t0
                    consecutive_timeouts += 1
                    logger.warning(
                        "WHISPER timeout after %.1fs (consecutive: %d) — waiting for thread",
                        elapsed, consecutive_timeouts,
                    )
                    try:
                        await fut
                    except Exception:
                        pass
                    logger.warning("WHISPER slow run finished after %.1fs total — discarding", time.time() - t0)

                    # After 2 consecutive timeouts, reset processor to break
                    # the hallucination→bad prompt→slow whisper death spiral
                    if consecutive_timeouts >= 2:
                        logger.warning("WHISPER resetting processor after %d consecutive timeouts", consecutive_timeouts)
                        offset = processor.buffer_time_offset + len(processor.audio_buffer) / SAMPLING_RATE
                        processor.init(offset=offset)
                        consecutive_timeouts = 0
                    continue
                except Exception:
                    continue
                elapsed = time.time() - t0
                windows_processed += 1
                processing_time_total += elapsed
                consecutive_timeouts = 0

                logger.debug(
                    "WHISPER #%d | %.1fs | confirmed=%r | speculative=%r",
                    windows_processed, elapsed,
                    confirmed[:150] if confirmed else "",
                    processor.get_speculative()[:150],
                )

                # Filter hallucinations
                is_halluc = False
                if confirmed and _is_hallucination(confirmed):
                    logger.debug("WHISPER filtered hallucination: %r", confirmed[:100])
                    confirmed = ""
                    is_halluc = True
                if confirmed:
                    words = confirmed.split()
                    if len(words) > 3:
                        unique_ratio = len(set(w.lower() for w in words)) / len(words)
                        if unique_ratio < 0.3:
                            confirmed = ""
                            is_halluc = True

                speculative_source = processor.get_speculative()
                if speculative_source and _is_hallucination(speculative_source):
                    speculative_source = ""
                    is_halluc = True

                # Stash raw whisper hypothesis for multi-hypothesis LLM feeding
                raw_hyp = processor.get_speculative()
                if raw_hyp and not _is_hallucination(raw_hyp):
                    recent_whisper_hypotheses.append(raw_hyp)
                    if len(recent_whisper_hypotheses) > 3:
                        recent_whisper_hypotheses.pop(0)

                # Track consecutive hallucinations — reset processor if stuck
                if is_halluc and not confirmed:
                    consecutive_hallucinations += 1
                    if consecutive_hallucinations >= 3:
                        logger.warning(
                            "WHISPER resetting processor after %d consecutive hallucinations",
                            consecutive_hallucinations,
                        )
                        offset = processor.buffer_time_offset + len(processor.audio_buffer) / SAMPLING_RATE
                        processor.init(offset=offset)
                        consecutive_hallucinations = 0
                else:
                    consecutive_hallucinations = 0

                if confirmed:
                    confirmed_source += (" " + confirmed if confirmed_source else confirmed)

                # ── translation ──
                # Only translate if we have something meaningful
                full_source = confirmed_source
                if speculative_source:
                    full_source += " " + speculative_source

                if not translator or not full_source.strip():
                    locked = (confirmed_display + " " + confirmed_source).strip() if confirmed_source else confirmed_display
                    spec = speculative_source
                elif confirmed or (cfg.translate_on_every_cycle and speculative_source):
                    # Only call LLM when there's new confirmed or new speculative
                    # Feed alternative whisper hypotheses for disambiguation
                    alt_hypotheses = [
                        h for h in recent_whisper_hypotheses[:-1]
                        if h != speculative_source
                    ][-2:]  # max 2 alternatives

                    try:
                        locked, spec = await loop.run_in_executor(
                            None, translator.translate, full_source, alt_hypotheses
                        )
                    except Exception:
                        locked = translator.confirmed_translation
                        spec = translator.speculative_translation
                else:
                    locked = translator.confirmed_translation
                    spec = translator.speculative_translation

            # ── display ──
            avg_ms = (
                (processing_time_total / windows_processed * 1000)
                if windows_processed
                else 0
            )
            logger.debug(
                "stats | windows=%d avg=%.0fms buf=%.0fs",
                windows_processed, avg_ms, buf_seconds,
            )
            status = _build_status()
            display.update(locked, spec, status)

            if log_file and confirmed:
                log_file.write(f"{confirmed}\n")
                log_file.flush()

    try:
        tasks = [
            asyncio.create_task(audio_loop()),
            asyncio.create_task(transcribe_and_translate()),
            asyncio.create_task(shutdown.wait()),
        ]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
    except asyncio.CancelledError:
        pass
    finally:
        # Flush remaining speculative
        _, _, remaining = processor.finish()
        if remaining and not _is_hallucination(remaining):
            confirmed_source += (" " + remaining if confirmed_source else remaining)

        full_final = ""
        all_source = (confirmed_display + " " + confirmed_source).strip()
        if translator and all_source:
            try:
                locked, spec = translator.translate(all_source)
                full_final = (locked + " " + spec).strip()
                display.update(full_final, "")
            except Exception:
                full_final = translator.confirmed_translation or all_source
                display.update(full_final, "")
        elif all_source:
            full_final = all_source
            display.update(all_source, "")

        duration = time.time() - start_time
        cost = translator.estimated_cost() if translator else 0.0
        final_text = full_final or all_source
        word_count = len(final_text.split()) if final_text.strip() else 0
        sentence_count = sum(final_text.count(p) for p in '.!?。！？') if final_text.strip() else 0
        llm_calls = translator.calls if translator else 0
        display.finish(
            duration=duration, words=word_count, sentences=sentence_count,
            llm_calls=llm_calls, cost=cost,
            has_content=bool(all_source),
        )

        # Offer transcript save
        if all_source and not plain and final_text.strip():
            _offer_transcript_save(final_text, start_time)

        await source.stop()
        if log_file:
            log_file.close()
