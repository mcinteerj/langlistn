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

import asyncio
import logging
import signal
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
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
    min_audio_seconds: float = 1.0
    buffer_trimming_sec: float = 18
    whisper_timeout: float = 8.0        # Kill whisper if it takes longer than this

    # Translation
    translate_on_every_cycle: bool = True
    max_context_chars: int = 2000
    force_confirm_after: int = 3

    # Silence / reset
    silence_reset_seconds: float = 10.0

def _rms_int16(chunk: bytes) -> float:
    samples = np.frombuffer(chunk, dtype=np.int16)
    if len(samples) == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))


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
    import termios
    try:
        sys.stdout.write("Save transcript? [Y/n] ")
        sys.stdout.flush()
        
        # Flush any buffered input (e.g. from Ctrl+C or typing during session)
        fd = sys.stdin.fileno()
        termios.tcflush(fd, termios.TCIFLUSH)
        
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
    display.update("", "", "⏸  waiting for speech...")

    if load_error:
        sys.stderr.write(f"\nModel load failed: {load_error}\n")
        return

    shutdown = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown.set)

    # ── shared state ──
    @dataclass
    class SharedState:
        confirmed_source: str = ""
        speculative_source: str = ""
        confirmed_translation: str = ""
        speculative_translation: str = ""
        confirmed_display: str = ""  # survives silence resets
        source_version: int = 0
        last_translated_version: int = 0
        last_speech_time: float = 0.0
        silence_reset_done: bool = False
        streaming_active: bool = False
        lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    state = SharedState(last_speech_time=time.time())
    source_changed = asyncio.Event()
    audio_ready = asyncio.Event()  # set by audio_loop when enough new audio arrives
    whisper_lock = asyncio.Lock()
    executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="langlistn")
    new_audio_seconds = 0.0  # tracks audio accumulated since last whisper run
    windows_processed = 0
    processing_time_total = 0.0
    start_time = time.time()
    consecutive_timeouts = 0
    consecutive_hallucinations = 0
    recent_whisper_hypotheses: list[str] = []


    def _build_status() -> str:
        runtime = int(time.time() - start_time)
        runtime_str = f"{runtime // 60}:{runtime % 60:02d}"
        cost_display = f"${translator.estimated_cost():.2f}" if translator else ""
        silence_dur = time.time() - state.last_speech_time
        if silence_dur > 2.0:
            return f"⏸  waiting for speech... · {runtime_str} · {cost_display}".rstrip(" ·")
        return f"🎧 {runtime_str} · {cost_display}".rstrip(" ·")

    async def audio_loop():
        nonlocal new_audio_seconds
        vad_cooldown = 0  # skip VAD for N chunks after speech detected
        while not shutdown.is_set():
            chunk = await source.read_chunk()
            if chunk is None:
                break
            if len(chunk) == 0:
                continue
            rms = _rms_int16(chunk)
            if rms < SILENCE_RMS_THRESHOLD:
                vad_cooldown = 0  # silence resets cooldown
                continue
            samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            if vad_model is not None:
                if vad_cooldown > 0:
                    vad_cooldown -= 1
                elif not _chunk_has_speech(vad_model, samples):
                    continue
                else:
                    vad_cooldown = 8  # ~0.5s: skip VAD during established speech
            state.last_speech_time = time.time()
            processor.insert_audio_chunk(samples)
            new_audio_seconds += len(samples) / SAMPLING_RATE
            if new_audio_seconds >= cfg.min_audio_seconds:
                audio_ready.set()


    async def whisper_loop():
        nonlocal windows_processed, processing_time_total
        nonlocal consecutive_timeouts, consecutive_hallucinations

        # Wait for initial audio
        while not shutdown.is_set():
            await asyncio.sleep(0.5)
            if len(processor.audio_buffer) / SAMPLING_RATE >= cfg.min_audio_seconds:
                break

        display.update("", "", "listening...")

        while not shutdown.is_set():
            # Event-driven: wait for audio_loop to signal enough new audio,
            # with timeout for silence detection and status updates
            try:
                await asyncio.wait_for(audio_ready.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                pass
            audio_ready.clear()
            new_audio_seconds = 0.0

            if shutdown.is_set():
                break

            buf_seconds = len(processor.audio_buffer) / SAMPLING_RATE

            # ── silence detection ──
            silence_duration = time.time() - state.last_speech_time
            if silence_duration > cfg.silence_reset_seconds:
                if not state.silence_reset_done:
                    async with state.lock:
                        if translator:
                            if translator.speculative_translation:
                                translator.confirmed_translation = (
                                    (translator.confirmed_translation + " " + translator.speculative_translation)
                                    .strip()
                                )
                                translator.speculative_translation = ""
                                display.update(translator.confirmed_translation, "", "silence — context locked")
                        else:
                            state.confirmed_display += (" " + state.confirmed_source if state.confirmed_display else state.confirmed_source)
                            state.confirmed_display = state.confirmed_display.strip()
                        state.confirmed_source = ""
                        state.silence_reset_done = True
                if buf_seconds < cfg.min_audio_seconds:
                    if not state.streaming_active:
                        locked_text = translator.confirmed_translation if translator else state.confirmed_display
                        display.update(locked_text, "", _build_status())
                    continue
            else:
                state.silence_reset_done = False

            if buf_seconds < cfg.min_audio_seconds:
                if not state.streaming_active:
                    if translator:
                        locked_text = translator.confirmed_translation
                    else:
                        locked_text = (state.confirmed_display + " " + state.confirmed_source).strip() if state.confirmed_source else state.confirmed_display
                    display.update(locked_text, "", _build_status())
                continue

            # Skip if whisper already running
            if whisper_lock.locked():
                continue

            async with whisper_lock:
                t0 = time.time()
                try:
                    fut = loop.run_in_executor(executor, processor.process_iter)
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
                        await asyncio.wait_for(fut, timeout=30.0)
                    except (asyncio.TimeoutError, Exception):
                        logger.error("WHISPER thread hung or failed — discarding")
                    logger.warning("WHISPER slow run finished after %.1fs total — discarding", time.time() - t0)
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

                speculative = processor.get_speculative()

                # Stash raw hypothesis before filtering
                if speculative and not _is_hallucination(speculative):
                    recent_whisper_hypotheses.append(speculative)
                elif speculative:
                    speculative = ""
                    is_halluc = True

                while len(recent_whisper_hypotheses) > 3:
                    recent_whisper_hypotheses.pop(0)

                # Track consecutive hallucinations
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

                # Update shared state
                async with state.lock:
                    if confirmed:
                        state.confirmed_source += (" " + confirmed if state.confirmed_source else confirmed)
                    state.speculative_source = speculative
                    state.source_version += 1
                    source_changed.set()

                # Display source immediately (unless translation is streaming)
                if not state.streaming_active:
                    if translator:
                        locked = translator.confirmed_translation
                        spec = translator.speculative_translation
                    else:
                        locked = (state.confirmed_display + " " + state.confirmed_source).strip() if state.confirmed_source else state.confirmed_display
                        spec = speculative
                    display.update(locked, spec, _build_status())

                if log_file and confirmed:
                    log_file.write(f"{confirmed}\n")
                    log_file.flush()

            # Brief cooldown between whisper runs — lets CPU/GPU clock down
            await asyncio.sleep(0.2)


    async def translate_loop():
        """Independent translation loop — reacts to source_version changes."""
        if not translator:
            return  # no-translate mode: this loop is a no-op

        while not shutdown.is_set():
            # Wait for source changes (with timeout for periodic retranslation)
            try:
                await asyncio.wait_for(source_changed.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                pass
            source_changed.clear()

            if shutdown.is_set():
                break

            # Skip if nothing new to translate
            async with state.lock:
                if state.source_version == state.last_translated_version:
                    continue
                if state.silence_reset_done:
                    continue
                full_source = state.confirmed_source
                if state.speculative_source:
                    full_source += " " + state.speculative_source
                current_version = state.source_version

            if not full_source.strip():
                continue

            alt_hypotheses = [
                h for h in recent_whisper_hypotheses[:-1]
                if h != state.speculative_source
            ][-2:]

            # Snapshot confirmed text BEFORE streaming starts — this stays
            # constant throughout the stream so the display doesn't see
            # the locked zone jump when _update_confirmation runs at the end.
            confirmed_snapshot = translator.confirmed_translation

            # Streaming callback for live display updates
            def on_token(partial_translation: str):
                # The LLM outputs the FULL translation from scratch each time.
                # While it's still reproducing the already-confirmed prefix,
                # don't show anything speculative (it would duplicate).
                # Only show the new tail once it extends past confirmed text.
                if len(partial_translation) <= len(confirmed_snapshot):
                    return  # still reproducing confirmed prefix — no update
                spec_part = partial_translation[len(confirmed_snapshot):].strip()
                if spec_part:
                    display.update(confirmed_snapshot, spec_part, _build_status())

            state.streaming_active = True
            try:
                locked, spec = await loop.run_in_executor(
                    executor, translator.translate_streaming, full_source, alt_hypotheses, on_token
                )
            except Exception:
                locked = translator.confirmed_translation
                spec = translator.speculative_translation
            finally:
                state.streaming_active = False

            async with state.lock:
                state.last_translated_version = current_version

            display.update(locked, spec, _build_status())

    try:
        tasks = [
            asyncio.create_task(audio_loop()),
            asyncio.create_task(whisper_loop()),
            asyncio.create_task(shutdown.wait()),
        ]
        if translator:
            tasks.append(asyncio.create_task(translate_loop()))
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        # Short timeout — don't block exit waiting for audio subprocess
        try:
            await asyncio.wait_for(
                asyncio.gather(*pending, return_exceptions=True), timeout=1.0
            )
        except asyncio.TimeoutError:
            pass
    except asyncio.CancelledError:
        pass
    finally:
        # Kill audio source immediately to avoid lingering subprocess
        await source.stop()
        # Flush remaining speculative
        _, _, remaining = processor.finish()
        if remaining and not _is_hallucination(remaining):
            state.confirmed_source += (" " + remaining if state.confirmed_source else remaining)

        full_final = ""
        all_source = (state.confirmed_display + " " + state.confirmed_source).strip()
        if translator and all_source:
            # Use existing translation if available, skip final LLM call
            existing = (translator.confirmed_translation + " " + translator.speculative_translation).strip()
            if existing:
                full_final = existing
                display.update(full_final, "")
            else:
                try:
                    locked, spec = translator.translate(all_source)
                    full_final = (locked + " " + spec).strip()
                    display.update(full_final, "")
                except Exception:
                    full_final = all_source
                    display.update(all_source, "")
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

        if log_file:
            log_file.close()

        executor.shutdown(wait=False)
