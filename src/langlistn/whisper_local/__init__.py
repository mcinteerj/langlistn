"""Local Whisper session — sliding window transcription/translation with streaming feel."""

import array
import asyncio
import logging
import math
import os
import time

# Prevent tqdm from creating multiprocessing locks inside Textual's event loop.
# tqdm → multiprocessing.RLock → resource_tracker → spawnv_passfds fails because
# Textual's terminal FDs are in a non-inheritable state.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
from collections import deque
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from ..config import SILENCE_RMS_THRESHOLD

logger = logging.getLogger(__name__)

# Re-use event types from realtime module for TUI compatibility
from ..realtime import EventKind, SessionEvent


# ---------------------------------------------------------------------------
# Sliding window constants
# ---------------------------------------------------------------------------
WINDOW_SECONDS = 5.0        # Total window fed to Whisper
STEP_SECONDS = 2.0          # How often we run Whisper (new audio per step)
OVERLAP_SECONDS = WINDOW_SECONDS - STEP_SECONDS  # 3s overlap
CONFIRM_APPEARANCES = 2     # Text must appear in N consecutive windows to confirm
SAMPLE_RATE = 16000          # Input sample rate from capture
BYTES_PER_SAMPLE = 2


def _rms_int16(chunk: bytes) -> float:
    samples = array.array("h")
    samples.frombytes(chunk)
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples))


def _seconds_to_bytes(seconds: float) -> int:
    return int(seconds * SAMPLE_RATE * BYTES_PER_SAMPLE)


def _is_hallucination(text: str) -> bool:
    """Detect Whisper hallucination loops (repeated phrases).

    Whisper hallucinates by repeating short phrases endlessly, e.g.
    "even if I do, even if I do, even if I do, ..."
    """
    words = text.split()
    if len(words) < 8:
        return False

    # Check for short repeating patterns (1-6 word ngrams)
    for ngram_size in range(1, 7):
        if len(words) < ngram_size * 3:
            continue
        # Count occurrences of each ngram
        ngrams: dict[tuple, int] = {}
        for i in range(len(words) - ngram_size + 1):
            gram = tuple(w.lower().strip(".,!?") for w in words[i:i + ngram_size])
            ngrams[gram] = ngrams.get(gram, 0) + 1

        # If any ngram appears in >40% of possible positions, it's a loop
        max_count = max(ngrams.values())
        possible = len(words) - ngram_size + 1
        if max_count >= max(4, possible * 0.4):
            logger.debug("Hallucination detected: ngram=%d max_count=%d/%d",
                         ngram_size, max_count, possible)
            return True

    return False


def _diff_suffix(previous: str, current: str) -> str:
    """Find the new text in `current` that wasn't in `previous`.

    Uses word-level tail matching to find the overlap point.
    """
    if not previous:
        return current
    if not current:
        return ""

    # Fast path: current starts with previous
    if current.startswith(previous):
        return current[len(previous):].lstrip()

    prev_words = previous.split()
    curr_words = current.split()

    # Find longest suffix of prev that matches a prefix of curr
    best_overlap = 0
    max_check = min(len(prev_words), len(curr_words))
    for size in range(max_check, 0, -1):
        if prev_words[-size:] == curr_words[:size]:
            best_overlap = size
            break

    if best_overlap > 0:
        new_words = curr_words[best_overlap:]
    else:
        # No overlap — check if curr is entirely contained in prev (duplicate window)
        curr_str = " ".join(curr_words).lower()
        prev_str = " ".join(prev_words).lower()
        if curr_str in prev_str:
            return ""
        new_words = curr_words

    return " ".join(new_words)


@dataclass
class WindowResult:
    """Result from one Whisper window."""
    text: str
    timestamp: float


@dataclass
class SessionStats:
    """Track session statistics."""
    windows_processed: int = 0
    audio_seconds_processed: float = 0
    silence_chunks_skipped: int = 0
    processing_time_total: float = 0
    connect_time: float = 0

    def status_line(self) -> str:
        parts = ["listening (local)"]
        if self.windows_processed > 0:
            avg_ms = (self.processing_time_total / self.windows_processed) * 1000
            parts.append(f"windows: {self.windows_processed}")
            parts.append(f"avg: {avg_ms:.0f}ms")
            parts.append(f"audio: {self.audio_seconds_processed:.0f}s")
        parts.append("$0.00")
        return " · ".join(parts)


class LocalWhisperSession:
    """Manages local Whisper transcription with sliding window for streaming feel.

    Architecture:
    - Audio chunks accumulate in a ring buffer
    - Every STEP_SECONDS, we grab a WINDOW_SECONDS window and run Whisper
    - We diff against previous output to emit only new text
    - Text appearing in consecutive windows is "confirmed" (solid)
    - Text only in latest window is "speculative" (dim)
    """

    def __init__(
        self,
        lang: str | None = None,
        model: str = "mlx-community/whisper-large-v3-mlx",
        task: str = "translate",
    ):
        self.lang = lang
        self.model_name = model
        self.task = task  # "translate" for → English, "transcribe" for source lang
        self._model = None
        self._processor = None
        self._shutdown = False

        # Audio ring buffer (raw PCM16 bytes)
        self._audio_buffer = bytearray()
        self._buffer_lock = asyncio.Lock()
        self._max_buffer_bytes = _seconds_to_bytes(WINDOW_SECONDS + 2.0)  # slight extra

        # Sliding window state
        self._previous_text = ""
        self._confirmed_texts: list[str] = []
        self._last_window_texts: deque[str] = deque(maxlen=CONFIRM_APPEARANCES)

        # Event queue (same interface as RealtimeSession)
        self._event_queue: asyncio.Queue[SessionEvent] = asyncio.Queue(maxsize=1000)

        self.stats = SessionStats()

    async def _emit(self, kind: EventKind, data: str = "") -> None:
        event = SessionEvent(kind, data)
        try:
            self._event_queue.put_nowait(event)
        except asyncio.QueueFull:
            try:
                self._event_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                self._event_queue.put_nowait(event)
            except asyncio.QueueFull:
                pass

    async def connect(self) -> None:
        """Load the Whisper model."""
        await self._emit(EventKind.STATUS, f"loading model {self.model_name}...")

        # Load model in a background thread to avoid blocking event loop.
        # Use threading directly since Textual's event loop can have FD issues
        # with asyncio.run_in_executor.
        import threading
        load_error = None
        load_done = asyncio.Event()

        def _bg_load():
            nonlocal load_error
            try:
                self._load_model()
            except Exception as e:
                load_error = e
            finally:
                # Schedule the event set on the event loop
                loop.call_soon_threadsafe(load_done.set)

        loop = asyncio.get_running_loop()
        thread = threading.Thread(target=_bg_load, daemon=True)
        thread.start()
        await load_done.wait()

        if load_error:
            raise load_error

        self.stats.connect_time = time.time()
        await self._emit(EventKind.STATUS, "model loaded · waiting for audio")

    def _load_model(self) -> None:
        """Load mlx-whisper model (runs in thread)."""
        import mlx_whisper
        # Warm up by running a tiny transcription
        import numpy as np
        dummy = np.zeros(SAMPLE_RATE, dtype=np.float32)  # 1s silence
        mlx_whisper.transcribe(
            dummy,
            path_or_hf_repo=self.model_name,
            task=self.task,
            language=self.lang,
            fp16=True,
            no_speech_threshold=0.6,
        )
        self._mlx_whisper = mlx_whisper
        logger.info("Whisper model loaded: %s", self.model_name)

    async def send_audio(self, chunk: bytes) -> None:
        """Buffer incoming audio chunks."""
        rms = _rms_int16(chunk)
        if rms < SILENCE_RMS_THRESHOLD:
            self.stats.silence_chunks_skipped += 1
            return

        async with self._buffer_lock:
            self._audio_buffer.extend(chunk)
            # Trim to max buffer size (keep most recent)
            if len(self._audio_buffer) > self._max_buffer_bytes:
                excess = len(self._audio_buffer) - self._max_buffer_bytes
                del self._audio_buffer[:excess]

    async def receive_loop(self) -> None:
        """Main processing loop — runs Whisper on sliding windows."""
        step_bytes = _seconds_to_bytes(STEP_SECONDS)
        window_bytes = _seconds_to_bytes(WINDOW_SECONDS)

        # Wait for enough audio to accumulate
        while not self._shutdown:
            await asyncio.sleep(0.2)
            if len(self._audio_buffer) >= step_bytes:
                break

        while not self._shutdown:
            # Grab window
            async with self._buffer_lock:
                buf_len = len(self._audio_buffer)
                if buf_len < step_bytes:
                    await asyncio.sleep(0.5)
                    continue
                # Take up to window_bytes from the end
                take = min(buf_len, window_bytes)
                window_pcm = bytes(self._audio_buffer[-take:])

            # Check if window has any audio energy
            rms = _rms_int16(window_pcm)
            if rms < SILENCE_RMS_THRESHOLD:
                await asyncio.sleep(STEP_SECONDS)
                continue

            # Run Whisper in executor
            loop = asyncio.get_running_loop()
            t0 = time.time()
            try:
                result = await loop.run_in_executor(
                    None, self._transcribe, window_pcm
                )
            except Exception as e:
                logger.error("Whisper error: %s", e)
                await self._emit(EventKind.ERROR, f"whisper: {e}")
                await asyncio.sleep(STEP_SECONDS)
                continue
            elapsed = time.time() - t0

            self.stats.windows_processed += 1
            self.stats.audio_seconds_processed += len(window_pcm) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
            self.stats.processing_time_total += elapsed

            # Extract text
            text = result.get("text", "").strip()
            if not text:
                await self._emit(EventKind.STATUS, self.stats.status_line())
                await asyncio.sleep(STEP_SECONDS)
                continue

            # Drop hallucinated repetition loops
            if _is_hallucination(text):
                logger.info("Dropped hallucination: %r", text[:80])
                await self._emit(EventKind.STATUS, self.stats.status_line())
                await asyncio.sleep(STEP_SECONDS)
                continue

            # Diff against previous to find new content
            new_text = _diff_suffix(self._previous_text, text)

            if new_text.strip():
                # Emit new text as streaming delta
                await self._emit(EventKind.TEXT, new_text + " ")
                # Finalize after each window (the TUI moves live text → history)
                await self._emit(EventKind.TURN_COMPLETE)

            # Also emit transcript (original language) if available
            segments = result.get("segments", [])
            if segments and self.task == "translate":
                # mlx-whisper doesn't separate original text in translate mode,
                # but we can note this for future transcribe+translate dual pass
                pass

            self._previous_text = text
            self._last_window_texts.append(text)

            await self._emit(EventKind.STATUS, self.stats.status_line())

            # Sleep for remainder of step interval
            sleep_time = max(0.1, STEP_SECONDS - elapsed)
            await asyncio.sleep(sleep_time)

    def _transcribe(self, pcm_bytes: bytes) -> dict:
        """Run Whisper on PCM16 audio (runs in thread)."""
        import numpy as np

        # Convert PCM16 bytes to float32 [-1.0, 1.0]
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        result = self._mlx_whisper.transcribe(
            samples,
            path_or_hf_repo=self.model_name,
            task=self.task,
            language=self.lang,
            fp16=True,
            no_speech_threshold=0.5,
            # IMPORTANT: disable condition_on_previous_text to prevent
            # hallucination snowballing across windows
            condition_on_previous_text=False,
            compression_ratio_threshold=2.4,  # default, catches repetitive outputs
            logprob_threshold=-1.0,
        )

        # Post-filter: drop segments with high no_speech_prob
        segments = result.get("segments", [])
        filtered = []
        for seg in segments:
            if seg.get("no_speech_prob", 0) > 0.5:
                logger.debug("Dropping no-speech segment: %r", seg.get("text", "")[:50])
                continue
            filtered.append(seg)

        # Rebuild text from filtered segments
        result["text"] = " ".join(s.get("text", "").strip() for s in filtered).strip()
        result["segments"] = filtered
        return result

    async def commit_watchdog(self) -> None:
        """No-op — local whisper doesn't need commit watchdog."""
        try:
            while not self._shutdown:
                await asyncio.sleep(5.0)
        except asyncio.CancelledError:
            pass

    async def get_event(self) -> SessionEvent:
        return await self._event_queue.get()

    async def disconnect(self) -> None:
        self._shutdown = True

    async def shutdown(self) -> None:
        self._shutdown = True
