"""Local Whisper session — sliding window transcription/translation."""

import array
import asyncio
import logging
import math
import os
import time

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

from collections import deque
from dataclasses import dataclass, field

from ..config import SILENCE_RMS_THRESHOLD

logger = logging.getLogger(__name__)

from ..realtime import EventKind, SessionEvent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WINDOW_SECONDS = 20.0
STEP_SECONDS = 5.0
OVERLAP_SECONDS = WINDOW_SECONDS - STEP_SECONDS
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2

# Speech detection for early processing
SPEECH_RMS_THRESHOLD = 80       # Higher than silence gate — confirms active speech
SILENCE_AFTER_SPEECH_S = 2.0    # Silence after speech triggers early process

# Model recommendations by available RAM
MODEL_BY_RAM = [
    (32, "mlx-community/whisper-large-v3-mlx"),       # 32GB+ → large-v3 (3GB)
    (16, "mlx-community/whisper-large-v3-mlx"),       # 16GB  → large-v3 (tight but works)
    (12, "mlx-community/whisper-medium-mlx"),          # 12GB  → medium (1.5GB)
    (8,  "mlx-community/whisper-small-mlx"),           # 8GB   → small (500MB)
    (0,  "mlx-community/whisper-tiny"),                # fallback
]


def recommend_model() -> str:
    """Pick best model based on available system RAM."""
    try:
        import subprocess
        out = subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"], text=True
        ).strip()
        ram_gb = int(out) / (1024 ** 3)
    except Exception:
        ram_gb = 16  # assume 16 if can't detect
    for min_ram, model in MODEL_BY_RAM:
        if ram_gb >= min_ram:
            logger.info("RAM: %.0fGB → model: %s", ram_gb, model)
            return model
    return MODEL_BY_RAM[-1][1]


def _rms_int16(chunk: bytes) -> float:
    samples = array.array("h")
    samples.frombytes(chunk)
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples))


def _seconds_to_bytes(seconds: float) -> int:
    return int(seconds * SAMPLE_RATE * BYTES_PER_SAMPLE)


def _is_hallucination(text: str) -> bool:
    """Detect Whisper hallucination loops (repeated phrases/characters)."""
    # CJK character repetition (e.g. 紅紅紅紅紅)
    if len(text) > 10:
        from collections import Counter
        char_counts = Counter(c for c in text if not c.isspace())
        if char_counts:
            most_common_count = char_counts.most_common(1)[0][1]
            total_chars = sum(char_counts.values())
            if most_common_count > 10 and most_common_count / total_chars > 0.5:
                logger.debug("Hallucination: char repetition %d/%d", most_common_count, total_chars)
                return True

    words = text.split()
    if len(words) < 8:
        return False
    for ngram_size in range(1, 7):
        if len(words) < ngram_size * 3:
            continue
        ngrams: dict[tuple, int] = {}
        for i in range(len(words) - ngram_size + 1):
            gram = tuple(w.lower().strip(".,!?") for w in words[i:i + ngram_size])
            ngrams[gram] = ngrams.get(gram, 0) + 1
        max_count = max(ngrams.values())
        possible = len(words) - ngram_size + 1
        if max_count >= max(4, possible * 0.4):
            logger.debug("Hallucination: ngram=%d count=%d/%d", ngram_size, max_count, possible)
            return True
    return False


@dataclass
class TimestampedChunk:
    """Audio chunk tagged with absolute capture time."""
    data: bytes
    abs_time: float  # when this chunk was captured (time.monotonic)
    rms: float


@dataclass
class SessionStats:
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
    """Local Whisper with speech-aware sliding window.

    Improvements over naive ring buffer:
    - Chunks are timestamped so we can track absolute audio position
    - Speech detection triggers early processing after silence gap
    - Silence chunks are excluded from the window (only speech is fed to Whisper)
    - Supports dual-mode: translate + transcribe for bilingual output
    """

    def __init__(
        self,
        lang: str | None = None,
        model: str | None = None,
        task: str = "translate",
        dual_lang: bool = False,
    ):
        self.lang = lang
        self.model_name = model or recommend_model()
        self.task = task
        self.dual_lang = dual_lang  # If True, run both transcribe and translate
        self._mlx_whisper = None
        self._shutdown = False

        # Timestamped chunk buffer
        self._chunks: deque[TimestampedChunk] = deque()
        self._buffer_lock = asyncio.Lock()
        self._max_buffer_seconds = WINDOW_SECONDS + 5.0

        # Speech state for early trigger
        self._last_speech_time = 0.0
        self._speech_active = False
        self._early_trigger = asyncio.Event()

        # Event queue
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
        await self._emit(EventKind.STATUS, f"loading {self.model_name}...")
        import threading
        load_error = None
        load_done = asyncio.Event()
        loop = asyncio.get_running_loop()

        def _bg_load():
            nonlocal load_error
            try:
                self._load_model()
            except Exception as e:
                load_error = e
            finally:
                loop.call_soon_threadsafe(load_done.set)

        threading.Thread(target=_bg_load, daemon=True).start()
        await load_done.wait()
        if load_error:
            raise load_error
        self.stats.connect_time = time.time()
        await self._emit(EventKind.STATUS, "model loaded · waiting for audio")

    def _load_model(self) -> None:
        import mlx_whisper
        import numpy as np
        dummy = np.zeros(SAMPLE_RATE, dtype=np.float32)
        mlx_whisper.transcribe(
            dummy, path_or_hf_repo=self.model_name,
            task=self.task, language=self.lang, fp16=True,
            no_speech_threshold=0.6,
        )
        self._mlx_whisper = mlx_whisper
        logger.info("Model loaded: %s", self.model_name)

    async def send_audio(self, chunk: bytes) -> None:
        """Buffer timestamped audio chunks. Detect speech→silence transitions."""
        rms = _rms_int16(chunk)
        now = time.monotonic()

        if rms < SILENCE_RMS_THRESHOLD:
            self.stats.silence_chunks_skipped += 1
            # Check for speech→silence transition (early trigger)
            if self._speech_active and (now - self._last_speech_time) > SILENCE_AFTER_SPEECH_S:
                self._speech_active = False
                self._early_trigger.set()
            return

        # Has speech
        if rms >= SPEECH_RMS_THRESHOLD:
            self._last_speech_time = now
            self._speech_active = True

        tc = TimestampedChunk(data=chunk, abs_time=now, rms=rms)
        async with self._buffer_lock:
            self._chunks.append(tc)
            # Trim old chunks
            cutoff = now - self._max_buffer_seconds
            while self._chunks and self._chunks[0].abs_time < cutoff:
                self._chunks.popleft()

    def _get_window_pcm(self, max_seconds: float = WINDOW_SECONDS) -> tuple[bytes, float]:
        """Get up to max_seconds of audio from buffer. Returns (pcm_bytes, duration_seconds)."""
        if not self._chunks:
            return b"", 0.0
        # Take chunks from the end spanning max_seconds
        now = self._chunks[-1].abs_time
        cutoff = now - max_seconds
        parts = []
        for c in self._chunks:
            if c.abs_time >= cutoff:
                parts.append(c.data)
        pcm = b"".join(parts)
        duration = len(pcm) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
        return pcm, duration

    def _transcribe(self, pcm_bytes: bytes, task: str | None = None) -> dict:
        """Run Whisper on PCM16 audio (runs in thread)."""
        import numpy as np
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        result = self._mlx_whisper.transcribe(
            samples,
            path_or_hf_repo=self.model_name,
            task=task or self.task,
            language=self.lang,
            fp16=True,
            no_speech_threshold=0.5,
            condition_on_previous_text=False,
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
        )
        # Filter no-speech segments
        segments = result.get("segments", [])
        filtered = [s for s in segments if s.get("no_speech_prob", 0) <= 0.5]
        result["text"] = " ".join(s.get("text", "").strip() for s in filtered).strip()
        result["segments"] = filtered
        return result

    def _transcribe_dual(self, pcm_bytes: bytes) -> tuple[dict, dict]:
        """Run both translate and transcribe for bilingual output."""
        translated = self._transcribe(pcm_bytes, task="translate")
        transcribed = self._transcribe(pcm_bytes, task="transcribe")
        return translated, transcribed

    async def commit_watchdog(self) -> None:
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

    # For TUI compatibility (receive_loop not used in CLI mode)
    async def receive_loop(self) -> None:
        """No-op for CLI mode — processing happens in cli_output.process_loop."""
        try:
            while not self._shutdown:
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            pass
