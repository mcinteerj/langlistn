"""Speaker diarization via diart (pyannote-based streaming).

Runs as a parallel loop alongside Whisper. Consumes the same audio stream
and produces speaker labels per time window. The pipeline maps whisper word
timestamps to diarization speaker segments.

Requires: pip install langlistn[diarize]
Uses pyannote models (CPU-only on Mac — no MPS support). No conflict with
MLX whisper which uses ANE/GPU.
"""

import logging
import os
import time
from dataclasses import dataclass, field

import numpy as np

# Suppress OMP deprecation/info messages (C-level, can't be caught by Python)
os.environ.setdefault("KMP_WARNINGS", "0")
os.environ.setdefault("OMP_DISPLAY_ENV", "FALSE")
# Prevent omp_set_nested deprecation info message
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


@dataclass
class SpeakerSegment:
    """A labelled speaker segment."""
    speaker: str
    start: float  # seconds from stream start
    end: float


@dataclass
class SpeakerTracker:
    """Streaming speaker diarization using diart.

    Accumulates audio in a window and periodically runs diarization.
    Maintains a mapping of speaker labels → friendly names (Speaker A, B, ...).
    """

    step_seconds: float = 0.5
    latency_seconds: float = 5.0

    # Internal state
    _pipeline: object | None = field(default=None, repr=False)
    _speaker_map: dict[str, str] = field(default_factory=dict)
    _next_speaker_idx: int = field(default=0)
    _segments: list[SpeakerSegment] = field(default_factory=list)
    _audio_buffer: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    _buffer_offset: float = field(default=0.0)
    _last_process_time: float = field(default=0.0)
    _available: bool = field(default=False)
    _first_run: bool = field(default=True)

    def load(self) -> bool:
        """Load diart pipeline. Returns False if diart not installed."""
        try:
            from diart import SpeakerDiarization, SpeakerDiarizationConfig
            from diart.models import SegmentationModel, EmbeddingModel
            import torch

            seg = SegmentationModel.from_pyannote("pyannote/segmentation-3.0")
            emb = EmbeddingModel.from_pyannote("pyannote/wespeaker-voxceleb-resnet34-LM")
            cfg = SpeakerDiarizationConfig(
                segmentation=seg,
                embedding=emb,
                step=self.step_seconds,
                latency=self.latency_seconds,
                tau_active=0.5,
                rho_update=0.3,
                delta_new=1.0,
                device=torch.device("cpu"),
                sample_rate=SAMPLE_RATE,
            )
            config = SpeakerDiarization(cfg)
            self._pipeline = config
            self._available = True
            logger.info(
                "Diarization loaded (step=%.1fs, latency=%.1fs)",
                self.step_seconds, self.latency_seconds,
            )
            return True
        except ImportError:
            logger.info("diart not installed — diarization disabled")
            return False
        except Exception as e:
            logger.warning("Diarization load failed: %s", e)
            return False

    @property
    def available(self) -> bool:
        return self._available

    def feed_audio(self, samples_f32: np.ndarray):
        """Feed audio samples (16kHz float32). Thread-safe accumulation."""
        self._audio_buffer = np.append(self._audio_buffer, samples_f32)

    def process(self) -> list[SpeakerSegment]:
        """Run diarization on accumulated audio. Returns new segments.

        Should be called periodically (~every step_seconds).
        Non-blocking if not enough audio has accumulated.
        """
        if not self._available or self._pipeline is None:
            return []

        buf_secs = len(self._audio_buffer) / SAMPLE_RATE
        if buf_secs < self.latency_seconds:
            return []
        logger.debug("Diarization process: buf=%.1fs", buf_secs)

        now = time.time()
        if now - self._last_process_time < self.step_seconds:
            return []
        self._last_process_time = now

        try:
            # First run triggers OMP C-level messages — suppress via fd redirect
            if self._first_run:
                self._first_run = False
                old_fd = os.dup(2)
                devnull = os.open(os.devnull, os.O_WRONLY)
                os.dup2(devnull, 2)
                os.close(devnull)
                try:
                    return self._run_diarization()
                finally:
                    os.dup2(old_fd, 2)
                    os.close(old_fd)
            return self._run_diarization()
        except Exception as e:
            logger.warning("Diarization error: %s", e)
            return []

    def _run_diarization(self) -> list[SpeakerSegment]:
        """Run diart on current buffer window using SlidingWindowFeature."""
        from pyannote.core import SlidingWindow, SlidingWindowFeature

        # Build a SlidingWindowFeature for the current buffer chunk
        # diart expects duration-sized chunks (latency_seconds)
        duration = self.latency_seconds
        step_samples = int(duration * SAMPLE_RATE)

        # Extract the latest chunk (step_seconds worth of audio)
        if len(self._audio_buffer) < step_samples:
            return []

        chunk = self._audio_buffer[-step_samples:]
        # SlidingWindowFeature: (num_samples, 1) with timing info
        waveform = SlidingWindowFeature(
            chunk.reshape(-1, 1),
            SlidingWindow(
                start=self._buffer_offset + (len(self._audio_buffer) - step_samples) / SAMPLE_RATE,
                duration=1 / SAMPLE_RATE,
                step=1 / SAMPLE_RATE,
            ),
        )

        pipeline = self._pipeline
        results = pipeline([waveform])
        new_segments = []
        if results:
            for annotation, _ in results:
                if annotation is not None:
                    for segment, _, label in annotation.itertracks(yield_label=True):
                        speaker_name = self._map_speaker(str(label))
                        seg = SpeakerSegment(
                            speaker=speaker_name,
                            start=segment.start,
                            end=segment.end,
                        )
                        new_segments.append(seg)

        # Trim buffer: keep last latency_seconds
        keep_samples = int(self.latency_seconds * SAMPLE_RATE)
        if len(self._audio_buffer) > keep_samples * 2:
            trim = len(self._audio_buffer) - keep_samples
            self._buffer_offset += trim / SAMPLE_RATE
            self._audio_buffer = self._audio_buffer[trim:]

        if new_segments:
            speakers = {s.speaker for s in new_segments}
            logger.debug("Diarization: %d segments, speakers=%s", len(new_segments), speakers)
            self._segments.extend(new_segments)
            cutoff = self._buffer_offset - 60
            self._segments = [s for s in self._segments if s.end > cutoff]

        return new_segments

    def _map_speaker(self, raw_label: str) -> str:
        """Map diart's numeric labels to friendly names."""
        if raw_label not in self._speaker_map:
            names = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            idx = self._next_speaker_idx % len(names)
            self._speaker_map[raw_label] = f"Speaker {names[idx]}"
            self._next_speaker_idx += 1
        return self._speaker_map[raw_label]

    def speaker_at(self, timestamp: float) -> str | None:
        """Get the speaker label active at a given timestamp."""
        for seg in reversed(self._segments):
            if seg.start <= timestamp <= seg.end:
                return seg.speaker
        return None

    def current_speaker(self) -> str | None:
        """Get the most recent active speaker."""
        if not self._segments:
            return None
        return self._segments[-1].speaker

    def reset(self):
        """Reset all state."""
        self._speaker_map.clear()
        self._next_speaker_idx = 0
        self._segments.clear()
        self._audio_buffer = np.array([], dtype=np.float32)
        self._buffer_offset = 0.0
