"""Vendored from ufal/whisper_streaming (MIT License).

Core classes: HypothesisBuffer + OnlineASRProcessor implementing LocalAgreement-2
policy for streaming Whisper transcription. Only numpy required.

Original: https://github.com/ufal/whisper_streaming
Authors: Dominik Macháček, et al.
License: MIT

Adapted for langlistn: removed backends, tokenizer deps, VAC. Kept pure
agreement logic + segment-based buffer trimming.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

SAMPLING_RATE = 16000


class HypothesisBuffer:
    """Tracks two consecutive Whisper outputs and commits their common prefix."""

    def __init__(self):
        self.commited_in_buffer: list[tuple] = []
        self.buffer: list[tuple] = []
        self.new: list[tuple] = []
        self.last_commited_time = 0.0
        self.last_commited_word = None

    def insert(self, new: list[tuple], offset: float):
        """Insert new timestamped words, filtering already-committed content."""
        new = [(a + offset, b + offset, t) for a, b, t in new]
        self.new = [(a, b, t) for a, b, t in new if a > self.last_commited_time - 0.1]

        if len(self.new) >= 1:
            a, b, t = self.new[0]
            if abs(a - self.last_commited_time) < 1:
                if self.commited_in_buffer:
                    # Drop 1–5 gram overlap with already-committed tail
                    cn = len(self.commited_in_buffer)
                    nn = len(self.new)
                    for i in range(1, min(min(cn, nn), 5) + 1):
                        c = " ".join(
                            [self.commited_in_buffer[-j][2] for j in range(1, i + 1)][::-1]
                        )
                        tail = " ".join(self.new[j - 1][2] for j in range(1, i + 1))
                        if c == tail:
                            for _ in range(i):
                                self.new.pop(0)
                            logger.debug("removing last %d overlapping words", i)
                            break

    def flush(self) -> list[tuple]:
        """Commit the longest common prefix of last two inserts (LocalAgreement-2)."""
        commit = []
        while self.new:
            na, nb, nt = self.new[0]
            if len(self.buffer) == 0:
                break
            if nt == self.buffer[0][2]:
                commit.append((na, nb, nt))
                self.last_commited_word = nt
                self.last_commited_time = nb
                self.buffer.pop(0)
                self.new.pop(0)
            else:
                break
        self.buffer = self.new
        self.new = []
        self.commited_in_buffer.extend(commit)
        return commit

    def pop_commited(self, time: float):
        while self.commited_in_buffer and self.commited_in_buffer[0][1] <= time:
            self.commited_in_buffer.pop(0)

    def complete(self) -> list[tuple]:
        """Return uncommitted (speculative) words."""
        return self.buffer


class MLXWhisperASR:
    """Thin ASR adapter for mlx-whisper, compatible with OnlineASRProcessor."""

    sep = " "

    def __init__(self, model_path: str, lang: str | None = None, task: str = "translate"):
        self.model_path = model_path
        self.lang = lang
        self.task = task
        self._mlx_whisper = None

    def load(self):
        """Load model (warm up with dummy audio)."""
        import mlx_whisper
        dummy = np.zeros(SAMPLING_RATE, dtype=np.float32)
        mlx_whisper.transcribe(
            dummy,
            path_or_hf_repo=self.model_path,
            task=self.task,
            language=self.lang,
            fp16=True,
            word_timestamps=True,
            no_speech_threshold=0.6,
        )
        self._mlx_whisper = mlx_whisper

    def transcribe(self, audio: np.ndarray, init_prompt: str = "") -> list[dict]:
        # Cap init_prompt to avoid KV cache bloat (perplexity confirms 2-5x slowdown)
        prompt = (init_prompt or "")[-100:] or None
        result = self._mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self.model_path,
            task=self.task,
            language=self.lang,
            initial_prompt=prompt,
            fp16=True,
            word_timestamps=True,
            no_speech_threshold=0.5,
            condition_on_previous_text=False,
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            temperature=0.0,
            best_of=1,
        )
        return result.get("segments", [])

    def ts_words(self, segments: list[dict]) -> list[tuple]:
        """Extract (start, end, word) tuples from segments."""
        out = []
        for seg in segments:
            if seg.get("no_speech_prob", 0) > 0.9:
                continue
            for w in seg.get("words", []):
                out.append((w["start"], w["end"], w["word"]))
        return out

    def segments_end_ts(self, segments: list[dict]) -> list[float]:
        return [s["end"] for s in segments]


class OnlineASRProcessor:
    """Growing-buffer processor with LocalAgreement-2 and segment trimming."""

    def __init__(self, asr: MLXWhisperASR, buffer_trimming_sec: float = 15):
        self.asr = asr
        self.buffer_trimming_sec = buffer_trimming_sec
        self.init()

    def init(self, offset: float | None = None):
        self.audio_buffer = np.array([], dtype=np.float32)
        self.transcript_buffer = HypothesisBuffer()
        self.buffer_time_offset = 0.0
        if offset is not None:
            self.buffer_time_offset = offset
        self.transcript_buffer.last_commited_time = self.buffer_time_offset
        self.commited: list[tuple] = []

    def insert_audio_chunk(self, audio: np.ndarray):
        self.audio_buffer = np.append(self.audio_buffer, audio)

    def prompt(self) -> tuple[str, str]:
        """Build prompt from committed text outside current buffer."""
        k = max(0, len(self.commited) - 1)
        while k > 0 and self.commited[k - 1][1] > self.buffer_time_offset:
            k -= 1
        p = [t for _, _, t in self.commited[:k]]
        prompt_parts = []
        length = 0
        while p and length < 200:
            x = p.pop(-1)
            length += len(x) + 1
            prompt_parts.append(x)
        non_prompt = self.asr.sep.join(t for _, _, t in self.commited[k:])
        return self.asr.sep.join(prompt_parts[::-1]), non_prompt

    def process_iter(self) -> tuple[float | None, float | None, str]:
        """Run Whisper on buffer, return confirmed text via LocalAgreement."""
        prompt, non_prompt = self.prompt()
        logger.debug(
            "transcribing %.2fs from %.2f",
            len(self.audio_buffer) / SAMPLING_RATE,
            self.buffer_time_offset,
        )
        res = self.asr.transcribe(self.audio_buffer, init_prompt=prompt)
        tsw = self.asr.ts_words(res)

        self.transcript_buffer.insert(tsw, self.buffer_time_offset)
        o = self.transcript_buffer.flush()
        self.commited.extend(o)

        # Segment-based buffer trimming
        if len(self.audio_buffer) / SAMPLING_RATE > self.buffer_trimming_sec:
            self._chunk_completed_segment(res)

        # Hard cap: if buffer exceeds 35s, force trim to keep buffer_trimming_sec
        buf_secs = len(self.audio_buffer) / SAMPLING_RATE
        if buf_secs > self.buffer_trimming_sec + 10:
            trim_to = self.buffer_time_offset + buf_secs - self.buffer_trimming_sec
            logger.debug("hard cap: trimming to %.2f (buffer was %.2fs)", trim_to, buf_secs)
            self._chunk_at(trim_to)

        logger.debug("buffer now: %.2fs", len(self.audio_buffer) / SAMPLING_RATE)
        return self._to_flush(o)

    def _chunk_completed_segment(self, res):
        if not self.commited:
            return
        ends = self.asr.segments_end_ts(res)
        t = self.commited[-1][1]
        if len(ends) > 1:
            e = ends[-2] + self.buffer_time_offset
            while len(ends) > 2 and e > t:
                ends.pop(-1)
                e = ends[-2] + self.buffer_time_offset
            if e <= t:
                logger.debug("segment chunked at %.2f", e)
                self._chunk_at(e)

    def _chunk_at(self, time: float):
        self.transcript_buffer.pop_commited(time)
        cut_seconds = time - self.buffer_time_offset
        self.audio_buffer = self.audio_buffer[int(cut_seconds * SAMPLING_RATE):]
        self.buffer_time_offset = time

    def finish(self) -> tuple[float | None, float | None, str]:
        """Flush remaining uncommitted text."""
        o = self.transcript_buffer.complete()
        f = self._to_flush(o)
        self.buffer_time_offset += len(self.audio_buffer) / SAMPLING_RATE
        return f

    def get_speculative(self) -> str:
        """Return current uncommitted (speculative) text."""
        words = self.transcript_buffer.complete()
        return self.asr.sep.join(w[2] for w in words)

    def _to_flush(self, sents) -> tuple[float | None, float | None, str]:
        t = self.asr.sep.join(s[2] for s in sents)
        if not sents:
            return (None, None, "")
        return (sents[0][0], sents[-1][1], t)
