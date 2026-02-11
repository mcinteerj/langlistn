"""OpenAI Realtime API session — audio in, text out.

Uses manual turn detection (no server VAD) to prevent the API from
cancelling in-progress responses when new speech arrives.  Client-side
silence detection decides when to commit audio and request a response.
"""

import array
import asyncio
import base64
import enum
import json
import logging
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field

import websockets
import websockets.exceptions

from ..config import (
    BACKOFF_INITIAL,
    BACKOFF_MAX,
    CLIENT_VAD_MIN_SPEECH_CHUNKS,
    CLIENT_VAD_SILENCE_CHUNKS,
    MAX_SPEECH_DURATION_S,
    RECONNECT_BUFFER_MAX,
    SILENCE_RMS_THRESHOLD,
    build_system_prompt,
)

logger = logging.getLogger(__name__)


class EventKind(enum.StrEnum):
    """Types of events emitted by RealtimeSession."""
    TEXT = "text"
    TURN_COMPLETE = "turn_complete"
    TRANSCRIPT = "transcript"
    STATUS = "status"
    ERROR = "error"


@dataclass
class SessionEvent:
    """Event emitted to the TUI."""
    kind: EventKind
    data: str = ""


# Pricing per 1M tokens by model
MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-realtime": {
        "audio_input": 100.0,
        "text_input": 5.0,
        "audio_output": 200.0,
        "text_output": 20.0,
    },
    "gpt-realtime-mini": {
        "audio_input": 10.0,
        "text_input": 0.60,
        "audio_output": 20.0,
        "text_output": 2.40,
    },
}
DEFAULT_PRICING = MODEL_PRICING["gpt-realtime-mini"]


def _resolve_pricing(deployment: str) -> dict[str, float]:
    """Match deployment name to pricing. Falls back to mini pricing."""
    dep = deployment.lower()
    if "mini" in dep:
        return MODEL_PRICING["gpt-realtime-mini"]
    if "realtime" in dep:
        return MODEL_PRICING["gpt-realtime"]
    return DEFAULT_PRICING


@dataclass
class SessionStats:
    """Track session statistics for status display."""
    audio_chunks_sent: int = 0
    audio_bytes_sent: int = 0
    silence_chunks_skipped: int = 0
    responses_received: int = 0
    responses_cancelled: int = 0
    speech_detected: int = 0
    connect_time: float = 0
    last_speech_time: float = 0
    input_tokens: int = 0
    output_tokens: int = 0
    audio_input_tokens: int = 0
    audio_output_tokens: int = 0
    text_input_tokens: int = 0
    text_output_tokens: int = 0
    pricing: dict[str, float] = field(default_factory=lambda: DEFAULT_PRICING)

    @property
    def estimated_cost(self) -> float:
        p = self.pricing
        return (
            self.audio_input_tokens * p["audio_input"] / 1_000_000
            + self.text_input_tokens * p["text_input"] / 1_000_000
            + self.audio_output_tokens * p["audio_output"] / 1_000_000
            + self.text_output_tokens * p["text_output"] / 1_000_000
        )

    def status_line(self) -> str:
        parts = ["listening"]
        if self.audio_chunks_sent > 0:
            kb = self.audio_bytes_sent / 1024
            parts.append(f"audio: {kb:.0f}KB")
        if self.speech_detected > 0:
            parts.append(f"speech: {self.speech_detected}×")
        if self.responses_received > 0:
            parts.append(f"translations: {self.responses_received}")
        cost = self.estimated_cost
        if cost > 0:
            parts.append(f"~${cost:.3f}")
        return " · ".join(parts)


def _rms_int16(chunk: bytes) -> float:
    """Compute RMS of int16 PCM chunk."""
    samples = array.array("h")
    samples.frombytes(chunk)
    if not samples:
        return 0.0
    sum_sq = sum(s * s for s in samples)
    return math.sqrt(sum_sq / len(samples))


def _upsample_16_to_24(chunk: bytes) -> bytes:
    """Upsample 16kHz PCM16 mono to 24kHz via linear interpolation (ratio 2:3)."""
    samples = array.array("h")
    samples.frombytes(chunk)
    n = len(samples)
    if n == 0:
        return b""
    out_len = (n * 3 + 1) // 2
    resampled = array.array("h", bytes(out_len * 2))
    for i in range(out_len):
        src = i * 2.0 / 3.0
        idx = int(src)
        if idx + 1 < n:
            frac = src - idx
            val = int(samples[idx] + frac * (samples[idx + 1] - samples[idx]))
        else:
            val = samples[min(idx, n - 1)]
        resampled[i] = max(-32768, min(32767, val))
    return resampled.tobytes()


class RealtimeSession:
    """Manages OpenAI Realtime API WebSocket connection.

    Uses turn_detection=null (manual mode) so that incoming audio never
    causes the API to cancel in-progress translation responses.  Speech
    boundaries are detected client-side via RMS silence gating.
    """

    def __init__(
        self,
        lang: str | None = None,
        deployment: str = "gpt-realtime-mini",
        api_version: str = "2025-04-01-preview",
    ):
        self.lang = lang
        self.deployment = deployment
        self.api_version = api_version
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._connected = False
        self._shutdown = False
        self._buffering = False
        self._flushing = False
        self._audio_buffer: deque[bytes] = deque(maxlen=RECONNECT_BUFFER_MAX)
        self._event_queue: asyncio.Queue[SessionEvent] = asyncio.Queue(maxsize=1000)
        self._send_lock = asyncio.Lock()
        self._last_commit_time: float = 0.0
        self._has_pending_audio: bool = False
        # Client-side VAD state
        self._speech_chunks: int = 0  # consecutive non-silent chunks
        self._silence_chunks: int = 0  # consecutive silent chunks
        self._in_speech: bool = False
        # Track whether a response is in progress to avoid overlap
        self._response_in_progress: bool = False
        self._pending_commit: bool = False  # commit queued while response active
        self.stats = SessionStats(pricing=_resolve_pricing(deployment))

    def _get_url(self) -> str:
        base = os.environ.get("OPENAI_API_BASE", "").rstrip("/")
        if not base:
            raise ValueError(
                "OPENAI_API_BASE not set. Set it to your Azure OpenAI endpoint:\n"
                "export OPENAI_API_BASE='https://your-resource.openai.azure.com/'"
            )
        return (
            f"{base.replace('https://', 'wss://')}"
            f"/openai/realtime"
            f"?api-version={self.api_version}"
            f"&deployment={self.deployment}"
        )

    def _get_key(self) -> str:
        key = os.environ.get("AZURE_OPENAI_API_KEY", "")
        if not key:
            raise ValueError(
                "AZURE_OPENAI_API_KEY not set. Export it or add to .env:\n"
                "export AZURE_OPENAI_API_KEY='your-key'"
            )
        return key

    async def _emit(self, kind: EventKind, data: str = "") -> None:
        """Emit event to TUI, dropping oldest if queue full."""
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
        await self._emit(EventKind.STATUS, "connecting to API...")
        url = self._get_url()
        key = self._get_key()

        try:
            self._ws = await websockets.connect(
                url,
                additional_headers={"api-key": key},
                max_size=2**24,
            )
        except websockets.exceptions.InvalidStatusCode as e:
            status = e.status_code
            if status == 401 or status == 403:
                raise ValueError(
                    f"API key rejected (HTTP {status}). Check AZURE_OPENAI_API_KEY."
                ) from e
            elif status == 429:
                raise ConnectionError(
                    "Rate limited (HTTP 429). Wait and retry, or check Azure quotas."
                ) from e
            raise

        self._connected = True
        self._buffering = False
        self._last_commit_time = time.time()
        self.stats.connect_time = time.time()

        await self._emit(EventKind.STATUS, "configuring session...")

        session_config = {
            "type": "session.update",
            "session": {
                "modalities": ["text"],
                "instructions": build_system_prompt(self.lang),
                "input_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "whisper-1",
                },
                "turn_detection": None,
            },
        }
        await self._ws.send(json.dumps(session_config))
        await self._emit(EventKind.STATUS, "connected · waiting for audio")

    async def disconnect(self) -> None:
        self._connected = False
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    async def send_audio(self, chunk: bytes) -> None:
        """Send 16kHz PCM16 audio, upsampled to 24kHz for OpenAI.

        Also runs client-side VAD: when silence follows speech, commits
        the buffer and requests a translation response.
        """
        rms = _rms_int16(chunk)
        is_speech = rms >= SILENCE_RMS_THRESHOLD

        if is_speech:
            self._silence_chunks = 0
            self._speech_chunks += 1
            if not self._in_speech:
                self._in_speech = True
                self.stats.speech_detected += 1
                self.stats.last_speech_time = time.time()
        else:
            self._silence_chunks += 1
            # Check if speech just ended
            if self._in_speech and self._silence_chunks >= CLIENT_VAD_SILENCE_CHUNKS:
                if self._speech_chunks >= CLIENT_VAD_MIN_SPEECH_CHUNKS:
                    await self._commit_and_respond()
                self._in_speech = False
                self._speech_chunks = 0
            if not self._in_speech:
                self.stats.silence_chunks_skipped += 1
                return

        # Send audio to API
        async with self._send_lock:
            if self._buffering or not self._connected or not self._ws:
                self._audio_buffer.append(chunk)
                return
            try:
                out_data = _upsample_16_to_24(chunk)
                if not out_data:
                    return

                b64 = base64.b64encode(out_data).decode("ascii")
                msg = json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": b64,
                })
                await self._ws.send(msg)
                self._has_pending_audio = True

                self.stats.audio_chunks_sent += 1
                self.stats.audio_bytes_sent += len(out_data)
            except (websockets.exceptions.ConnectionClosed, ConnectionError, OSError):
                self._audio_buffer.append(chunk)
                if not self._shutdown:
                    asyncio.get_running_loop().create_task(self._reconnect())

    async def _commit_and_respond(self) -> None:
        """Commit audio buffer and request a translation response.

        If a response is already in progress, queues the commit so it
        fires as soon as the current response finishes.
        """
        async with self._send_lock:
            if not self._connected or not self._ws or not self._has_pending_audio:
                return
            if self._response_in_progress:
                self._pending_commit = True
                logger.debug("response in progress — queued commit")
                return
            try:
                await self._ws.send(
                    json.dumps({"type": "input_audio_buffer.commit"})
                )
                await self._ws.send(
                    json.dumps({"type": "response.create"})
                )
                self._has_pending_audio = False
                self._response_in_progress = True
                self._last_commit_time = time.time()
                logger.debug("committed audio + requested response")
            except (websockets.exceptions.ConnectionClosed, ConnectionError, OSError):
                pass

    async def _flush_pending_commit(self) -> None:
        """If a commit was queued while response was active, fire it now."""
        if self._pending_commit and self._has_pending_audio:
            self._pending_commit = False
            await self._commit_and_respond()
        else:
            self._pending_commit = False

    async def _flush_buffer(self) -> None:
        """Replay buffered audio after reconnect."""
        if self._flushing:
            return
        self._flushing = True
        try:
            while self._audio_buffer and self._connected and not self._shutdown:
                chunk = self._audio_buffer.popleft()
                try:
                    out_data = _upsample_16_to_24(chunk)
                    if not out_data:
                        continue
                    b64 = base64.b64encode(out_data).decode("ascii")
                    msg = json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": b64,
                    })
                    async with self._send_lock:
                        if self._ws:
                            await self._ws.send(msg)
                    self.stats.audio_chunks_sent += 1
                    self.stats.audio_bytes_sent += len(out_data)
                except (websockets.exceptions.ConnectionClosed, ConnectionError, OSError):
                    break
        finally:
            self._flushing = False

    async def _reconnect(self) -> None:
        if self._shutdown or self._buffering:
            return
        self._buffering = True
        self._connected = False
        await self._emit(EventKind.STATUS, "reconnecting...")
        await self.disconnect()

        max_retries = 10
        delay = BACKOFF_INITIAL
        for attempt in range(1, max_retries + 1):
            if self._shutdown:
                return
            try:
                await self.connect()
                await self._emit(EventKind.STATUS, "reconnected · replaying buffer")
                await self._flush_buffer()
                await self._emit(EventKind.STATUS, self.stats.status_line())
                return
            except ValueError as e:
                await self._emit(EventKind.ERROR, str(e))
                return
            except Exception as e:
                if attempt == max_retries:
                    await self._emit(
                        EventKind.ERROR,
                        f"reconnect failed after {max_retries} attempts — restart langlistn",
                    )
                    return
                await self._emit(
                    EventKind.STATUS,
                    f"reconnect failed ({attempt}/{max_retries}), retry in {delay:.0f}s",
                )
                logger.warning("reconnect attempt %d failed: %s", attempt, e)
                await asyncio.sleep(delay)
                delay = min(delay * 2, BACKOFF_MAX)

    async def force_commit(self) -> None:
        """Force commit the audio buffer to trigger a response."""
        await self._commit_and_respond()

    async def commit_watchdog(self) -> None:
        """Force commit if continuous speech exceeds MAX_SPEECH_DURATION_S."""
        try:
            while not self._shutdown:
                await asyncio.sleep(1.0)
                if not self._connected or not self._has_pending_audio:
                    continue
                elapsed = time.time() - self._last_commit_time
                if elapsed >= MAX_SPEECH_DURATION_S:
                    await self._commit_and_respond()
                    # Reset client VAD state for fresh segment
                    self._speech_chunks = 0
        except asyncio.CancelledError:
            pass

    async def receive_loop(self) -> None:
        """Main receive loop. Parses OpenAI Realtime events."""
        while not self._shutdown:
            if not self._ws:
                await asyncio.sleep(0.1)
                continue
            try:
                async for raw in self._ws:
                    if self._shutdown:
                        break
                    try:
                        event = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    etype = event.get("type", "")

                    if etype == "response.text.delta":
                        delta = event.get("delta", "")
                        if delta:
                            await self._emit(EventKind.TEXT, delta)

                    elif etype == "response.done":
                        resp = event.get("response", {})
                        status = resp.get("status", "")
                        if status == "cancelled":
                            self.stats.responses_cancelled += 1
                            logger.debug("response cancelled by API")
                        else:
                            self.stats.responses_received += 1
                        self._response_in_progress = False
                        self._last_commit_time = time.time()
                        usage = resp.get("usage", {})
                        if usage:
                            self.stats.input_tokens += usage.get("input_tokens", 0)
                            self.stats.output_tokens += usage.get("output_tokens", 0)
                            inp = usage.get("input_token_details", {})
                            out = usage.get("output_token_details", {})
                            self.stats.audio_input_tokens += inp.get("audio_tokens", 0)
                            self.stats.text_input_tokens += inp.get("text_tokens", 0)
                            self.stats.audio_output_tokens += out.get("audio_tokens", 0)
                            self.stats.text_output_tokens += out.get("text_tokens", 0)
                        await self._emit(EventKind.TURN_COMPLETE)
                        await self._emit(EventKind.STATUS, self.stats.status_line())
                        # Fire any queued commit now that response is done
                        await self._flush_pending_commit()

                    elif etype == "conversation.item.input_audio_transcription.completed":
                        transcript = event.get("transcript", "")
                        if transcript:
                            await self._emit(EventKind.TRANSCRIPT, transcript)

                    elif etype == "input_audio_buffer.committed":
                        self._last_commit_time = time.time()
                        self._has_pending_audio = False

                    elif etype in ("session.created", "session.updated"):
                        await self._emit(EventKind.STATUS, "listening")

                    elif etype == "error":
                        err = event.get("error", {})
                        msg = err.get("message", str(err))
                        code = err.get("code", "")
                        if "buffer" in msg.lower() and "small" in msg.lower():
                            logger.debug("API buffer too small (harmless): %s", msg)
                        elif "rate" in code.lower() or "429" in msg:
                            await self._emit(EventKind.ERROR, f"rate limited: {msg}")
                        else:
                            await self._emit(EventKind.ERROR, msg)
                            logger.error("API error: code=%s msg=%s", code, msg)

            except websockets.exceptions.ConnectionClosed as e:
                logger.warning("WebSocket closed: %s", e)
                if not self._shutdown:
                    await self._reconnect()
            except Exception as e:
                logger.error("receive_loop error: %s: %s", type(e).__name__, e)
                if not self._shutdown:
                    await self._reconnect()

    async def get_event(self) -> SessionEvent:
        return await self._event_queue.get()

    async def shutdown(self) -> None:
        self._shutdown = True
        await self.disconnect()
