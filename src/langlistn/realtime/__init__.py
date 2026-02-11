"""OpenAI Realtime API session — audio in, text out."""

import array
import asyncio
import base64
import json
import os
import struct
import time
from collections import deque
from dataclasses import dataclass, field

import websockets

from ..config import (
    BACKOFF_INITIAL,
    BACKOFF_MAX,
    RECONNECT_BUFFER_MAX,
    build_system_prompt,
)


@dataclass
class SessionEvent:
    """Event emitted to the TUI."""
    kind: str  # "text", "turn_complete", "transcript", "status", "error"
    data: str = ""


@dataclass
class SessionStats:
    """Track session statistics for status display."""
    audio_chunks_sent: int = 0
    audio_bytes_sent: int = 0
    responses_received: int = 0
    speech_detected: int = 0
    connect_time: float = 0
    last_speech_time: float = 0
    # Token tracking for cost estimation
    input_tokens: int = 0
    output_tokens: int = 0
    audio_input_tokens: int = 0
    audio_output_tokens: int = 0
    text_input_tokens: int = 0
    text_output_tokens: int = 0

    # Pricing per 1M tokens (gpt-4o-realtime, as of 2025)
    # Audio: ~100 tokens/sec, so $0.06/min ≈ $10/1M tokens
    AUDIO_INPUT_PRICE_PER_M: float = 10.0
    TEXT_INPUT_PRICE_PER_M: float = 5.0
    AUDIO_OUTPUT_PRICE_PER_M: float = 40.0
    TEXT_OUTPUT_PRICE_PER_M: float = 20.0

    @property
    def estimated_cost(self) -> float:
        return (
            self.audio_input_tokens * self.AUDIO_INPUT_PRICE_PER_M / 1_000_000
            + self.text_input_tokens * self.TEXT_INPUT_PRICE_PER_M / 1_000_000
            + self.audio_output_tokens * self.AUDIO_OUTPUT_PRICE_PER_M / 1_000_000
            + self.text_output_tokens * self.TEXT_OUTPUT_PRICE_PER_M / 1_000_000
        )

    def status_line(self) -> str:
        parts = ["listening"]
        if self.audio_chunks_sent > 0:
            kb = self.audio_bytes_sent / 1024
            parts.append(f"audio: {kb:.0f}KB sent")
        if self.speech_detected > 0:
            parts.append(f"speech: {self.speech_detected}×")
        if self.responses_received > 0:
            parts.append(f"translations: {self.responses_received}")
        cost = self.estimated_cost
        if cost > 0:
            parts.append(f"~${cost:.3f}")
        return " · ".join(parts)


class RealtimeSession:
    """Manages OpenAI Realtime API WebSocket connection."""

    def __init__(
        self,
        lang: str | None = None,
        deployment: str = "gpt-realtime",
        api_version: str = "2025-04-01-preview",
    ):
        self.lang = lang
        self.deployment = deployment
        self.api_version = api_version
        self._ws = None
        self._connected = False
        self._shutdown = False
        self._buffering = False
        self._audio_buffer: deque[bytes] = deque(maxlen=RECONNECT_BUFFER_MAX)
        self._event_queue: asyncio.Queue[SessionEvent] = asyncio.Queue()
        self._send_lock = asyncio.Lock()
        self.stats = SessionStats()

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

    async def connect(self) -> None:
        await self._event_queue.put(SessionEvent("status", "connecting to API..."))
        url = self._get_url()
        key = self._get_key()
        self._ws = await websockets.connect(
            url,
            additional_headers={"api-key": key},
            max_size=2**24,
        )
        self._connected = True
        self._buffering = False
        self.stats.connect_time = time.time()

        await self._event_queue.put(SessionEvent("status", "configuring session..."))

        session_config = {
            "type": "session.update",
            "session": {
                "modalities": ["text"],
                "instructions": build_system_prompt(self.lang),
                "input_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "whisper-1",
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.3,
                    "prefix_padding_ms": 200,
                    "silence_duration_ms": 300,
                },
            },
        }
        await self._ws.send(json.dumps(session_config))
        await self._event_queue.put(SessionEvent("status", "connected · waiting for audio"))

    async def disconnect(self) -> None:
        self._connected = False
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    async def send_audio(self, chunk: bytes) -> None:
        """Send 16kHz PCM16 audio, upsampled to 24kHz for OpenAI."""
        if self._buffering or not self._connected or not self._ws:
            self._audio_buffer.append(chunk)
            return
        try:
            # Fast 16kHz→24kHz upsample (ratio 2:3) using array module
            samples = array.array('h')
            samples.frombytes(chunk)
            n = len(samples)
            out_len = int(n * 1.5)
            resampled = array.array('h', bytes(out_len * 2))
            for i in range(out_len):
                src = i * 2 / 3  # inverse of 1.5x
                idx = int(src)
                if idx + 1 < n:
                    frac = src - idx
                    val = int(samples[idx] + frac * (samples[idx + 1] - samples[idx]))
                else:
                    val = samples[-1]
                resampled[i] = max(-32768, min(32767, val))
            out_data = resampled.tobytes()

            b64 = base64.b64encode(out_data).decode("ascii")
            msg = json.dumps({
                "type": "input_audio_buffer.append",
                "audio": b64,
            })
            async with self._send_lock:
                await self._ws.send(msg)

            self.stats.audio_chunks_sent += 1
            self.stats.audio_bytes_sent += len(out_data)
        except (websockets.exceptions.ConnectionClosed, ConnectionError, OSError):
            self._audio_buffer.append(chunk)
            if not self._shutdown:
                await self._reconnect()

    async def _flush_buffer(self) -> None:
        while self._audio_buffer and self._connected:
            chunk = self._audio_buffer.popleft()
            await self.send_audio(chunk)

    async def _reconnect(self) -> None:
        if self._shutdown:
            return
        self._buffering = True
        self._connected = False
        await self._event_queue.put(SessionEvent("status", "reconnecting..."))
        await self.disconnect()

        max_retries = 10
        delay = BACKOFF_INITIAL
        for attempt in range(1, max_retries + 1):
            if self._shutdown:
                return
            try:
                await self.connect()
                await self._flush_buffer()
                return
            except Exception:
                if attempt == max_retries:
                    await self._event_queue.put(
                        SessionEvent("error", f"reconnect failed after {max_retries} attempts")
                    )
                    return
                await self._event_queue.put(
                    SessionEvent("status", f"reconnect failed ({attempt}/{max_retries}), retry in {delay:.0f}s")
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, BACKOFF_MAX)

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

                    # Streaming text delta from model
                    if etype == "response.text.delta":
                        delta = event.get("delta", "")
                        if delta:
                            await self._event_queue.put(
                                SessionEvent("text", delta)
                            )

                    # Response done
                    elif etype == "response.done":
                        self.stats.responses_received += 1
                        # Extract token usage for cost tracking
                        resp = event.get("response", {})
                        usage = resp.get("usage", {})
                        if usage:
                            self.stats.input_tokens += usage.get("input_tokens", 0)
                            self.stats.output_tokens += usage.get("output_tokens", 0)
                            inp_detail = usage.get("input_token_details", {})
                            out_detail = usage.get("output_token_details", {})
                            self.stats.audio_input_tokens += inp_detail.get("audio_tokens", 0)
                            self.stats.text_input_tokens += inp_detail.get("text_tokens", 0)
                            self.stats.audio_output_tokens += out_detail.get("audio_tokens", 0)
                            self.stats.text_output_tokens += out_detail.get("text_tokens", 0)
                        await self._event_queue.put(
                            SessionEvent("turn_complete")
                        )
                        await self._event_queue.put(
                            SessionEvent("status", self.stats.status_line())
                        )

                    # Input audio transcription
                    elif etype == "conversation.item.input_audio_transcription.completed":
                        transcript = event.get("transcript", "")
                        if transcript:
                            await self._event_queue.put(
                                SessionEvent("transcript", transcript)
                            )

                    # Speech detection
                    elif etype == "input_audio_buffer.speech_started":
                        self.stats.speech_detected += 1
                        self.stats.last_speech_time = time.time()

                    # Session created/updated
                    elif etype in ("session.created", "session.updated"):
                        await self._event_queue.put(
                            SessionEvent("status", "listening")
                        )

                    # Errors
                    elif etype == "error":
                        err = event.get("error", {})
                        msg = err.get("message", str(err))
                        await self._event_queue.put(
                            SessionEvent("error", msg)
                        )

            except websockets.exceptions.ConnectionClosed:
                if not self._shutdown:
                    await self._reconnect()
            except Exception:
                if not self._shutdown:
                    await self._reconnect()

    async def get_event(self) -> SessionEvent:
        return await self._event_queue.get()

    async def shutdown(self) -> None:
        self._shutdown = True
        await self.disconnect()
