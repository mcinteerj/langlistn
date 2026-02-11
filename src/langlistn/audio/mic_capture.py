"""Audio capture from microphone via sounddevice."""

import asyncio
import logging
import queue

import sounddevice as sd

from ..config import CHUNK_FRAMES, SAMPLE_RATE

logger = logging.getLogger(__name__)


def list_devices() -> list[dict]:
    """List available input audio devices."""
    devices = sd.query_devices()
    result = []
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            result.append({
                "index": i,
                "name": d["name"],
                "channels": d["max_input_channels"],
                "default_sr": d["default_samplerate"],
            })
    return result


class MicCapture:
    """Captures audio from microphone using sounddevice."""

    def __init__(self, device: str | int | None = None):
        self._device = device
        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=200)
        self._stream: sd.RawInputStream | None = None
        self._drop_count: int = 0

    def _callback(self, indata: bytes, frames: int, time_info, status) -> None:
        if status:
            logger.warning("mic audio status: %s", status)
        try:
            self._queue.put_nowait(bytes(indata))
        except queue.Full:
            self._drop_count += 1
            if self._drop_count % 50 == 1:
                logger.warning("mic queue full — dropped %d frames", self._drop_count)

    async def start(self) -> None:
        # Drain any stale data from previous session
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._drop_count = 0

        try:
            self._stream = sd.RawInputStream(
                samplerate=SAMPLE_RATE,
                blocksize=CHUNK_FRAMES,
                device=self._device,
                dtype="int16",
                channels=1,
                callback=self._callback,
            )
            self._stream.start()
        except sd.PortAudioError as e:
            raise RuntimeError(
                f"Failed to open microphone: {e}\n"
                "Check that the device exists (--list-devices) and your terminal "
                "has microphone permission in System Settings → Privacy & Security."
            ) from e

    async def read_chunk(self) -> bytes | None:
        """Read one chunk. Non-blocking poll to avoid holding executor threads.

        Returns None only when the stream has died (device disconnected).
        Returns empty bytes b'' when no audio is available yet.
        """
        if self._stream and not self._stream.active:
            logger.warning("mic stream is no longer active (device disconnected?)")
            return None
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.05)  # yield to event loop
            return b""

    async def stop(self) -> tuple[int, str]:
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        return 0, ""

    @property
    def returncode(self) -> int | None:
        """Compatibility with AppCapture interface. No subprocess."""
        return None
