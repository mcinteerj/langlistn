"""Audio capture from app via Swift ScreenCaptureKit helper."""

from __future__ import annotations

import asyncio
import atexit
import logging
import signal
from pathlib import Path
from typing import Protocol, runtime_checkable

from ..config import CHUNK_FRAMES, SAMPLE_RATE

logger = logging.getLogger(__name__)


@runtime_checkable
class AudioSource(Protocol):
    """Common interface for audio capture backends."""

    async def start(self) -> None: ...
    async def read_chunk(self) -> bytes | None: ...
    async def stop(self) -> tuple[int, str]: ...

    @property
    def returncode(self) -> int | None: ...

# Resolve Swift binary relative to package root
_PKG_ROOT = Path(__file__).resolve().parent.parent.parent.parent
HELPER_BIN = _PKG_ROOT / "swift" / ".build" / "release" / "AudioCaptureHelper"

# Track live helper PIDs for cleanup on unexpected exit
_live_helper_pids: set[int] = set()


def _cleanup_helpers() -> None:
    """Kill any orphaned helper processes on interpreter exit."""
    for pid in list(_live_helper_pids):
        try:
            import os as _os
            _os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass
    _live_helper_pids.clear()


atexit.register(_cleanup_helpers)


async def list_apps() -> list[str]:
    """List capturable app names."""
    proc = await asyncio.create_subprocess_exec(
        str(HELPER_BIN), "--list-apps",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"list-apps failed: {stderr.decode().strip()}")
    return [line for line in stdout.decode().splitlines() if line.strip()]


class AppCapture:
    """Captures audio from a named macOS app via Swift helper subprocess."""

    def __init__(self, app_name: str):
        self.app_name = app_name
        self._proc: asyncio.subprocess.Process | None = None
        self._stderr_task: asyncio.Task | None = None
        self._last_stderr: str = ""
        self._chunk_bytes = CHUNK_FRAMES * 2  # 16-bit = 2 bytes per sample

    async def start(self) -> None:
        if not HELPER_BIN.exists():
            raise RuntimeError(
                f"Swift helper not found at {HELPER_BIN}\n"
                "Build it first: bash swift/build.sh"
            )

        self._proc = await asyncio.create_subprocess_exec(
            str(HELPER_BIN),
            "--app", self.app_name,
            "--rate", str(SAMPLE_RATE),
            "--channels", "1",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        if self._proc.pid:
            _live_helper_pids.add(self._proc.pid)

        # Start streaming stderr in background
        self._stderr_task = asyncio.create_task(self._read_stderr())

        # Brief check for immediate crash (e.g., app not found)
        await asyncio.sleep(0.2)
        if self._proc.returncode is not None:
            stderr = self._last_stderr or "unknown error"
            raise RuntimeError(f"Audio capture failed: {stderr.strip()}")

    async def _read_stderr(self) -> None:
        """Read helper stderr lines and log them."""
        if not self._proc or not self._proc.stderr:
            return
        try:
            while True:
                line = await self._proc.stderr.readline()
                if not line:
                    break
                text = line.decode(errors="replace").rstrip()
                self._last_stderr = text
                if text.startswith("Error:"):
                    logger.error("helper: %s", text)
                else:
                    logger.info("helper: %s", text)
        except (asyncio.CancelledError, Exception):
            pass

    async def read_chunk(self) -> bytes | None:
        """Read one chunk of PCM audio. Returns None on EOF/error."""
        if not self._proc or not self._proc.stdout:
            return None
        # Detect crash between reads
        if self._proc.returncode is not None:
            return None
        try:
            data = await self._proc.stdout.readexactly(self._chunk_bytes)
            return data
        except (asyncio.IncompleteReadError, ConnectionError):
            return None

    async def stop(self) -> tuple[int, str]:
        """Kill helper and return (exit_code, stderr)."""
        if not self._proc:
            return 0, ""
        pid = self._proc.pid
        try:
            self._proc.send_signal(signal.SIGTERM)
            await asyncio.wait_for(self._proc.wait(), timeout=5.0)
        except (asyncio.TimeoutError, ProcessLookupError):
            try:
                self._proc.kill()
                await self._proc.wait()
            except ProcessLookupError:
                pass
        if self._stderr_task:
            self._stderr_task.cancel()
            try:
                await self._stderr_task
            except asyncio.CancelledError:
                pass
        rc = self._proc.returncode or 0
        stderr = self._last_stderr
        self._proc = None
        if pid:
            _live_helper_pids.discard(pid)
        return rc, stderr

    @property
    def returncode(self) -> int | None:
        return self._proc.returncode if self._proc else None
