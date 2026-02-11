"""Audio capture from app via Swift ScreenCaptureKit helper."""

import asyncio
import signal
from pathlib import Path

from ..config import CHUNK_FRAMES, SAMPLE_RATE

# Resolve Swift binary relative to package root
_PKG_ROOT = Path(__file__).resolve().parent.parent.parent.parent
HELPER_BIN = _PKG_ROOT / "swift" / ".build" / "release" / "AudioCaptureHelper"


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
        self._chunk_bytes = CHUNK_FRAMES * 2  # 16-bit = 2 bytes per sample

    async def start(self) -> None:
        self._proc = await asyncio.create_subprocess_exec(
            str(HELPER_BIN),
            "--app", self.app_name,
            "--rate", str(SAMPLE_RATE),
            "--channels", "1",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    async def read_chunk(self) -> bytes | None:
        """Read one chunk of PCM audio. Returns None on EOF/error."""
        if not self._proc or not self._proc.stdout:
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
        stderr_bytes = b""
        try:
            self._proc.send_signal(signal.SIGTERM)
            if self._proc.stderr:
                stderr_bytes = await asyncio.wait_for(
                    self._proc.stderr.read(), timeout=3.0
                )
            await asyncio.wait_for(self._proc.wait(), timeout=5.0)
        except (asyncio.TimeoutError, ProcessLookupError):
            self._proc.kill()
            await self._proc.wait()
        rc = self._proc.returncode or 0
        stderr = stderr_bytes.decode(errors="replace") if stderr_bytes else ""
        self._proc = None
        return rc, stderr

    @property
    def returncode(self) -> int | None:
        return self._proc.returncode if self._proc else None
