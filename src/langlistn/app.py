"""Main async orchestrator — ties audio, session (API or local), and TUI together."""

from __future__ import annotations

import asyncio

from .audio import AppCapture, AudioSource
from .audio.mic_capture import MicCapture
from .tui import TranslateApp


async def run_app(
    app_name: str | None = None,
    mic: bool = False,
    device: str | None = None,
    lang: str | None = None,
    deployment: str = "gpt-realtime-mini",
    log_path: str | None = None,
    show_transcript: bool = False,
    local: bool = False,
    model: str = "mlx-community/whisper-large-v3-mlx",
) -> None:
    from .config import resolve_language_name

    lang_name = resolve_language_name(lang)

    # Build audio source
    source: AudioSource
    if app_name:
        source = AppCapture(app_name)
        mode = f"app: {app_name}"
    elif mic:
        source = MicCapture(device=device)
        mode = "mic"
    else:
        raise ValueError("Must specify --app or --mic")

    # Build session — local Whisper or OpenAI Realtime
    if local:
        from .whisper_local import LocalWhisperSession
        session = LocalWhisperSession(lang=lang, model=model)
        mode += " · local whisper"
    else:
        from .realtime import RealtimeSession
        session = RealtimeSession(lang=lang, deployment=deployment)

    # Build TUI
    tui = TranslateApp(
        source_lang=lang_name or lang,
        source_code=lang,
        mode=mode,
        log_path=log_path,
        session=session,
        audio_source=source,
    )
    if show_transcript:
        tui._show_original = True

    await tui.run_async()
