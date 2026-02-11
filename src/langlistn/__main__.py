"""CLI entry point for langlistn."""

import os

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"

try:
    import multiprocessing.resource_tracker as _rt
    _rt.ensure_running()
except Exception:
    pass

import argparse
import asyncio
import json

from dotenv import load_dotenv

from . import __version__


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        prog="langlistn",
        description="Real-time audio translation to English. Runs locally with Whisper by default — free, offline, no API key needed.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # Translate app audio (local Whisper, default)
  langlistn --app "Google Chrome"
  langlistn --app "zoom.us" --source ko
  langlistn --app "Microsoft Teams" --dual-lang

  # Microphone
  langlistn --mic
  langlistn --mic --source ja

  # Pipe-friendly (no ANSI, confirmed text only)
  langlistn --app "Google Chrome" --plain | tee meeting.txt

  # Azure OpenAI Realtime API (streaming, costs money)
  langlistn --app "Google Chrome" --remote

  # Discovery
  langlistn --list-apps
  langlistn --list-devices
""",
    )

    # Audio source
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--app", metavar="NAME", help="Capture audio from named app")
    source.add_argument("--mic", action="store_true", help="Capture from microphone")

    # Discovery
    parser.add_argument("--list-apps", action="store_true", help="List capturable apps and exit")
    parser.add_argument("--list-devices", action="store_true", help="List audio input devices and exit")

    # Common options
    parser.add_argument("--device", metavar="NAME", help="Microphone device name (with --mic)")
    parser.add_argument("--source", metavar="CODE", dest="lang",
                        help="Source language hint (ko, ja, zh, fr, de, etc). Improves speed and accuracy.")
    parser.add_argument("--log", metavar="FILE", help="Save translations to file")

    # Local whisper options (default mode)
    local_group = parser.add_argument_group("local whisper (default)")
    local_group.add_argument("--model", metavar="NAME", default=None,
                             help="Whisper model (default: auto-select based on RAM)")
    local_group.add_argument("--dual-lang", action="store_true",
                             help="Show original language above English translation")
    local_group.add_argument("--plain", action="store_true",
                             help="Pipe-friendly — no ANSI, no speculative, just confirmed text")

    # Remote API options
    remote_group = parser.add_argument_group("remote API (--remote)")
    remote_group.add_argument("--remote", action="store_true",
                              help="Use Azure OpenAI Realtime API instead of local Whisper")
    remote_group.add_argument("--deployment", metavar="NAME", default="gpt-realtime-mini",
                              help="Azure deployment name (default: gpt-realtime-mini)")
    remote_group.add_argument("--transcript", action="store_true",
                              help="Show source-language transcript (remote only)")
    remote_group.add_argument("--tui", action="store_true",
                              help="Use full TUI (remote default, optional for local)")

    # Output
    parser.add_argument("--json", dest="output_json", action="store_true", help="JSON output (for --list-*)")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    args = parser.parse_args()

    # Discovery commands
    if args.list_apps:
        from .audio import list_apps
        apps = asyncio.run(list_apps())
        if args.output_json:
            print(json.dumps(apps))
        else:
            for a in apps:
                print(a)
        return

    if args.list_devices:
        from .audio.mic_capture import list_devices
        devices = list_devices()
        if args.output_json:
            print(json.dumps(devices))
        else:
            for d in devices:
                print(f"  [{d['index']}] {d['name']} ({d['channels']}ch)")
        return

    if not args.app and not args.mic:
        parser.error("Must specify --app NAME or --mic (see --list-apps / --list-devices)")

    try:
        if args.remote or args.tui:
            # Remote API mode (or TUI mode for local)
            from .app import run_app
            asyncio.run(
                run_app(
                    app_name=args.app,
                    mic=args.mic,
                    device=args.device,
                    lang=args.lang,
                    deployment=args.deployment,
                    log_path=args.log,
                    show_transcript=args.transcript,
                    local=not args.remote,
                    model=args.model or "mlx-community/whisper-large-v3-mlx",
                )
            )
        else:
            # Local whisper with terminal output (default)
            from .cli_output import run_cli
            asyncio.run(
                run_cli(
                    app_name=args.app,
                    mic=args.mic,
                    device=args.device,
                    lang=args.lang,
                    model=args.model,
                    log_path=args.log,
                    plain=args.plain,
                    dual_lang=args.dual_lang,
                )
            )
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
