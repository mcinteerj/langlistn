"""CLI entry point for langlistn."""

import os

# Prevent tqdm/huggingface from spawning multiprocessing resource tracker
# which fails inside Textual due to non-inheritable FDs.
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"

# Pre-start the multiprocessing resource tracker BEFORE Textual takes over
# the terminal FDs. This ensures it's already running when tqdm needs it.
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
        description="Real-time audio translation to English.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # Translate Korean YouTube video in Chrome
  langlistn --app "Google Chrome"

  # Hint the source language for better accuracy
  langlistn --app "Google Chrome" --source ko

  # Translate Teams meeting
  langlistn --app "Microsoft Teams"

  # Translate from microphone
  langlistn --mic

  # List available audio sources
  langlistn --list-apps
  langlistn --list-devices

  # Show source-language transcript alongside translation
  langlistn --app "Google Chrome" --transcript

environment variables:
  AZURE_OPENAI_API_KEY     Azure OpenAI API key
  OPENAI_API_BASE          Azure OpenAI endpoint URL
""",
    )

    # Audio source (mutually exclusive)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--app", metavar="NAME", help="Capture audio from named app")
    source.add_argument("--mic", action="store_true", help="Capture from microphone")

    # Discovery
    parser.add_argument("--list-apps", action="store_true", help="List capturable apps and exit")
    parser.add_argument("--list-devices", action="store_true", help="List audio input devices and exit")

    # Options
    parser.add_argument("--device", metavar="NAME", help="Microphone device name (with --mic)")
    parser.add_argument("--source", metavar="CODE", dest="lang",
                        help="Source language hint (ISO 639-1, e.g. ko, ja, zh). Auto-detected if omitted.")
    parser.add_argument("--transcript", action="store_true", help="Show source-language transcript")
    parser.add_argument("--log", metavar="FILE", help="Save translations to file")
    parser.add_argument("--deployment", metavar="NAME", default="gpt-realtime-mini",
                        help="Azure OpenAI deployment name (default: gpt-realtime-mini)")
    parser.add_argument("--local", action="store_true",
                        help="Use local Whisper instead of OpenAI Realtime API (free, offline)")
    parser.add_argument("--model", metavar="NAME", default="mlx-community/whisper-large-v3-mlx",
                        help="Whisper model for --local mode (default: mlx-community/whisper-large-v3-mlx)")
    parser.add_argument("--tui", action="store_true",
                        help="Use full TUI instead of plain terminal output (with --local)")
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

    # Validate source
    if not args.app and not args.mic:
        parser.error("Must specify --app NAME or --mic (see --list-apps / --list-devices)")

    from .app import run_app

    try:
        if args.local and not args.tui:
            # Plain terminal output — no Textual, no FD issues
            from .cli_output import run_cli
            asyncio.run(
                run_cli(
                    app_name=args.app,
                    mic=args.mic,
                    device=args.device,
                    lang=args.lang,
                    model=args.model,
                    log_path=args.log,
                )
            )
        else:
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
                    local=args.local,
                    model=args.model,
                )
            )
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
