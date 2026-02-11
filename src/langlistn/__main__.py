"""CLI entry point for langlistn."""

import argparse
import asyncio
import json

from . import __version__


def main():
    parser = argparse.ArgumentParser(
        prog="langlistn",
        description="Real-time audio translation and transcription to English.",
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
  OPENAI_API_BASE          Azure OpenAI endpoint URL (optional override)
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
    parser.add_argument("--target", metavar="CODE", default="en",
                        help="Target language (default: en). Currently only English supported.")
    parser.add_argument("--transcript", action="store_true", help="Show source-language transcript")
    parser.add_argument("--log", metavar="FILE", help="Save translations to file")
    parser.add_argument("--deployment", metavar="NAME", default="gpt-realtime",
                        help="Azure OpenAI deployment name (default: gpt-realtime)")
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
        asyncio.run(
            run_app(
                app_name=args.app,
                mic=args.mic,
                device=args.device,
                lang=args.lang,
                deployment=args.deployment,
                log_path=args.log,
                show_transcript=args.transcript,
            )
        )
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
