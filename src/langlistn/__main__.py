"""CLI entry point for langlistn — interactive setup or direct args."""

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
import logging
import sys

from dotenv import load_dotenv

from . import __version__


def _pick(prompt: str, options: list[str], default: int = 0) -> str:
    """Simple interactive picker. Returns selected option."""
    print(f"\n{prompt}")
    for i, opt in enumerate(options):
        marker = "→" if i == default else " "
        print(f"  {marker} [{i + 1}] {opt}")
    while True:
        try:
            raw = input(f"\nChoice [{default + 1}]: ").strip()
            if not raw:
                return options[default]
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return options[idx]
        except (ValueError, EOFError):
            pass
        print(f"  Enter 1-{len(options)}")


def _interactive_setup() -> dict:
    """Interactive setup wizard. Returns kwargs for run_pipeline."""
    from .config import LANGUAGE_MAP

    print("\n🎧 langlistn — real-time audio translation\n")

    # 1. Source type
    source_type = _pick("Audio source:", ["App audio", "Microphone"])

    app_name = None
    mic = False
    device = None

    if source_type == "App audio":
        # List running apps
        from .audio import list_apps
        try:
            apps = asyncio.run(list_apps())
        except Exception as e:
            print(f"\n  Error listing apps: {e}")
            sys.exit(1)

        if not apps:
            print("\n  No capturable apps found. Make sure an app is running and playing audio.")
            sys.exit(1)

        app_name = _pick("Choose app:", apps)
    else:
        mic = True
        from .audio.mic_capture import list_devices
        devices = list_devices()
        if devices:
            names = [f"{d['name']} ({d['channels']}ch)" for d in devices]
            names.insert(0, "Default microphone")
            choice = _pick("Choose microphone:", names)
            if choice != "Default microphone":
                idx = names.index(choice) - 1
                device = devices[idx]["name"]

    # 2. Source language
    lang_options = ["Auto-detect"] + [
        f"{code} — {name}" for code, name in sorted(LANGUAGE_MAP.items())
    ]
    lang_choice = _pick("Source language:", lang_options)
    lang = None
    if lang_choice != "Auto-detect":
        lang = lang_choice.split(" — ")[0]

    # 3. Translation model
    model_choice = _pick(
        "Translation model:",
        ["haiku (fastest, ~$0.30/hr)", "sonnet (better, ~$1/hr)", "opus (best, ~$5/hr)", "none (transcription only)"],
    )
    translate_model = None
    no_translate = False
    if "none" in model_choice:
        no_translate = True
    else:
        translate_model = model_choice.split(" ")[0]

    return {
        "app_name": app_name,
        "mic": mic,
        "device": device,
        "lang": lang,
        "translate_model": translate_model,
        "no_translate": no_translate,
    }


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        prog="langlistn",
        description="Real-time audio translation. Run without args for interactive setup.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  langlistn                                    # interactive setup
  langlistn --app "Google Chrome" --source ko  # direct
  langlistn --mic --source ja
  langlistn --list-apps
""",
    )

    source = parser.add_mutually_exclusive_group()
    source.add_argument("--app", metavar="NAME", help="Capture audio from named app")
    source.add_argument("--mic", action="store_true", help="Capture from microphone")

    parser.add_argument("--list-apps", action="store_true", help="List capturable apps")
    parser.add_argument("--list-devices", action="store_true", help="List audio input devices")
    parser.add_argument("--device", metavar="NAME", help="Microphone device name")
    parser.add_argument("--source", metavar="CODE", dest="lang", help="Source language (ko, ja, zh, etc)")
    parser.add_argument("--log", metavar="FILE", help="Save to file")
    parser.add_argument("--model", metavar="NAME", default=None, help="Whisper model override")
    parser.add_argument("--plain", action="store_true", help="Pipe-friendly output")
    parser.add_argument("--translate-model", metavar="TIER", default="haiku",
                        choices=["haiku", "sonnet", "opus"], help="Claude model tier")
    parser.add_argument("--no-translate", action="store_true", help="Transcribe only")
    parser.add_argument("--json", dest="output_json", action="store_true", help="JSON output for --list-*")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    # Pipeline tunables
    parser.add_argument("--max-context", type=int, default=2000, help="Max context chars for LLM")
    parser.add_argument("--silence-reset", type=float, default=10.0, help="Seconds of silence before context reset")
    parser.add_argument("--force-confirm", type=int, default=3, help="Force-lock translation after N unstable cycles")
    parser.add_argument("--debug-log", metavar="FILE", default=None, help="Write debug log to file")

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

    # Debug logging to file
    if args.debug_log:
        handler = logging.FileHandler(args.debug_log, mode="w", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S"))
        logging.getLogger("langlistn").setLevel(logging.DEBUG)
        logging.getLogger("langlistn").addHandler(handler)

    # Interactive mode if no source specified
    if not args.app and not args.mic:
        try:
            kwargs = _interactive_setup()
        except KeyboardInterrupt:
            print("\n")
            return
        # Merge with any CLI overrides
        if args.model:
            kwargs["model"] = args.model
        if args.log:
            kwargs["log_path"] = args.log
        kwargs["plain"] = args.plain
    else:
        kwargs = {
            "app_name": args.app,
            "mic": args.mic,
            "device": args.device,
            "lang": args.lang,
            "translate_model": args.translate_model,
            "no_translate": args.no_translate,
            "model": args.model,
            "log_path": args.log,
            "plain": args.plain,
        }

    # Build audio source
    from .audio import AppCapture
    from .audio.mic_capture import MicCapture
    from .pipeline import PipelineConfig, run_pipeline

    app_name = kwargs.pop("app_name", None)
    mic = kwargs.pop("mic", False)
    device = kwargs.pop("device", None)

    if app_name:
        audio_source = AppCapture(app_name)
        mode = f"app: {app_name}"
    elif mic:
        audio_source = MicCapture(device=device)
        mode = "mic"
    else:
        parser.error("Must specify --app or --mic")
        return

    pipeline_config = PipelineConfig(
        max_context_chars=args.max_context,
        silence_reset_seconds=args.silence_reset,
        force_confirm_after=args.force_confirm,
    )

    try:
        asyncio.run(
            run_pipeline(
                source=audio_source,
                mode=mode,
                lang=kwargs.get("lang"),
                model=kwargs.get("model"),
                translate_model=kwargs.get("translate_model"),
                no_translate=kwargs.get("no_translate", False),
                plain=kwargs.get("plain", False),
                log_path=kwargs.get("log_path"),
                config=pipeline_config,
            )
        )
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
