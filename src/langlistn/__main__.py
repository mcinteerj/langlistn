"""CLI entry point for langlistn — interactive setup or direct args."""

import os

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"
os.environ.setdefault("KMP_WARNINGS", "off")  # suppress OMP deprecation messages


try:
    import multiprocessing.resource_tracker as _rt
    _rt.ensure_running()
except Exception:
    pass

import argparse
import asyncio
import json
import logging
import select
import sys
import termios
import tty
from pathlib import Path

from dotenv import load_dotenv

from . import __version__

_CONFIG_DIR = Path.home() / ".config" / "langlistn"
_RECENT_FILE = _CONFIG_DIR / "recent.json"
_MAX_RECENT = 5


def _load_recent() -> list[dict]:
    """Load recent session configs (newest first)."""
    try:
        data = json.loads(_RECENT_FILE.read_text())
        if isinstance(data, list):
            return data[:_MAX_RECENT]
        # Migrate from old single-session format
        if isinstance(data, dict) and (data.get("app") or data.get("mic")):
            return [data]
    except Exception:
        pass
    # Migrate from old last.json if it exists
    old = _CONFIG_DIR / "last.json"
    try:
        data = json.loads(old.read_text())
        if isinstance(data, dict) and (data.get("app") or data.get("mic")):
            return [data]
    except Exception:
        pass
    return []


def _save_recent(entry: dict):
    """Save a session config, deduplicating and keeping most recent 5."""
    recent = _load_recent()
    # Deduplicate: remove any existing entry with same app/mic/lang/model
    key = (entry.get("app"), entry.get("mic"), entry.get("lang"), entry.get("translate_model"))
    recent = [r for r in recent if (r.get("app"), r.get("mic"), r.get("lang"), r.get("translate_model")) != key]
    recent.insert(0, entry)
    recent = recent[:_MAX_RECENT]
    try:
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        _RECENT_FILE.write_text(json.dumps(recent))
    except Exception:
        pass


def _session_label(cfg: dict) -> str:
    """Human-readable label for a session config."""
    source = cfg.get("app") or ("microphone" if cfg.get("mic") else "?")
    lang = cfg.get("lang") or "auto"
    t_model = cfg.get("translate_model") or "transcribe-only"
    return f"{source} / {lang} / {t_model}"


def _pick_recent_or_new(recent: list[dict]) -> dict | None:
    """Show recent configs + new option. Returns config dict or None for new setup."""
    print("\n🎧 langlistn — real-time audio translation\n")
    print("  Recent:")
    for i, cfg in enumerate(recent):
        print(f"    [{i + 1}] {_session_label(cfg)}")
    print(f"    [n] New configuration\n")

    while True:
        try:
            raw = input("  Choice [1]: ").strip().lower()
            if not raw:
                return recent[0]
            if raw == "n":
                return None
            idx = int(raw) - 1
            if 0 <= idx < len(recent):
                return recent[idx]
        except (ValueError, EOFError):
            pass
        except KeyboardInterrupt:
            raise


def _pick(prompt: str, options: list[str], default: int = 0) -> str:
    """Simple interactive picker. Returns selected option."""
    print(f"\n{prompt}")
    for i, opt in enumerate(options):
        marker = "*" if i == default else " "
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


def _pick_app_simple(apps: list[str]) -> str:
    """Fallback app picker for non-interactive terminals."""
    recent = _load_recent()
    last_app = recent[0].get("app") if recent else None
    if last_app and last_app in apps:
        apps = [last_app] + [a for a in apps if a != last_app]

    print("\n  Choose app:")
    for i, app in enumerate(apps[:15]):
        tag = " (last)" if app == last_app else ""
        print(f"    [{i + 1}] {app}{tag}")
    if len(apps) > 15:
        print(f"    ... {len(apps) - 15} more")
    while True:
        try:
            raw = input("\n  Choice [1]: ").strip()
            if not raw:
                return apps[0]
            idx = int(raw) - 1
            if 0 <= idx < len(apps):
                return apps[idx]
        except (ValueError, EOFError):
            pass


def _pick_app(apps: list[str]) -> str:
    """Type-to-filter app picker with raw-mode keystrokes and last-used memory."""
    recent = _load_recent()
    last_app = recent[0].get("app") if recent else None

    if last_app and last_app in apps:
        apps = [last_app] + [a for a in apps if a != last_app]

    # Guard: fall back for non-interactive stdin
    try:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
    except (OSError, termios.error):
        return _pick_app_simple(apps)

    query = ""
    filtered = apps
    MAX_VISIBLE = 10
    selected_idx = 0

    MOVE_UP = "\033[A"
    CLEAR_LINE = "\033[2K"

    def _draw(lines_to_erase: int) -> int:
        for _ in range(lines_to_erase):
            sys.stdout.write(f"{MOVE_UP}{CLEAR_LINE}\r")

        shown = filtered[:MAX_VISIBLE]
        lines = 0

        for i, app in enumerate(shown):
            marker = "→" if i == selected_idx else " "
            tag = " (last)" if app == last_app and not query else ""
            sys.stdout.write(f"  {marker} [{i + 1}] {app}{tag}\n")
            lines += 1

        if not shown:
            sys.stdout.write("    No matches\n")
            lines += 1

        if len(filtered) > MAX_VISIBLE:
            sys.stdout.write(f"    ... {len(filtered) - MAX_VISIBLE} more\n")
            lines += 1

        sys.stdout.write(f"  Filter: {query}█")
        sys.stdout.flush()
        lines += 1
        return lines

    try:
        sys.stdout.write("\n  Choose app (type to filter, Enter to select):\n")
        drawn = _draw(0)

        tty.setraw(fd)
        while True:
            ch = sys.stdin.read(1)

            if ch in ('\r', '\n'):
                break
            elif ch == '\x03':
                raise KeyboardInterrupt
            elif ch in ('\x7f', '\x08'):
                query = query[:-1]
            elif ch == '\x1b':
                seq = sys.stdin.read(2)
                if seq == '[A':
                    selected_idx = max(0, selected_idx - 1)
                elif seq == '[B':
                    selected_idx = min(len(filtered[:MAX_VISIBLE]) - 1, selected_idx + 1)
                # Restore cooked briefly to redraw
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                drawn = _draw(drawn)
                tty.setraw(fd)
                continue
            elif ch.isdigit():
                # Direct number selection from visible filtered list
                idx = int(ch) - 1
                shown = filtered[:MAX_VISIBLE]
                if 0 <= idx < len(shown):
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                    for _ in range(drawn):
                        sys.stdout.write(f"{MOVE_UP}{CLEAR_LINE}\r")
                    sys.stdout.flush()
                    choice = shown[idx]
                    sys.stdout.write(f"  ✓ {choice}\n")
                    sys.stdout.flush()
                    return choice
                continue
            elif ch.isprintable():
                query += ch
            else:
                continue

            filtered = [a for a in apps if query.lower() in a.lower()] if query else apps
            selected_idx = 0

            # Restore cooked briefly to redraw
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            if len(filtered) == 1:
                # Auto-select single match — erase and break
                for _ in range(drawn):
                    sys.stdout.write(f"{MOVE_UP}{CLEAR_LINE}\r")
                sys.stdout.flush()
                choice = filtered[0]
                sys.stdout.write(f"  ✓ {choice}\n")
                sys.stdout.flush()
                return choice
            drawn = _draw(drawn)
            tty.setraw(fd)

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        # Clean exit: erase picker
        for _ in range(drawn):
            sys.stdout.write(f"{MOVE_UP}{CLEAR_LINE}\r")
        sys.stdout.flush()

    if filtered:
        choice = filtered[min(selected_idx, len(filtered) - 1)]
        sys.stdout.write(f"  ✓ {choice}\n")
        sys.stdout.flush()
        return choice

    return apps[0]


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

        app_name = _pick_app(apps)
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

    # 2. Source language (default from last session)
    recent = _load_recent()
    last = recent[0] if recent else {}
    lang_options = ["Auto-detect"] + [
        f"{code} — {name}" for code, name in sorted(LANGUAGE_MAP.items())
    ]
    lang_default = 0
    if last.get("lang"):
        for i, opt in enumerate(lang_options):
            if opt.startswith(f"{last['lang']} — "):
                lang_default = i
                break
    lang_choice = _pick("Source language:", lang_options, default=lang_default)
    lang = None
    if lang_choice != "Auto-detect":
        lang = lang_choice.split(" — ")[0]

    # 3. Translation model (default from last session)
    model_options = ["haiku (fastest, ~$0.30/hr)", "sonnet (better, ~$1/hr)", "opus (best, ~$5/hr)", "none (transcription only)"]
    model_default = 0
    if last.get("translate_model"):
        for i, opt in enumerate(model_options):
            if opt.startswith(last["translate_model"]):
                model_default = i
                break
    model_choice = _pick("Translation model:", model_options, default=model_default)
    translate_model = None
    no_translate = False
    if "none" in model_choice:
        no_translate = True
    else:
        translate_model = model_choice.split(" ")[0]

    result = {
        "app_name": app_name,
        "mic": mic,
        "device": device,
        "lang": lang,
        "translate_model": translate_model,
        "no_translate": no_translate,
    }

    # Remember for next time
    _save_recent({
        "app": app_name,
        "mic": mic,
        "device": device,
        "lang": lang,
        "translate_model": translate_model,
    })

    # Print equivalent command for future use
    cmd_parts = ["langlistn"]
    if app_name:
        cmd_parts.append(f'--app "{app_name}"')
    elif mic:
        cmd_parts.append("--mic")
    if device:
        cmd_parts.append(f'--device "{device}"')
    if lang:
        cmd_parts.append(f"--source {lang}")
    if no_translate:
        cmd_parts.append("--no-translate")
    elif translate_model and translate_model != "haiku":
        cmd_parts.append(f"--translate-model {translate_model}")
    print(f"\n  💡 Next time, run directly:\n     {' '.join(cmd_parts)}\n")

    return result


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
            recent = _load_recent()
            if recent:
                chosen = _pick_recent_or_new(recent)
                if chosen is not None:
                    # Re-save to bump to top of recent list
                    _save_recent(chosen)
                    kwargs = {
                        "app_name": chosen.get("app"),
                        "mic": chosen.get("mic", False),
                        "device": chosen.get("device"),
                        "lang": chosen.get("lang"),
                        "translate_model": chosen.get("translate_model"),
                        "no_translate": chosen.get("translate_model") is None,
                    }
                else:
                    kwargs = _interactive_setup()
            else:
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
