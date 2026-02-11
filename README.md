# langlistn

Real-time audio translation to English, running locally on your Mac. Point it at any app or microphone and get live English text — no API keys, no cloud, no cost.

Powered by [Whisper](https://github.com/openai/whisper) via [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) on Apple Silicon.

> **macOS 15+** · **Apple Silicon (M1+)** · **Python 3.11+**

## How it works

```
App/Mic audio → Swift helper (ScreenCaptureKit) → 16kHz PCM
    → sliding window (20s, 5s step) → Whisper large-v3 (local)
    → confirmed/speculative display → terminal
```

Audio is captured per-app via ScreenCaptureKit, buffered into overlapping 20-second windows, and translated by Whisper running locally on your GPU. Text appears as **confirmed** (bold — seen in two consecutive windows) or *speculative* (dim — may change). Speculative text is overwritten in-place as new audio arrives.

## Quick start

```bash
git clone https://github.com/mcinteerj/langlistn.git
cd langlistn
python3 -m venv .venv
.venv/bin/pip install .
bash swift/build.sh
```

Grant your terminal **Screen & System Audio Recording** permissions in System Settings → Privacy & Security. Restart your terminal after.

```bash
langlistn --app "Google Chrome"
```

That's it. First run downloads the Whisper model (~3GB). Subsequent runs start in a few seconds.

## Usage

### App audio

```bash
langlistn --app "Google Chrome"
langlistn --app "zoom.us"
langlistn --app "Microsoft Teams"
langlistn --app "Spotify"
langlistn --app "Discord"
```

### Language hints

Auto-detection works, but hinting the source language improves speed and accuracy:

```bash
langlistn --app "Google Chrome" --source ko     # Korean
langlistn --app "zoom.us" --source ja            # Japanese
langlistn --app "Microsoft Teams" --source zh    # Mandarin
langlistn --app "Google Chrome" --source fr      # French
langlistn --app "Google Chrome" --source de      # German
```

<details>
<summary>All supported language codes</summary>

`ko` Korean · `ja` Japanese · `zh` Mandarin · `zh-yue` Cantonese · `th` Thai · `vi` Vietnamese · `fr` French · `de` German · `es` Spanish · `ar` Arabic · `hi` Hindi · `pt` Portuguese · `it` Italian · `ru` Russian · `id` Indonesian · `ms` Malay · `tl` Tagalog

Whisper supports 99 languages — these codes are hints, not requirements.
</details>

### Microphone

```bash
langlistn --mic
langlistn --mic --device "MacBook Pro Microphone"
langlistn --mic --source ko
```

### Dual language

Show the original language alongside the English translation:

```bash
langlistn --app "Google Chrome" --dual-lang
langlistn --app "zoom.us" --source ko --dual-lang
```

Original text appears dim above the bold English translation.

### Pipe-friendly output

Strip all ANSI formatting and speculative text — only confirmed translations:

```bash
langlistn --app "Google Chrome" --plain | tee meeting.txt
langlistn --app "zoom.us" --plain --source ja > transcript.txt
```

### Logging

```bash
langlistn --app "Google Chrome" --log meeting.txt
```

### Discovery

```bash
langlistn --list-apps       # Show capturable apps (must be running)
langlistn --list-devices    # Show audio input devices
```

## Model selection

The model is auto-selected based on your Mac's RAM:

| RAM | Model | Size | Speed (M4 Max) |
|-----|-------|------|-----------------|
| 32GB+ | `whisper-large-v3` | 3GB | ~1.5s per 20s window |
| 16GB | `whisper-large-v3` | 3GB | ~2-3s per 20s window |
| 12GB | `whisper-medium` | 1.5GB | ~1-2s per 20s window |
| 8GB | `whisper-small` | 500MB | ~0.5-1s per 20s window |

Override with `--model`:

```bash
langlistn --app "Google Chrome" --model mlx-community/whisper-large-v3-mlx
langlistn --app "Google Chrome" --model mlx-community/whisper-medium-mlx
```

## How the sliding window works

```
Audio timeline:
|----5s----|----5s----|----5s----|----5s----|
0          5          10         15         20

Window 1 (at t=20s):
|==================== 20s ====================|
0                                             20

Window 2 (at t=25s):    5s step →
          |==================== 20s ====================|
          5                                             25

Each window:
  [0-10s]  already confirmed → skip
  [10-15s] was speculative last time → CONFIRM (bold)
  [15-20s] new audio → speculative (dim, overwritten next cycle)
```

Longer windows give Whisper more context, dramatically improving translation quality. The 15-second overlap between consecutive windows means each piece of audio is seen multiple times before being confirmed.

## Remote API mode

For streaming via Azure OpenAI Realtime API (requires API key, costs ~$0.10-0.20/min):

```bash
pip install "langlistn[remote]"

export AZURE_OPENAI_API_KEY="your-key"
export OPENAI_API_BASE="https://your-resource.openai.azure.com/"

langlistn --app "Google Chrome" --remote
langlistn --app "Google Chrome" --remote --deployment gpt-realtime
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| **No audio / silence** | Grant both Screen Recording AND System Audio Recording permissions. Restart terminal. |
| **Swift build fails** | Install Xcode Command Line Tools: `xcode-select --install` |
| **App not in `--list-apps`** | The app must be running and producing audio. |
| **Slow first run** | Model download (~3GB for large-v3). Cached after first run. |
| **Out of memory** | Use a smaller model: `--model mlx-community/whisper-small-mlx` |
| **Hallucination loops** | Built-in detection suppresses these. Try adding `--source` hint. |
| **Intel Mac** | Not supported. mlx-whisper requires Apple Silicon (M1+). |

## Project structure

```
langlistn/
├── pyproject.toml
├── src/langlistn/
│   ├── __main__.py           # CLI entry point
│   ├── cli_output.py         # Terminal output with ANSI overwriting
│   ├── app.py                # Async orchestrator (TUI/remote mode)
│   ├── config.py             # Languages, constants, prompts
│   ├── audio/
│   │   ├── __init__.py       # App capture (ScreenCaptureKit)
│   │   └── mic_capture.py    # Mic capture (sounddevice)
│   ├── whisper_local/
│   │   └── __init__.py       # Local Whisper session (sliding window)
│   ├── realtime/
│   │   └── __init__.py       # Azure OpenAI Realtime API session
│   └── tui/
│       └── __init__.py       # Terminal UI (Textual, optional)
└── swift/
    └── AudioCaptureHelper/   # Swift ScreenCaptureKit helper
```

## License

MIT — see [LICENSE](LICENSE).
