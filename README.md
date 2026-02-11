# langlistn

Real-time audio translation to English on your Mac. Point it at any app or microphone and get live English subtitles — works with Korean, Japanese, Thai, Cantonese, and 90+ other languages.

Local Whisper transcription + cloud translation via AWS Bedrock Claude for natural, accurate results at ~$0.10/hr.

> **macOS 15+** · **Apple Silicon (M1+)** · **Python 3.11+** · **AWS account** (for translation)

## How it works

```
App/Mic audio → Swift helper (ScreenCaptureKit) → 16kHz PCM
    → mlx-whisper large-v3 (local, free)
    → LocalAgreement-2 (confirmed when 2 runs agree)
    → English? → passthrough
    → Other?  → AWS Bedrock Claude → natural English
    → terminal (bold confirmed, dim speculative)
```

Audio is captured per-app via ScreenCaptureKit. Whisper transcribes in the source language locally using a growing buffer with [LocalAgreement-2](https://github.com/ufal/whisper_streaming) — text is confirmed only when two consecutive Whisper runs produce the same words. Confirmed non-English text is translated by Claude on AWS Bedrock. English speech passes through without an API call.

## Quick start

```bash
git clone https://github.com/mcinteerj/langlistn.git
cd langlistn
python3 -m venv .venv
.venv/bin/pip install .
bash swift/build.sh
```

Grant your terminal **Screen & System Audio Recording** permissions in System Settings → Privacy & Security. Restart your terminal after.

### AWS setup

Translation requires AWS credentials with Bedrock access:

```bash
aws sso login    # or configure credentials however you prefer
```

Your AWS profile needs access to Bedrock Claude models in your configured region.

### Run

```bash
langlistn --app "Google Chrome" --source ko
```

First run downloads the Whisper model (~3GB). Subsequent runs start in a few seconds.

## Usage

### App audio

```bash
langlistn --app "Google Chrome"
langlistn --app "zoom.us" --source ko
langlistn --app "Microsoft Teams" --source ja
langlistn --app "Discord" --source zh
```

### Language hints

Auto-detection works, but hinting the source language improves speed and accuracy:

```bash
langlistn --app "Google Chrome" --source ko     # Korean
langlistn --app "zoom.us" --source ja            # Japanese
langlistn --app "Microsoft Teams" --source zh    # Mandarin
langlistn --app "Google Chrome" --source th      # Thai
langlistn --app "Google Chrome" --source fr      # French
```

<details>
<summary>All supported language codes</summary>

`ko` Korean · `ja` Japanese · `zh` Mandarin · `th` Thai · `vi` Vietnamese · `fr` French · `de` German · `es` Spanish · `ar` Arabic · `hi` Hindi · `pt` Portuguese · `it` Italian · `ru` Russian · `id` Indonesian · `ms` Malay · `tl` Tagalog

Whisper supports 99 languages — these codes are hints, not requirements.
</details>

### Translation model

Choose Claude model tier for translation quality vs speed:

```bash
langlistn --app "zoom.us" --source ko                          # haiku (default, fastest)
langlistn --app "zoom.us" --source ko --translate-model sonnet  # better quality
langlistn --app "zoom.us" --source ko --translate-model opus    # best quality
```

| Model | Latency | Cost/hr | Best for |
|-------|---------|---------|----------|
| `haiku` | ~300ms | ~$0.10 | Meetings, casual use |
| `sonnet` | ~500ms | ~$0.30 | Important conversations |
| `opus` | ~800ms | ~$1.50 | Maximum accuracy |

### Transcribe only (no translation)

Skip the API call — just show Whisper's raw transcription:

```bash
langlistn --app "Google Chrome" --source ko --no-translate
```

### Dual language

Show original language above the English translation:

```bash
langlistn --app "Google Chrome" --source ko --dual-lang
```

### Microphone

```bash
langlistn --mic
langlistn --mic --device "MacBook Pro Microphone"
langlistn --mic --source ko
```

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

## Mixed-language meetings

langlistn handles meetings where speakers switch between languages. English speech passes through directly (no API call), non-English speech gets translated. No configuration needed — it detects the language automatically.

```bash
# Korean/English meeting
langlistn --app "zoom.us" --source ko

# The --source hint helps Whisper with the primary non-English language
# but English segments still pass through untranslated
```

## Model selection

The Whisper model is auto-selected based on your Mac's RAM:

| RAM | Model | Size |
|-----|-------|------|
| 16GB+ | `whisper-large-v3` | 3GB |
| 12GB | `whisper-medium` | 1.5GB |
| 8GB | `whisper-small` | 500MB |

Override with `--model`:

```bash
langlistn --app "Google Chrome" --model mlx-community/whisper-large-v3-mlx
langlistn --app "Google Chrome" --model mlx-community/whisper-medium-mlx
```

## Architecture

### Cascade pipeline

```
Audio capture (Swift ScreenCaptureKit, 16kHz PCM mono)
         │
         ▼
Local Whisper transcription (mlx-whisper, growing buffer)
         │
         ▼
LocalAgreement-2 (commit text when 2 consecutive runs agree)
         │
    ┌────┴────┐
    │         │
 English   Non-English
    │         │
 passthru  Bedrock Claude translate (with context window)
    │         │
    └────┬────┘
         │
         ▼
Terminal display (bold confirmed, dim speculative)
```

### Key design decisions

- **Transcribe, don't translate with Whisper**: Whisper's `translate` task rephrases the same audio differently each run, breaking agreement-based confirmation. Transcribing in the source language produces consistent output that LocalAgreement can work with.
- **LocalAgreement-2**: Vendored from [whisper_streaming](https://github.com/ufal/whisper_streaming) (MIT). Text is confirmed only when two consecutive Whisper runs agree on the same words — eliminates content loss and excessive repetition.
- **Growing buffer with segment trimming**: Audio buffer grows naturally, trimmed at segment boundaries (15s threshold, 30s hard cap) to keep processing time bounded.
- **Context-aware translation**: Each Claude translation call includes the last ~500 chars of confirmed English as context, so translations flow naturally between chunks.
- **Non-blocking translation**: API calls run concurrently with Whisper processing — translation latency doesn't delay the next transcription cycle.

### Cost comparison

| Mode | Cost/hr | Quality |
|------|---------|---------|
| langlistn (Haiku) | ~$0.10 | Good transcription + natural translation |
| langlistn (Sonnet) | ~$0.30 | Good transcription + high-quality translation |
| GPT Realtime API | ~$3.60 | Streaming, lower latency |
| AWS Transcribe + translate | ~$1.50 | Good, no local compute |

## Remote API mode

For streaming via Azure OpenAI Realtime API (lower latency, higher cost):

```bash
pip install "langlistn[remote]"

export AZURE_OPENAI_API_KEY="your-key"
export OPENAI_API_BASE="https://your-resource.openai.azure.com/"

langlistn --app "Google Chrome" --remote
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| **No audio / silence** | Grant both Screen Recording AND System Audio Recording permissions. Restart terminal. |
| **Swift build fails** | Install Xcode Command Line Tools: `xcode-select --install` |
| **Translation errors** | Check AWS auth: `aws sts get-caller-identity`. Ensure Bedrock Claude access. |
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
│   ├── cli_output.py         # Cascade pipeline + terminal display
│   ├── streaming_asr.py      # LocalAgreement-2 engine (from whisper_streaming)
│   ├── translate.py          # AWS Bedrock Claude translation
│   ├── config.py             # Languages, constants
│   ├── audio/
│   │   ├── __init__.py       # App capture (ScreenCaptureKit)
│   │   └── mic_capture.py    # Mic capture (sounddevice)
│   ├── whisper_local/
│   │   └── __init__.py       # Whisper model loading, hallucination detection
│   ├── realtime/             # Azure OpenAI Realtime API (--remote)
│   └── tui/                  # Terminal UI (Textual, --remote --tui)
└── swift/
    └── AudioCaptureHelper/   # Swift ScreenCaptureKit helper
```

## License

MIT — see [LICENSE](LICENSE).
