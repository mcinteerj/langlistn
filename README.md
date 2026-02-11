# langlistn

Real-time audio translation to English on your Mac. Point it at any app or microphone and get live English subtitles — works with Korean, Japanese, Thai, Cantonese, and 90+ other languages.

Local Whisper transcription + cloud translation via AWS Bedrock Claude for natural, accurate results.

> **macOS 15+** · **Apple Silicon (M1+)** · **Python 3.11+** · **AWS account** (for translation)

## How it works

```
App/Mic audio → Swift helper (ScreenCaptureKit) → 16kHz PCM
    ↓
Silero VAD gate (only speech passes through)
    ↓
mlx-whisper large-v2 (local, free)
    ↓
LocalAgreement-2 (confirmed when 2 consecutive runs agree)
    ↓
Claude translation (continuation-based prompting)
    ↓
Two-zone terminal (bold = confirmed, dim italic = speculative)
```

### Two confidence loops

```
┌──────────────────────────────────────────────────┐
│  TRANSCRIPTION LOOP (~1-2s cycle)                │
│                                                  │
│  Audio chunks                                    │
│    → Silero VAD (discard non-speech)             │
│    → mlx-whisper transcribe (25s buffer)         │
│      init_prompt = prior confirmed text          │
│    → LocalAgreement-2 → confirmed source         │
│    → stash last 3 speculative hypotheses         │
│                                                  │
│  Speculative: current whisper hypothesis         │
│  Confirmed: agreed by 2 consecutive runs         │
└──────────────┬───────────────────────────────────┘
               ↓
┌──────────────────────────────────────────────────┐
│  TRANSLATION LOOP (on every whisper cycle)       │
│                                                  │
│  Prompt to Claude:                               │
│    "Korean transcript: {full_source}"            │
│    "Alt transcriptions: {recent hypotheses}"     │
│    "Reference translation: {confirmed_en}"       │
│    "Produce full updated translation"            │
│                                                  │
│  Confirmation:                                   │
│    Compare against last 4 LLM outputs            │
│    Lock at word/sentence boundary                │
│    Force-lock after 3 unstable cycles            │
│                                                  │
│  confirmed_en fed back as reference ───────────┐ │
└──────────────┬─────────────────────────────────┘ │
               ↓                          ↑       │
        Terminal display                  └───────┘
        bold = locked, dim italic = live tail
```

Audio is captured per-app via ScreenCaptureKit. Non-speech audio (music, silence) is filtered by Silero VAD before reaching Whisper. Whisper transcribes in the source language using a growing buffer with LocalAgreement-2 — text is confirmed only when two consecutive runs agree. The full source transcript and confirmed English are sent to Claude, which produces an updated translation. Translation is confirmed by diffing consecutive LLM outputs — stable prefixes are locked and fed back as context.

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
# Interactive setup — choose app, language, model
langlistn

# Or direct
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
```

### Translation model

Choose Claude model tier for translation quality vs speed:

```bash
langlistn --app "zoom.us" --source ko                          # haiku (default, fastest)
langlistn --app "zoom.us" --source ko --translate-model sonnet  # better quality
langlistn --app "zoom.us" --source ko --translate-model opus    # best quality
```

| Model | Latency | Cost/hr | Best for |
|-------|---------|---------|----------|
| `haiku` | ~300ms | ~$0.30 | Meetings, casual use |
| `sonnet` | ~500ms | ~$1.00 | Important conversations |
| `opus` | ~800ms | ~$5.00 | Maximum accuracy |

### Transcribe only (no translation)

```bash
langlistn --app "Google Chrome" --source ko --no-translate
```

### Microphone

```bash
langlistn --mic
langlistn --mic --source ko
```

### Pipe-friendly output

Strip all ANSI formatting and speculative text — only confirmed translations:

```bash
langlistn --app "Google Chrome" --plain | tee meeting.txt
```

### Debug logging

Full whisper transcripts and LLM responses to file:

```bash
langlistn --app "Google Chrome" --source ko --debug-log debug.log
```

### Tunables

```bash
langlistn --app "Google Chrome" --source ko \
  --max-context 2000 \       # Max chars of source/translation context sent to LLM
  --silence-reset 10 \       # Seconds of silence before resetting translation context
  --force-confirm 3          # Force-lock translation after N unstable cycles
```

### Discovery

```bash
langlistn --list-apps       # Show capturable apps (must be running)
langlistn --list-devices    # Show audio input devices
```

### All flags

| Flag | Description |
|------|-------------|
| `--app NAME` | Capture audio from named app |
| `--mic` | Capture from microphone |
| `--source CODE` | Source language hint (ko, ja, zh, etc.) |
| `--translate-model TIER` | Claude model: haiku (default), sonnet, opus |
| `--no-translate` | Transcribe only, skip translation |
| `--model NAME` | Whisper model override (default: auto by RAM) |
| `--device NAME` | Microphone device name |
| `--plain` | Pipe-friendly output (no ANSI, confirmed only) |
| `--log FILE` | Save confirmed text to file |
| `--json` | JSON output for `--list-apps` / `--list-devices` |
| `--debug-log FILE` | Write debug log to file |
| `--max-context N` | Max context chars for LLM (default: 2000) |
| `--silence-reset N` | Seconds of silence before context reset (default: 10) |
| `--force-confirm N` | Force-lock translation after N unstable cycles (default: 3) |
| `--list-apps` | Show capturable apps |
| `--list-devices` | Show audio input devices |
| `--version` | Show version |

## Architecture

### Key design decisions

- **Transcribe, don't translate with Whisper**: Whisper's `translate` task rephrases the same audio differently each run, breaking agreement-based confirmation. Transcribing in the source language produces consistent output.
- **Whisper large-v2 over v3**: v3 hallucinates significantly more on non-English audio (4x higher in benchmarks). v2 is more reliable for Korean, Japanese, and other Asian languages.
- **Silero VAD gate**: Filters non-speech audio before Whisper. Prevents hallucination loops on silence, music, and background noise — the primary cause of Whisper producing garbage output.
- **LocalAgreement-2**: Text confirmed only when two consecutive Whisper runs agree on the same words — eliminates content loss and repetition.
- **Continuation-based translation**: Each Claude call gets the full source transcript + confirmed English as reference. The LLM produces a full updated translation, not just a continuation — this avoids overlap/duplication errors.
- **Two-zone confirmation**: Translation is confirmed by diffing recent LLM outputs. Stable prefixes lock at word boundaries. Force-confirm after 3 unstable cycles prevents text from staying speculative forever.
- **Ring buffer comparison**: Compares current LLM output against the last 4 outputs, not just the previous one. If outputs 1 and 3 agree (even though 2 differed), confirmation still fires.
- **Multi-hypothesis translation**: Recent whisper speculative outputs (which may differ across runs) are fed to Claude as alternatives, helping disambiguate homophones and boundary-ambiguous words.
- **25s audio buffer**: Whisper always processes a 30s spectrogram internally (shorter audio is zero-padded). Keeping 25s in the buffer maximises context for CJK languages where word boundaries are ambiguous.

### Project structure

```
langlistn/
├── pyproject.toml
├── src/langlistn/
│   ├── __main__.py           # Interactive setup + CLI entry point
│   ├── pipeline.py           # Two-loop orchestrator (transcription + translation)
│   ├── streaming_asr.py      # LocalAgreement-2 engine (from whisper_streaming)
│   ├── translate.py          # Continuation-based Bedrock Claude translation
│   ├── display.py            # Two-zone terminal renderer
│   ├── config.py             # Languages, constants, model selection
│   ├── audio/
│   │   ├── __init__.py       # App capture (ScreenCaptureKit)
│   │   └── mic_capture.py    # Mic capture (sounddevice)
└── swift/
    └── AudioCaptureHelper/   # Swift ScreenCaptureKit helper
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| **No audio / silence** | Grant both Screen Recording AND System Audio Recording permissions. Restart terminal. |
| **Swift build fails** | Install Xcode Command Line Tools: `xcode-select --install` |
| **Translation errors** | Check AWS auth: `aws sts get-caller-identity`. Ensure Bedrock Claude access. |
| **App not in `--list-apps`** | The app must be running and producing audio. |
| **Slow first run** | Model download (~3GB for large-v2). Cached after first run. |
| **Hallucination loops** | Built-in VAD + hallucination detection. Try adding `--source` hint. |
| **Intel Mac** | Not supported. mlx-whisper requires Apple Silicon (M1+). |

## License

MIT — see [LICENSE](LICENSE).
