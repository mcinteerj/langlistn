# langlistn

Real-time audio translation to English on your Mac. Point it at any app or microphone and get live English subtitles — works with Korean, Japanese, Thai, Cantonese, and 90+ other languages.

Local Whisper transcription + cloud translation via AWS Bedrock Claude for natural, accurate results.

> **macOS 15+** · **Apple Silicon (M1+)** · **Python 3.11+** · **AWS account** (for translation)

## Quickstart

```bash
git clone https://github.com/mcinteerj/langlistn.git
cd langlistn
python3 -m venv .venv && .venv/bin/pip install .
bash swift/build.sh
aws sso login                          # or however you auth to AWS
langlistn                              # interactive setup walks you through it
```

The interactive wizard lets you pick an audio source (app or mic), source language, and translation model. Your last 5 configurations are saved — on next launch, just pick a recent config or start a new one.

> **First run:** Downloads the Whisper model (~3 GB). Grant your terminal **Screen Recording** and **System Audio Recording** permissions in System Settings → Privacy & Security, then restart your terminal.

### AWS setup

Translation uses [AWS Bedrock](https://aws.amazon.com/bedrock/) with cross-region inference (no specific region required). You need:

1. An AWS account with valid credentials (`aws configure` / SSO / env vars)
2. **Bedrock model access** enabled for Anthropic Claude models — go to [AWS Console → Bedrock → Model access](https://console.aws.amazon.com/bedrock/home#/modelaccess) and request access to the Claude models you want to use
3. IAM permissions: `bedrock:InvokeModel` and `bedrock:InvokeModelWithResponseStream` on `arn:aws:bedrock:*::foundation-model/anthropic.claude-*`

> Currently AWS Bedrock is the only supported translation provider. Other providers (e.g. direct Anthropic API, local models) could be added in future if there's interest — open an issue.

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
│    → mlx-whisper transcribe (18s buffer)         │
│      init_prompt = prior confirmed text          │
│    → LocalAgreement-2 → confirmed source         │
│    → stash last 3 speculative hypotheses         │
│                                                  │
│  Speculative: current whisper hypothesis         │
│  Confirmed: agreed by 2 consecutive runs         │
└──────────────┬───────────────────────────────────┘
               ↓
┌──────────────────────────────────────────────────┐
│  TRANSLATION LOOP (parallel, event-driven)       │
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

Audio is captured per-app via ScreenCaptureKit. Non-speech audio (music, silence) is filtered by Silero VAD before reaching Whisper. Whisper transcribes in the source language using a growing buffer with LocalAgreement-2 — text is confirmed only when two consecutive runs agree. Transcription and translation run as parallel async loops — whisper updates the display immediately, while Claude translates independently using the latest source text. Translation is confirmed by diffing consecutive LLM outputs — stable prefixes are locked and fed back as context. Streaming LLM responses provide incremental display updates as tokens arrive.

## Usage

The simplest way to use langlistn is the interactive wizard — just run `langlistn` with no arguments. The sections below cover CLI flags for scripting, automation, and power-user control.

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

> Cost estimates as of early 2026. See [AWS Bedrock pricing](https://aws.amazon.com/bedrock/pricing/) for current rates.

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
- **18s audio buffer**: Whisper always processes a 30s spectrogram internally (shorter audio is zero-padded). 18s balances context for CJK languages (where word boundaries are ambiguous) against inference speed.

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
| **Translation errors** | Check AWS auth: `aws sts get-caller-identity`. Ensure Bedrock Claude model access is enabled. |
| **App not in `--list-apps`** | The app must be running and producing audio. |
| **Slow first run** | Model download (~3 GB for large-v2). Cached after first run. |
| **Hallucination loops** | Built-in VAD + hallucination detection. Try adding `--source` hint. |
| **Intel Mac** | Not supported. mlx-whisper requires Apple Silicon (M1+). |

## License

MIT — see [LICENSE](LICENSE).
