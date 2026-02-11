"""Constants, system prompt, language map."""

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_FRAMES = 1024

RECONNECT_BUFFER_SECONDS = 30
RECONNECT_BUFFER_MAX = int(RECONNECT_BUFFER_SECONDS * SAMPLE_RATE / CHUNK_FRAMES)

BACKOFF_INITIAL = 1.0
BACKOFF_MAX = 30.0

# VAD tuning
SILENCE_DURATION_MS = 500
VAD_THRESHOLD = 0.3
PREFIX_PADDING_MS = 100

# Max seconds of continuous speech before forcing a commit
MAX_SPEECH_DURATION_S = 10.0

# Client-side silence gate — RMS below this skips sending (int16 range)
SILENCE_RMS_THRESHOLD = 50

LANGUAGE_MAP: dict[str, str] = {
    "ko": "Korean",
    "zh": "Mandarin Chinese",
    "zh-yue": "Cantonese",
    "th": "Thai",
    "ja": "Japanese",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "ms": "Malay",
    "tl": "Tagalog",
    "hi": "Hindi",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "pt": "Portuguese",
    "it": "Italian",
    "ru": "Russian",
    "ar": "Arabic",
}

SYSTEM_PROMPT_BASE = """\
You are a real-time speech translator. You receive a continuous audio stream split into segments by voice activity detection. Each segment may overlap slightly with the previous one.

Your job: translate each segment into English. Output ONLY the new English translation.

Critical rules:
- NEVER repeat or rephrase something you already translated in a previous turn
- If a segment overlaps with what you already translated, skip the overlapping part and translate only the NEW speech
- If a segment contains nothing new, output a single empty line
- Do not output the original language — only English
- Automatically detect the source language from the audio
- Preserve meaning, tone, and intent
- Do not add commentary, explanations, or timestamps
- If speech is already in English, transcribe it as-is

Speaker identification:
- Always label speakers (Speaker 1, Speaker 2, etc.) at the start of each utterance
- Track speakers across turns — maintain consistent labels throughout the session
- Best-effort from mono audio: use voice characteristics to distinguish speakers"""


def build_system_prompt(lang_code: str | None = None) -> str:
    prompt = SYSTEM_PROMPT_BASE
    if lang_code:
        name = LANGUAGE_MAP.get(lang_code, lang_code)
        prompt += f"\n\nHint: the primary language being spoken is {name} ({lang_code})."
    return prompt


def resolve_language_name(lang_code: str | None) -> str | None:
    if not lang_code:
        return None
    return LANGUAGE_MAP.get(lang_code, lang_code)
