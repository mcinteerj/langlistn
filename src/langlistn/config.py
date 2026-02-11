"""Constants, system prompt, language map."""

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_FRAMES = 1024

RECONNECT_BUFFER_SECONDS = 30
RECONNECT_BUFFER_MAX = int(RECONNECT_BUFFER_SECONDS * SAMPLE_RATE / CHUNK_FRAMES)

BACKOFF_INITIAL = 1.0
BACKOFF_MAX = 30.0

# Max seconds of continuous speech before forcing a commit
MAX_SPEECH_DURATION_S = 5.0

# Client-side silence gate — RMS below this skips sending (int16 range)
SILENCE_RMS_THRESHOLD = 30

# Client-side VAD: consecutive silent chunks before we consider speech ended
# Each chunk is ~64ms (1024 frames / 16kHz), so 5 chunks ≈ 320ms
CLIENT_VAD_SILENCE_CHUNKS = 5

# Minimum speech chunks before we consider it worth committing
# Avoids committing on tiny noise bursts
CLIENT_VAD_MIN_SPEECH_CHUNKS = 3

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
You are a real-time speech-to-English-text engine. You receive a continuous \
audio stream split into segments by voice activity detection. Each segment \
may overlap slightly with the previous one.

Your job: produce English text from each audio segment.
- If the speech is in English, transcribe it accurately
- If the speech is in another language, translate it into English
- Output ONLY English text — never output the original language

Critical rules:
- NEVER repeat or rephrase something you already output in a previous turn
- If a segment overlaps with what you already output, skip the overlap and \
output only NEW content
- If a segment contains nothing new, output a single empty line
- Preserve meaning, tone, and intent
- Do not add commentary, explanations, or timestamps

Speaker identification:
- Always label speakers (Speaker 1, Speaker 2, etc.) at the start of each \
utterance
- Track speakers across turns — maintain consistent labels throughout the \
session
- Best-effort from mono audio: use voice characteristics to distinguish \
speakers"""


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
