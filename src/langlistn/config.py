"""Constants, system prompt, language map."""

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_FRAMES = 1024

RECONNECT_BUFFER_SECONDS = 30
RECONNECT_BUFFER_MAX = int(RECONNECT_BUFFER_SECONDS * SAMPLE_RATE / CHUNK_FRAMES)

BACKOFF_INITIAL = 1.0
BACKOFF_MAX = 30.0

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
You are a real-time speech translator and transcriber.

Your job is to translate all speech into English. Output ONLY the English translation.

Rules:
- Do not output the original language — only English
- Automatically detect the source language from the audio
- When you detect different speakers, label them (Speaker 1, Speaker 2, etc.) — best-effort from mono audio
- Preserve meaning, tone, and intent
- For incomplete sentences, provide your best current translation
- Do not add commentary, explanations, or timestamps
- If speech is already in English, transcribe it as-is
- Separate distinct utterances with newlines"""


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
