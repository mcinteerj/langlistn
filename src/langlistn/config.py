"""Constants and language config."""

import subprocess

SAMPLE_RATE = 16000
CHUNK_FRAMES = 1024

# Client-side silence gate — RMS below this skips sending (int16 range)
SILENCE_RMS_THRESHOLD = 30

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

# Model recommendations by available RAM
MODEL_BY_RAM = [
    (16, "mlx-community/whisper-large-v2-mlx"),
    (12, "mlx-community/whisper-medium-mlx"),
    (8, "mlx-community/whisper-small-mlx"),
    (0, "mlx-community/whisper-tiny"),
]


def recommend_model() -> str:
    """Pick best Whisper model based on system RAM."""
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
        ram_gb = int(out) / (1024**3)
    except Exception:
        ram_gb = 16
    for min_ram, model in MODEL_BY_RAM:
        if ram_gb >= min_ram:
            return model
    return MODEL_BY_RAM[-1][1]


def resolve_language_name(lang_code: str | None) -> str | None:
    if not lang_code:
        return None
    return LANGUAGE_MAP.get(lang_code, lang_code)
