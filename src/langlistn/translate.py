"""Translation via AWS Bedrock Claude models."""

import json
import logging

import boto3

logger = logging.getLogger(__name__)

# Model tiers — global cross-region inference profiles (lower latency, ~10% cheaper)
MODELS = {
    "haiku": "global.anthropic.claude-haiku-4-5-20251001-v1:0",
    "sonnet": "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "opus": "global.anthropic.claude-opus-4-6-v1",
}

LANGUAGE_NAMES = {
    "ko": "Korean",
    "ja": "Japanese",
    "zh": "Chinese",
    "th": "Thai",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "ms": "Malay",
    "tl": "Tagalog",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "pt": "Portuguese",
    "ru": "Russian",
    "ar": "Arabic",
    "hi": "Hindi",
}


class BedrockTranslator:
    """Translate text chunks via Bedrock Claude, maintaining context for coherence."""

    def __init__(self, model_tier: str = "haiku", region: str | None = None):
        self.model_id = MODELS.get(model_tier, MODELS["haiku"])
        self.model_tier = model_tier
        self._client = None
        self._region = region

        # Sliding context window of recent confirmed English
        self._context: list[str] = []
        self._max_context_chars = 500

        # Stats
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0

    def _get_client(self):
        if self._client is None:
            kwargs = {}
            if self._region:
                kwargs["region_name"] = self._region
            self._client = boto3.client("bedrock-runtime", **kwargs)
        return self._client

    def translate(self, text: str, source_lang: str | None = None) -> str:
        """Translate a text chunk to English.

        Args:
            text: source language text to translate
            source_lang: ISO code (ko, ja, zh, th, etc.) or None for auto-detect

        Returns:
            English translation
        """
        if not text.strip():
            return ""

        lang_name = LANGUAGE_NAMES.get(source_lang, source_lang or "the source language")

        # Build context string
        context_str = ""
        if self._context:
            context_str = (
                "\n\nPrior English translation (for continuity — do NOT retranslate):\n"
                + " ".join(self._context)
            )

        prompt = (
            f"Translate the following text to natural, spoken English. "
            f"Source language: {lang_name}. "
            f"This is live speech from a meeting/conversation — use informal, natural register. "
            f"Continue naturally from the prior context if provided. "
            f"RULES: "
            f"- Output ONLY the English translation, no commentary, no explanations, no questions. "
            f"- If the text contains mixed languages or garbled characters, translate what you can and skip the rest. "
            f"- If the text is already English, output it as-is. "
            f"- Never refuse to translate. Never mention the source language."
            f"{context_str}\n\n"
            f"Translate this:\n{text}"
        )

        client = self._get_client()
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
        }

        try:
            response = client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body),
                contentType="application/json",
            )
            result = json.loads(response["body"].read())
            translation = result["content"][0]["text"].strip()

            # Track usage
            usage = result.get("usage", {})
            self.input_tokens += usage.get("input_tokens", 0)
            self.output_tokens += usage.get("output_tokens", 0)
            self.calls += 1

            # Update context window
            self._context.append(translation)
            # Trim to max chars
            while self._context and sum(len(s) for s in self._context) > self._max_context_chars:
                self._context.pop(0)

            return translation

        except Exception as e:
            logger.error("Bedrock translation failed: %s", e)
            return f"[translation error: {e}]"

    def estimated_cost(self) -> float:
        """Rough cost estimate based on Haiku 4.5 pricing."""
        # Haiku 4.5: $1.00/1M input, $5.00/1M output
        # Sonnet 4.5: ~$3.00/1M input, $15.00/1M output
        # Opus: ~$15.00/1M input, $75.00/1M output
        pricing = {
            "haiku": (1.0, 5.0),
            "sonnet": (3.0, 15.0),
            "opus": (15.0, 75.0),
        }
        inp_rate, out_rate = pricing.get(self.model_tier, (1.0, 5.0))
        cost = (self.input_tokens * inp_rate + self.output_tokens * out_rate) / 1_000_000
        return cost
