"""Translation via AWS Bedrock Claude — continuation-based prompting.

Strategy: send source transcript + confirmed English prefix, ask LLM to
produce the FULL translation (not just continuation). Then diff against
prior output to find the stable prefix for confirmation.

This avoids the fragile "continue only" pattern where overlap/duplication
errors compound across calls.
"""

import json
import logging
import re
import sys

import boto3
from botocore.exceptions import (
    ClientError,
    NoCredentialsError,
    TokenRetrievalError,
    UnauthorizedSSOTokenError,
)

from .config import LANGUAGE_MAP

logger = logging.getLogger(__name__)

MODELS = {
    "haiku": "global.anthropic.claude-haiku-4-5-20251001-v1:0",
    "sonnet": "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "opus": "global.anthropic.claude-opus-4-6-v1",
}

PRICING = {
    "haiku": (1.0, 5.0),
    "sonnet": (3.0, 15.0),
    "opus": (15.0, 75.0),
}


def _sentence_boundaries(text: str) -> list[int]:
    """Return list of positions immediately after sentence-ending punctuation+space or paragraph breaks."""
    boundaries = [m.end() for m in re.finditer(r'[.!?。！？…]+\s*', text)]
    # Also treat paragraph breaks as boundaries
    boundaries.extend(m.end() for m in re.finditer(r'\n\s*\n', text))
    return sorted(set(boundaries))


def _longest_common_prefix_len(a: str, b: str) -> int:
    """Character-level common prefix length."""
    limit = min(len(a), len(b))
    for i in range(limit):
        if a[i] != b[i]:
            return i
    return limit


class ContinuationTranslator:
    """Translate by sending full source + English context, get full translation back.

    Confirmation: compares current output against the last 4 outputs. The best
    common prefix is found, and text is locked at the furthest sentence or word
    boundary within it (word boundary requires ≥20 chars of agreement).

    Force-confirm: if speculative text hasn't been promoted after
    `force_confirm_after` cycles, force-lock the latest output.
    """

    WORD_LOCK_MIN_CHARS = 20

    def __init__(
        self,
        model_tier: str = "haiku",
        source_lang: str | None = None,
        target_lang: str = "English",
        max_context_chars: int = 2000,
        region: str | None = None,
        force_confirm_after: int = 3,
    ):
        self.model_id = MODELS.get(model_tier, MODELS["haiku"])
        self.model_tier = model_tier
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.max_context_chars = max_context_chars
        self._client = None
        self._region = region
        self.force_confirm_after = force_confirm_after

        # State
        self.confirmed_translation: str = ""
        self.speculative_translation: str = ""
        self._last_full_output: str = ""
        self._recent_outputs: list[str] = []  # ring buffer of last N outputs
        self._max_recent: int = 4
        self._cycles_since_last_confirm: int = 0

        # Stats
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0

    _auth_failed: bool = False
    _auth_error_shown: bool = False

    def _get_client(self):
        if self._client is None:
            kwargs = {}
            if self._region:
                kwargs["region_name"] = self._region
            try:
                self._client = boto3.client("bedrock-runtime", **kwargs)
            except (NoCredentialsError, TokenRetrievalError, UnauthorizedSSOTokenError) as e:
                self._handle_auth_error(e)
                raise
        return self._client

    def _handle_auth_error(self, error: Exception):
        """Show a clear auth error message once."""
        if self._auth_error_shown:
            return
        self._auth_error_shown = True
        self._auth_failed = True
        msg = str(error)
        sys.stderr.write(
            f"\n\033[1;31m✗ AWS auth failed\033[0m: {msg}\n"
            f"  Run \033[1maws sso login\033[0m and try again.\n\n"
        )
        logger.error("AWS auth failed: %s", msg)

    def translate_streaming(
        self,
        source_text: str,
        alt_hypotheses: list[str] | None = None,
        on_token: "Callable[[str], None] | None" = None,
    ) -> tuple[str, str]:
        """Streaming translation. on_token receives accumulated text so far.

        Falls back to non-streaming translate() on error.
        """
        if not source_text.strip():
            return self.confirmed_translation, self.speculative_translation

        lang_name = LANGUAGE_MAP.get(
            self.source_lang, self.source_lang or "the source language"
        )
        source_trimmed = source_text[-self.max_context_chars:]
        confirmed_ctx = self.confirmed_translation[-self.max_context_chars:]
        prompt = self._build_prompt(lang_name, source_trimmed, confirmed_ctx, alt_hypotheses)

        if self._auth_failed:
            return self.confirmed_translation, self.speculative_translation

        try:
            client = self._get_client()
        except (NoCredentialsError, TokenRetrievalError, UnauthorizedSSOTokenError):
            return self.confirmed_translation, self.speculative_translation

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
        }

        try:
            response = client.invoke_model_with_response_stream(
                modelId=self.model_id,
                body=json.dumps(body),
                contentType="application/json",
            )
            stream = response.get("body")
            full_text = ""
            for event in stream:
                chunk = event.get("chunk")
                if chunk:
                    data = json.loads(chunk["bytes"])
                    if data.get("type") == "content_block_delta":
                        token = data.get("delta", {}).get("text", "")
                        full_text += token
                        if on_token:
                            on_token(full_text)
                    elif data.get("type") == "message_delta":
                        usage = data.get("usage", {})
                        self.output_tokens += usage.get("output_tokens", 0)
                    elif data.get("type") == "message_start":
                        usage = data.get("message", {}).get("usage", {})
                        self.input_tokens += usage.get("input_tokens", 0)

            full_translation = full_text.strip()
            self.calls += 1

            logger.debug(
                "LLM stream #%d | SOURCE: %s | LLM_OUT: %s",
                self.calls, source_text[:200], full_translation[:200],
            )

            self._update_confirmation(full_translation)
            return self.confirmed_translation, self.speculative_translation

        except (NoCredentialsError, TokenRetrievalError, UnauthorizedSSOTokenError) as e:
            self._handle_auth_error(e)
            return self.confirmed_translation, self.speculative_translation
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in ("ExpiredTokenException", "UnrecognizedClientException", "AccessDeniedException"):
                self._handle_auth_error(e)
            else:
                logger.error("Streaming translation failed: %s — falling back", e)
                return self.translate(source_text, alt_hypotheses)
            return self.confirmed_translation, self.speculative_translation
        except Exception as e:
            logger.error("Streaming translation failed: %s — falling back", e)
            return self.translate(source_text, alt_hypotheses)

    def translate(
        self, source_text: str, alt_hypotheses: list[str] | None = None,
    ) -> tuple[str, str]:
        """Translate source text. Returns (confirmed, speculative).

        alt_hypotheses: alternative whisper transcriptions of the speculative
        tail — different runs may transcribe ambiguous audio differently.
        Feeding these to the LLM helps disambiguate homophones.
        """
        if not source_text.strip():
            return self.confirmed_translation, self.speculative_translation

        lang_name = LANGUAGE_MAP.get(
            self.source_lang, self.source_lang or "the source language"
        )

        # Only send tail of source + confirmed to stay within context budget
        source_trimmed = source_text
        if len(source_trimmed) > self.max_context_chars:
            source_trimmed = source_trimmed[-self.max_context_chars:]

        confirmed_ctx = self.confirmed_translation
        if len(confirmed_ctx) > self.max_context_chars:
            confirmed_ctx = confirmed_ctx[-self.max_context_chars:]

        prompt = self._build_prompt(
            lang_name, source_trimmed, confirmed_ctx, alt_hypotheses
        )

        # Skip all LLM calls after auth failure
        if self._auth_failed:
            return self.confirmed_translation, self.speculative_translation

        try:
            client = self._get_client()
        except (NoCredentialsError, TokenRetrievalError, UnauthorizedSSOTokenError):
            return self.confirmed_translation, self.speculative_translation

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
            full_translation = result["content"][0]["text"].strip()

            usage = result.get("usage", {})
            self.input_tokens += usage.get("input_tokens", 0)
            self.output_tokens += usage.get("output_tokens", 0)
            self.calls += 1

            logger.debug(
                "LLM call #%d | in=%d out=%d tokens\n"
                "  SOURCE: %s\n  LLM_OUT: %s\n  CONFIRMED_EN: %s",
                self.calls,
                usage.get("input_tokens", 0),
                usage.get("output_tokens", 0),
                source_text[:200],
                full_translation[:200],
                self.confirmed_translation[:200],
            )

            self._update_confirmation(full_translation)
            return self.confirmed_translation, self.speculative_translation

        except (NoCredentialsError, TokenRetrievalError, UnauthorizedSSOTokenError) as e:
            self._handle_auth_error(e)
            return self.confirmed_translation, self.speculative_translation
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in ("ExpiredTokenException", "UnrecognizedClientException", "AccessDeniedException"):
                self._handle_auth_error(e)
            else:
                logger.error("Translation failed: %s", e)
            return self.confirmed_translation, self.speculative_translation
        except Exception as e:
            logger.error("Translation failed: %s", e)
            return self.confirmed_translation, self.speculative_translation

    def _build_prompt(
        self,
        lang_name: str,
        source_text: str,
        confirmed_english: str,
        alt_hypotheses: list[str] | None = None,
    ) -> str:
        """Ask for FULL translation, but provide confirmed English as guidance."""
        parts = [
            f"You are translating live {lang_name} speech to {self.target_lang}.",
            "",
            "RULES:",
            f"- Output ONLY the complete {self.target_lang} translation of the transcript below.",
            "- No commentary, no explanations, no meta-text, no labels.",
            "- Use natural, spoken register.",
            "- If the source is already in the target language, output it as-is.",
            "",
            f"=== {lang_name} transcript ===",
            source_text,
        ]

        # Alternative whisper hypotheses for disambiguation
        if alt_hypotheses:
            parts.extend([
                "",
                "=== Alternative transcriptions of recent audio (may help disambiguate) ===",
            ])
            for i, hyp in enumerate(alt_hypotheses, 1):
                # Only include tail — these are just the speculative portion
                parts.append(f"Alt {i}: {hyp[-300:]}")

        if confirmed_english:
            parts.extend([
                "",
                f"=== Reference: your prior {self.target_lang} translation (keep consistent) ===",
                confirmed_english,
                "",
                "Produce the full updated translation. Keep the beginning consistent "
                "with the reference above — only change/extend the end as needed "
                "for new source content.",
            ])
        else:
            parts.extend([
                "",
                f"Translate the above to {self.target_lang}.",
            ])

        return "\n".join(parts)

    def _update_confirmation(self, full_output: str):
        """Diff full_output against recent outputs. Lock stable prefix.

        Compares against all recent outputs (not just the last one) —
        if outputs N and N+2 agree even though N+1 differed, we still confirm.
        """
        # Add to ring buffer
        self._recent_outputs.append(full_output)
        if len(self._recent_outputs) > self._max_recent:
            self._recent_outputs.pop(0)

        if len(self._recent_outputs) < 2:
            self._last_full_output = full_output
            self.speculative_translation = full_output
            self._cycles_since_last_confirm += 1
            self._maybe_force_confirm(full_output)
            return

        # Find the best (longest) common prefix across all recent pairs
        best_common_len = 0
        for prev in self._recent_outputs[:-1]:
            cl = _longest_common_prefix_len(full_output, prev)
            if cl > best_common_len:
                best_common_len = cl
        common_len = best_common_len

        common_text = full_output[:common_len]
        boundaries = _sentence_boundaries(common_text)

        # Always lock at the furthest safe boundary in the common prefix.
        # Prefer sentence boundaries, but word boundaries are fine too —
        # in streaming translation the LLM rephrases constantly, so waiting
        # for sentence-level agreement is too conservative.
        new_lock_pos = -1
        if boundaries:
            new_lock_pos = boundaries[-1]
        if common_len >= self.WORD_LOCK_MIN_CHARS:
            last_space = common_text.rfind(" ")
            if last_space > new_lock_pos:
                new_lock_pos = last_space

        if new_lock_pos > len(self.confirmed_translation):
            self.confirmed_translation = full_output[:new_lock_pos].rstrip()
            self._cycles_since_last_confirm = 0
            logger.debug(
                "CONFIRM natural: locked %d chars (common_prefix=%d)",
                len(self.confirmed_translation), common_len,
            )
        else:
            self._cycles_since_last_confirm += 1

        # Speculative = everything after confirmed in the NEW output
        self.speculative_translation = full_output[len(self.confirmed_translation):].strip()
        self._last_full_output = full_output

        logger.debug(
            "CONFIRM state: common_prefix=%d lock_pos=%d confirmed_len=%d "
            "cycles_unstable=%d spec=%r",
            common_len, new_lock_pos, len(self.confirmed_translation),
            self._cycles_since_last_confirm,
            self.speculative_translation[:100],
        )

        # Force-confirm if stuck
        self._maybe_force_confirm(full_output)

    def _maybe_force_confirm(self, full_output: str):
        """If speculative text hasn't promoted in N cycles, force-lock it.
        Also: if LLM output is shorter than confirmed, trim confirmed back
        to match — the confirmed text has become stale."""
        # Stale confirmed detection: LLM output doesn't cover our confirmed text
        if len(full_output) < len(self.confirmed_translation) - 20:
            # LLM is producing less text than we have confirmed. This means
            # the source text has changed (e.g. after hallucination filtering)
            # and our confirmed text no longer matches. Trim back to what the
            # LLM actually produced, snapping to a sentence boundary.
            boundaries = _sentence_boundaries(full_output)
            if boundaries:
                trim_to = boundaries[-1]
            else:
                trim_to = len(full_output)
            if trim_to < len(self.confirmed_translation):
                logger.warning(
                    "CONFIRM trimming stale confirmed: %d → %d chars (LLM output was %d)",
                    len(self.confirmed_translation), trim_to, len(full_output),
                )
                self.confirmed_translation = full_output[:trim_to].rstrip()
                self.speculative_translation = full_output[trim_to:].strip()
                self._cycles_since_last_confirm = 0
                return

        if (
            self._cycles_since_last_confirm >= self.force_confirm_after
            and self.speculative_translation
        ):
            # Lock everything up to the last word boundary in full output
            lock_pos = len(full_output)
            last_space = full_output.rfind(" ")
            if last_space > len(self.confirmed_translation):
                lock_pos = last_space

            self.confirmed_translation = full_output[:lock_pos].rstrip()
            self.speculative_translation = full_output[lock_pos:].strip()
            self._cycles_since_last_confirm = 0
            logger.info(
                "CONFIRM forced: locked %d chars after %d unstable cycles",
                len(self.confirmed_translation), self.force_confirm_after,
            )

    def estimated_cost(self) -> float:
        inp_rate, out_rate = PRICING.get(self.model_tier, (1.0, 5.0))
        return (self.input_tokens * inp_rate + self.output_tokens * out_rate) / 1_000_000
