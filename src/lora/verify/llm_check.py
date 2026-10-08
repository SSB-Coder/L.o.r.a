"""V4: Cross-family LLM Fact-Checker."""

import json
import logging
import re
from typing import Any

from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)


class LlmFactCheckResult(BaseModel):
    """Result of cross-family LLM fact check."""

    model_config = ConfigDict(frozen=True)

    is_supported: bool
    supporting_quote: str | None = None
    reason: str = ""


class LlmFactChecker:
    """Uses a verifier model of a different family than the writer to audit claims."""

    def __init__(self, ollama_client: Any = None, model_name: str = "granite4.2:8b") -> None:
        self.ollama_client = ollama_client
        self.model_name = model_name

    async def verify_claim(self, premise: str, claim: str) -> LlmFactCheckResult:
        """Verify whether a claim is directly supported by the cited source chunk."""
        if not premise or not claim:
            return LlmFactCheckResult(is_supported=False, supporting_quote=None, reason="Empty input")

        if self.ollama_client is None:
            # Fallback heuristic if no Ollama instance is configured
            c_clean = re.sub(r"[^\w\s]", "", claim.lower())
            p_clean = re.sub(r"[^\w\s]", "", premise.lower())
            c_words = [w for w in c_clean.split() if len(w) > 2]
            p_words = set(p_clean.split())
            if not c_words:
                is_sub = True
            else:
                matched = sum(1 for w in c_words if w in p_words)
                is_sub = (matched / len(c_words)) >= 0.60
            return LlmFactCheckResult(
                is_supported=is_sub,
                supporting_quote=premise if is_sub else None,
                reason="Direct heuristic match" if is_sub else "Unsupported claim",
            )

        prompt = (
            "You are a strict fact-checking supervisor.\n"
            f"Source Premise:\n\"\"\"{premise}\"\"\"\n\n"
            f"Claim to Verify:\n\"\"\"{claim}\"\"\"\n\n"
            "Task: Check if the claim is directly supported by the source premise. "
            "Quote the supporting sentence from the premise or output NONE if not found.\n"
            "Respond ONLY with valid JSON in this schema:\n"
            "{\n"
            "  \"supported\": true|false,\n"
            "  \"supporting_quote\": \"exact quote from premise\" | null,\n"
            "  \"reason\": \"brief explanation\"\n"
            "}\n"
        )

        try:
            resp = await self.ollama_client.generate(prompt=prompt)
            raw = resp.response if hasattr(resp, "response") else str(resp)

            # Extract JSON block
            json_match = re.search(r"\{[\s\S]*\}", raw)
            if json_match:
                data = json.loads(json_match.group(0))
                supp = bool(data.get("supported", False))
                quote = data.get("supporting_quote")
                if quote == "NONE" or quote == "":
                    quote = None
                return LlmFactCheckResult(
                    is_supported=supp,
                    supporting_quote=quote,
                    reason=str(data.get("reason", "")),
                )
        except Exception as e:
            logger.warning("Error running LLM fact-checker: %s", e)

        # Fallback to lexical containment
        c_clean = re.sub(r"[^\w\s]", "", claim.lower())
        p_clean = re.sub(r"[^\w\s]", "", premise.lower())
        c_words = [w for w in c_clean.split() if len(w) > 2]
        p_words = set(p_clean.split())
        is_contained = bool(c_words and (sum(1 for w in c_words if w in p_words) / len(c_words) >= 0.60))
        return LlmFactCheckResult(
            is_supported=is_contained,
            supporting_quote=premise if is_contained else None,
            reason="Fallback evaluation" if is_contained else "Unsupported claim",
        )
