"""V5: Cross-source corroboration check."""

from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict

from lora.grounding.cards import EvidenceCard


class CorroborationResult(BaseModel):
    """Result of cross-source corroboration verification."""

    model_config = ConfigDict(frozen=True)

    is_corroborated: bool
    independent_source_count: int
    is_single_source: bool


class CorroborationChecker:
    """Checks whether critical claims are corroborated across multiple independent sources."""

    def __init__(self, min_independent_sources: int = 2) -> None:
        self.min_independent_sources = min_independent_sources

    def _extract_domain(self, url: str) -> str:
        try:
            parsed = urlparse(url)
            return parsed.netloc.lower() or url.lower()
        except Exception:
            return url.lower()

    def check_corroboration(self, cards: list[EvidenceCard]) -> CorroborationResult:
        """Evaluate if evidence cards represent multiple independent sources."""
        if not cards:
            return CorroborationResult(
                is_corroborated=False,
                independent_source_count=0,
                is_single_source=False,
            )

        unique_domains = {self._extract_domain(c.source_url) for c in cards if c.source_url}
        count = len(unique_domains)
        is_single = count <= 1
        is_corroborated = count >= self.min_independent_sources

        return CorroborationResult(
            is_corroborated=is_corroborated,
            independent_source_count=count,
            is_single_source=is_single,
        )
