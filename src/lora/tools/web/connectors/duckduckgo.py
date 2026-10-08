"""DuckDuckGo Search Connector for L.O.R.A."""


from lora.tools.web.cache import HttpCache
from lora.tools.web.connectors.base import BaseConnector, Document, ReliabilityTier, SourceHit
from lora.tools.web.limiter import TokenBucketLimiter


class DuckDuckGoConnector(BaseConnector):
    """DuckDuckGo instant answer and search connector."""

    API_URL = "https://api.duckduckgo.com/"

    def __init__(
        self,
        limiter: TokenBucketLimiter | None = None,
        cache: HttpCache | None = None,
    ) -> None:
        super().__init__(
            name="duckduckgo",
            reliability_tier=ReliabilityTier.T3,
            limiter=limiter,
            cache=cache,
        )

    async def search(self, query: str, limit: int = 10) -> list[SourceHit]:
        """Fetch instant answers and related topics from DuckDuckGo."""
        params = {
            "q": query.strip(),
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1",
        }

        try:
            data = await self.get_json(self.API_URL, params=params)
            hits = []

            # 1. Primary abstract if present
            heading = data.get("Heading", "")
            abstract = data.get("AbstractText", "")
            abstract_url = data.get("AbstractURL", "")
            source = data.get("AbstractSource", "DuckDuckGo")

            if abstract:
                hits.append(SourceHit(
                    id="ddg:abstract",
                    title=heading or query,
                    url=abstract_url or f"https://duckduckgo.com/?q={query}",
                    source_name="duckduckgo",
                    reliability_tier=self.reliability_tier,
                    snippet=abstract,
                    is_preprint=False,
                    metadata={"source": source},
                ))

            # 2. Related topics
            for i, topic in enumerate(data.get("RelatedTopics", [])):
                if len(hits) >= limit:
                    break
                text = topic.get("Text", "")
                first_url = topic.get("FirstURL", "")
                if text and first_url:
                    hits.append(SourceHit(
                        id=f"ddg:topic:{i + 1}",
                        title=text.split(" - ")[0] if " - " in text else text[:60],
                        url=first_url,
                        source_name="duckduckgo",
                        reliability_tier=self.reliability_tier,
                        snippet=text,
                        is_preprint=False,
                    ))

            return hits
        except Exception:
            return []

    async def fetch(self, hit: SourceHit) -> Document:
        """Return Document from DuckDuckGo hit."""
        return Document(
            id=hit.id,
            url=hit.url,
            title=hit.title,
            source="duckduckgo",
            reliability_tier=self.reliability_tier,
            is_preprint=False,
            text=f"{hit.title}\n{hit.snippet}",
            summary=hit.snippet,
            metadata=hit.metadata,
        )
