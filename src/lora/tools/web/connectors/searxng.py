"""SearXNG Privacy-Preserving Meta-Search Connector for L.O.R.A."""


from lora.tools.web.cache import HttpCache
from lora.tools.web.connectors.base import BaseConnector, Document, ReliabilityTier, SourceHit
from lora.tools.web.limiter import TokenBucketLimiter


class SearXNGConnector(BaseConnector):
    """SearXNG JSON meta-search connector."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8888",
        limiter: TokenBucketLimiter | None = None,
        cache: HttpCache | None = None,
    ) -> None:
        super().__init__(
            name="searxng",
            reliability_tier=ReliabilityTier.T3,  # General web search
            limiter=limiter,
            cache=cache,
        )
        self.base_url = base_url.rstrip("/")

    async def search(self, query: str, limit: int = 10) -> list[SourceHit]:
        """Query SearXNG JSON endpoint."""
        search_url = f"{self.base_url}/search"
        params = {
            "q": query.strip(),
            "format": "json",
        }

        try:
            data = await self.get_json(search_url, params=params)
            results = data.get("results", [])
            hits = []

            for r in results[:limit]:
                title = r.get("title", "")
                url = r.get("url", "")
                content = r.get("content", "")
                engine = r.get("engine", "web")
                published_date = r.get("publishedDate", "")

                hits.append(SourceHit(
                    id=f"searxng:{len(hits) + 1}",
                    title=title,
                    url=url,
                    source_name="searxng",
                    reliability_tier=self.reliability_tier,
                    snippet=content,
                    is_preprint=False,
                    published_date=published_date,
                    metadata={"engine": engine},
                ))

            return hits
        except Exception:
            return []

    async def fetch(self, hit: SourceHit) -> Document:
        """Return Document from SearXNG search result."""
        return Document(
            id=hit.id,
            url=hit.url,
            title=hit.title,
            published_date=hit.published_date,
            source="searxng",
            reliability_tier=self.reliability_tier,
            is_preprint=False,
            text=f"{hit.title}\n{hit.snippet}",
            summary=hit.snippet,
            metadata=hit.metadata,
        )
