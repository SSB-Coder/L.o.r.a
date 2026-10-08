"""Open Library Books Connector for L.O.R.A."""


from lora.tools.web.cache import HttpCache
from lora.tools.web.connectors.base import BaseConnector, Document, ReliabilityTier, SourceHit
from lora.tools.web.limiter import TokenBucketLimiter


class OpenLibraryConnector(BaseConnector):
    """Open Library API connector for book metadata and publications."""

    SEARCH_API = "https://openlibrary.org/search.json"

    def __init__(
        self,
        limiter: TokenBucketLimiter | None = None,
        cache: HttpCache | None = None,
    ) -> None:
        super().__init__(
            name="openlibrary",
            reliability_tier=ReliabilityTier.T2,
            limiter=limiter,
            cache=cache,
        )

    async def search(self, query: str, limit: int = 10) -> list[SourceHit]:
        """Search books on Open Library."""
        params = {
            "q": query.strip(),
            "limit": min(limit, 15),
        }

        try:
            data = await self.get_json(self.SEARCH_API, params=params)
            docs = data.get("docs", [])
            hits = []

            for doc in docs:
                title = doc.get("title", "Untitled Book")
                authors = doc.get("author_name", [])
                first_publish_year = str(doc.get("first_publish_year", ""))
                key = doc.get("key", "")
                url = f"https://openlibrary.org{key}" if key.startswith("/") else f"https://openlibrary.org/{key}"
                publisher = (doc.get("publisher") or [""])[0]

                snippet = f"Author: {', '.join(authors[:3])} | First Published: {first_publish_year} | Publisher: {publisher}"

                hits.append(SourceHit(
                    id=f"ol:{key.strip('/')}",
                    title=title,
                    url=url,
                    source_name="openlibrary",
                    reliability_tier=self.reliability_tier,
                    snippet=snippet,
                    is_preprint=False,
                    authors=authors,
                    published_date=first_publish_year,
                    metadata={"key": key, "publisher": publisher},
                ))

            return hits
        except Exception:
            return []

    async def fetch(self, hit: SourceHit) -> Document:
        """Return Document representation of Open Library book hit."""
        return Document(
            id=hit.id,
            url=hit.url,
            title=hit.title,
            authors=hit.authors,
            published_date=hit.published_date,
            source="openlibrary",
            reliability_tier=self.reliability_tier,
            is_preprint=False,
            license="Public Domain / CC0",
            text=f"Book: {hit.title}\n{hit.snippet}",
            summary=hit.snippet,
            metadata=hit.metadata,
        )
