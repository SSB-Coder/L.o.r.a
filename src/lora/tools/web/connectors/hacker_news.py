"""Hacker News (Algolia) Community Connector for L.O.R.A."""


from lora.tools.web.cache import HttpCache
from lora.tools.web.connectors.base import BaseConnector, Document, ReliabilityTier, SourceHit
from lora.tools.web.limiter import TokenBucketLimiter


class HackerNewsConnector(BaseConnector):
    """Hacker News Algolia search connector for developer sentiment and discussions."""

    SEARCH_API = "https://hn.algolia.com/api/v1/search"

    def __init__(
        self,
        limiter: TokenBucketLimiter | None = None,
        cache: HttpCache | None = None,
    ) -> None:
        super().__init__(
            name="hackernews",
            reliability_tier=ReliabilityTier.T4,  # Community unverified discussions
            limiter=limiter,
            cache=cache,
        )

    async def search(self, query: str, limit: int = 10) -> list[SourceHit]:
        """Search Hacker News stories via Algolia API."""
        params = {
            "query": query.strip(),
            "tags": "story",
            "hitsPerPage": min(limit, 20),
        }

        try:
            data = await self.get_json(self.SEARCH_API, params=params)
            hits_data = data.get("hits", [])
            hits = []

            for h in hits_data:
                title = h.get("title", "HN Post")
                author = h.get("author", "unknown")
                points = h.get("points", 0)
                num_comments = h.get("num_comments", 0)
                object_id = str(h.get("objectID", ""))
                hn_url = f"https://news.ycombinator.com/item?id={object_id}"
                story_url = h.get("url") or hn_url
                created_at = h.get("created_at", "")

                snippet = f"Points: {points} | Comments: {num_comments} | Author: {author}"

                hits.append(SourceHit(
                    id=f"hn:{object_id}",
                    title=title,
                    url=story_url,
                    source_name="hackernews",
                    reliability_tier=self.reliability_tier,
                    snippet=snippet,
                    is_preprint=False,
                    authors=[author],
                    published_date=created_at,
                    metadata={"points": points, "comments": num_comments, "hn_url": hn_url},
                ))

            return hits
        except Exception:
            return []

    async def fetch(self, hit: SourceHit) -> Document:
        """Return Document representation of HN discussion hit."""
        return Document(
            id=hit.id,
            url=hit.url,
            title=hit.title,
            authors=hit.authors,
            published_date=hit.published_date,
            source="hackernews",
            reliability_tier=self.reliability_tier,
            is_preprint=False,
            license="",
            text=f"{hit.title}\n{hit.snippet}\nHN Thread: {hit.metadata.get('hn_url', '')}",
            summary=hit.snippet,
            metadata=hit.metadata,
        )
