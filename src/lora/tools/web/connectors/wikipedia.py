"""Wikipedia Knowledge Connector for L.O.R.A."""

import urllib.parse

from lora.tools.web.cache import HttpCache
from lora.tools.web.connectors.base import BaseConnector, Document, ReliabilityTier, SourceHit
from lora.tools.web.limiter import TokenBucketLimiter


class WikipediaConnector(BaseConnector):
    """Wikipedia REST and Action API Connector."""

    SEARCH_API = "https://en.wikipedia.org/w/api.php"
    SUMMARY_API = "https://en.wikipedia.org/api/rest_v1/page/summary/"

    def __init__(
        self,
        limiter: TokenBucketLimiter | None = None,
        cache: HttpCache | None = None,
    ) -> None:
        super().__init__(
            name="wikipedia",
            reliability_tier=ReliabilityTier.T2,
            limiter=limiter,
            cache=cache,
        )

    async def search(self, query: str, limit: int = 10) -> list[SourceHit]:
        """Search Wikipedia pages using Action API."""
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query.strip(),
            "format": "json",
            "utf8": 1,
            "srlimit": min(limit, 20),
        }

        try:
            data = await self.get_json(self.SEARCH_API, params=params)
            search_results = data.get("query", {}).get("search", [])
            hits = []

            for item in search_results:
                title = item.get("title", "")
                pageid = str(item.get("pageid", ""))
                snippet = item.get("snippet", "").replace('<span class="searchmatch">', "").replace("</span>", "")
                encoded_title = urllib.parse.quote(title.replace(" ", "_"))
                url = f"https://en.wikipedia.org/wiki/{encoded_title}"

                hits.append(SourceHit(
                    id=f"wiki:{pageid or title}",
                    title=title,
                    url=url,
                    source_name="wikipedia",
                    reliability_tier=self.reliability_tier,
                    snippet=snippet,
                    is_preprint=False,
                    metadata={"title": title, "wordcount": item.get("wordcount", 0)},
                ))
            return hits
        except Exception:
            return []

    async def fetch(self, hit: SourceHit) -> Document:
        """Fetch summary extract and page details for a Wikipedia hit."""
        page_title = hit.metadata.get("title") or hit.title
        encoded_title = urllib.parse.quote(page_title.replace(" ", "_"))
        url = f"{self.SUMMARY_API}{encoded_title}"

        try:
            data = await self.get_json(url)
            extract = data.get("extract", "")
            description = data.get("description", "")
            page_url = data.get("content_urls", {}).get("desktop", {}).get("page", hit.url)

            return Document(
                id=hit.id,
                url=page_url,
                title=data.get("title", hit.title),
                authors=["Wikipedia Contributors"],
                published_date=data.get("timestamp", ""),
                source="wikipedia",
                reliability_tier=self.reliability_tier,
                is_preprint=False,
                license="CC BY-SA 4.0",
                text=extract or hit.snippet,
                summary=description or extract[:300],
                metadata=data,
            )
        except Exception:
            return Document(
                id=hit.id,
                url=hit.url,
                title=hit.title,
                authors=["Wikipedia Contributors"],
                source="wikipedia",
                reliability_tier=self.reliability_tier,
                is_preprint=False,
                license="CC BY-SA 4.0",
                text=hit.snippet,
                summary=hit.snippet,
            )
