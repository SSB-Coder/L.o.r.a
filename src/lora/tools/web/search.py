"""Unified Web and Knowledge Search Dispatcher for L.O.R.A."""

import asyncio
from typing import Any

from lora.tools.web.connectors.arxiv import ArxivConnector
from lora.tools.web.connectors.base import SourceConnector, SourceHit
from lora.tools.web.connectors.crossref import CrossrefConnector
from lora.tools.web.connectors.duckduckgo import DuckDuckGoConnector
from lora.tools.web.connectors.github import GitHubConnector
from lora.tools.web.connectors.hacker_news import HackerNewsConnector
from lora.tools.web.connectors.open_library import OpenLibraryConnector
from lora.tools.web.connectors.pubmed import PubMedConnector
from lora.tools.web.connectors.searxng import SearXNGConnector
from lora.tools.web.connectors.wikipedia import WikipediaConnector


class WebSearchDispatcher:
    """Dispatches knowledge queries to appropriate free connectors or search engines."""

    def __init__(self) -> None:
        self.connectors: dict[str, SourceConnector] = {
            "wikipedia": WikipediaConnector(),
            "arxiv": ArxivConnector(),
            "crossref": CrossrefConnector(),
            "pubmed": PubMedConnector(),
            "github": GitHubConnector(),
            "hackernews": HackerNewsConnector(),
            "openlibrary": OpenLibraryConnector(),
            "searxng": SearXNGConnector(),
            "duckduckgo": DuckDuckGoConnector(),
        }

    async def close(self) -> None:
        """Release connector cache workers before their owning loop stops."""
        caches = {id(connector.cache): connector.cache for connector in self.connectors.values()
                  if hasattr(connector, "cache")}
        for cache in caches.values():
            await cache.close()

    async def search(
        self,
        query: str,
        source: str = "auto",
        limit: int = 5,
    ) -> dict[str, Any]:
        """Perform search across specified source or auto-route across multiple sources."""
        clean_src = source.strip().lower()

        if clean_src in self.connectors:
            connector = self.connectors[clean_src]
            hits = await connector.search(query, limit=limit)
            return {
                "status": "success",
                "query": query,
                "source": clean_src,
                "count": len(hits),
                "hits": [h.model_dump() for h in hits],
            }

        # Auto mode: query Wikipedia and arXiv concurrently
        targets = [self.connectors["wikipedia"], self.connectors["arxiv"]]
        tasks = [t.search(query, limit=limit) for t in targets]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        combined_hits: list[SourceHit] = []
        for r in results:
            if isinstance(r, list):
                combined_hits.extend(r)

        return {
            "status": "success",
            "query": query,
            "source": "auto",
            "count": len(combined_hits),
            "hits": [h.model_dump() for h in combined_hits],
        }
