"""Base Protocol and Normalized Models for L.O.R.A Web Connectors."""

from typing import Any, Protocol, runtime_checkable

import httpx
from pydantic import BaseModel, Field

from lora.tools.web.cache import HttpCache
from lora.tools.web.limiter import TokenBucketLimiter


class ReliabilityTier:
    T1 = "T1"          # Peer-reviewed / Primary scientific literature
    T1_P = "T1-P"      # Unreviewed preprints (arXiv, bioRxiv, medRxiv)
    T2 = "T2"          # Encyclopedias, official docs, government/standards
    T3 = "T3"          # Reputable press, technical blogs, news
    T4 = "T4"          # Forums, HN, Reddit, AI-generated chatbot responses


class SourceHit(BaseModel):
    """Search hit returned by a knowledge connector."""
    id: str
    title: str
    url: str
    source_name: str
    reliability_tier: str
    snippet: str = ""
    is_preprint: bool = False
    authors: list[str] = Field(default_factory=list)
    published_date: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    """Normalized document fetched and extracted from a source."""
    id: str
    url: str
    title: str
    authors: list[str] = Field(default_factory=list)
    published_date: str = ""
    source: str
    reliability_tier: str
    is_preprint: bool = False
    license: str = ""
    text: str
    summary: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class SourceConnector(Protocol):
    """Protocol implemented by all knowledge source connectors."""
    name: str
    reliability_tier: str

    async def search(self, query: str, limit: int = 10) -> list[SourceHit]:
        """Search the source provider for matching hits."""
        ...

    async def fetch(self, hit: SourceHit) -> Document:
        """Fetch full content and metadata for a given hit."""
        ...


class BaseConnector:
    """Base connector providing rate-limiting, caching, and polite User-Agent."""

    USER_AGENT = "LORA-Research-Assistant/0.1.0 (https://github.com/shash/LORA; polite research bot)"

    def __init__(
        self,
        name: str,
        reliability_tier: str,
        limiter: TokenBucketLimiter | None = None,
        cache: HttpCache | None = None,
    ) -> None:
        self.name = name
        self.reliability_tier = reliability_tier
        self.limiter = limiter or TokenBucketLimiter()
        self.cache = cache or HttpCache()

    async def get_json(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        ttl_seconds: float = 3600.0,
    ) -> dict[str, Any]:
        """Fetch JSON endpoint with caching and polite rate limiting."""
        # 1. Build complete URL representation for caching
        req_headers = {"User-Agent": self.USER_AGENT}
        if headers:
            req_headers.update(headers)

        full_url = str(httpx.URL(url, params=params))

        # 2. Check cache
        cached = await self.cache.get(full_url)
        if cached:
            import json
            try:
                return json.loads(cached.body)
            except Exception:
                pass

        # 3. Rate limiter
        await self.limiter.acquire(url)

        # 4. Execute request
        async with httpx.AsyncClient(headers=req_headers, timeout=15.0) as client:
            resp = await client.get(url, params=params)
            if resp.status_code == 429:
                self.limiter.handle_rate_limit_response(url, resp.headers.get("Retry-After"))
                resp.raise_for_status()
            resp.raise_for_status()

            # Store in cache
            await self.cache.put(
                url=full_url,
                status_code=resp.status_code,
                headers=dict(resp.headers),
                body=resp.text,
                etag=resp.headers.get("ETag", ""),
                last_modified=resp.headers.get("Last-Modified", ""),
                ttl_seconds=ttl_seconds,
            )
            return resp.json()

    async def get_text(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        ttl_seconds: float = 3600.0,
    ) -> str:
        """Fetch raw text endpoint with caching and polite rate limiting."""
        req_headers = {"User-Agent": self.USER_AGENT}
        if headers:
            req_headers.update(headers)

        full_url = str(httpx.URL(url, params=params))

        cached = await self.cache.get(full_url)
        if cached:
            return cached.body

        await self.limiter.acquire(url)

        async with httpx.AsyncClient(headers=req_headers, timeout=20.0) as client:
            resp = await client.get(url, params=params)
            if resp.status_code == 429:
                self.limiter.handle_rate_limit_response(url, resp.headers.get("Retry-After"))
                resp.raise_for_status()
            resp.raise_for_status()

            await self.cache.put(
                url=full_url,
                status_code=resp.status_code,
                headers=dict(resp.headers),
                body=resp.text,
                etag=resp.headers.get("ETag", ""),
                last_modified=resp.headers.get("Last-Modified", ""),
                ttl_seconds=ttl_seconds,
            )
            return resp.text
