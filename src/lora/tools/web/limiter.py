"""Asynchronous Token-Bucket Rate Limiter with Domain Policies for L.O.R.A."""

import asyncio
import time
from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass
class RatePolicy:
    requests_per_second: float
    burst_capacity: int = 1
    min_interval_seconds: float = 0.0


class TokenBucketLimiter:
    """Manages asynchronous per-domain rate limiting and backoff."""

    # Default policies for known services
    DOMAIN_POLICIES: dict[str, RatePolicy] = {
        # arXiv allows max 1 request every 3 seconds
        "export.arxiv.org": RatePolicy(requests_per_second=0.333, burst_capacity=1, min_interval_seconds=3.0),
        "arxiv.org": RatePolicy(requests_per_second=0.333, burst_capacity=1, min_interval_seconds=3.0),
        # Wikimedia APIs: 1 rps polite serial access
        "en.wikipedia.org": RatePolicy(requests_per_second=1.0, burst_capacity=1, min_interval_seconds=1.0),
        "api.wikimedia.org": RatePolicy(requests_per_second=1.0, burst_capacity=1, min_interval_seconds=1.0),
        "query.wikidata.org": RatePolicy(requests_per_second=0.5, burst_capacity=1, min_interval_seconds=2.0),
        # Crossref polite pool: ~3 rps
        "api.crossref.org": RatePolicy(requests_per_second=3.0, burst_capacity=3, min_interval_seconds=0.33),
        # Europe PMC / NCBI: 3 rps
        "europepmc.org": RatePolicy(requests_per_second=3.0, burst_capacity=3, min_interval_seconds=0.33),
        "eutils.ncbi.nlm.nih.gov": RatePolicy(requests_per_second=3.0, burst_capacity=3, min_interval_seconds=0.33),
        # Open Library / HN: 2 rps
        "openlibrary.org": RatePolicy(requests_per_second=2.0, burst_capacity=2, min_interval_seconds=0.5),
        "hn.algolia.com": RatePolicy(requests_per_second=5.0, burst_capacity=5, min_interval_seconds=0.2),
        # GitHub REST: 1 rps
        "api.github.com": RatePolicy(requests_per_second=1.0, burst_capacity=2, min_interval_seconds=1.0),
    }

    DEFAULT_POLICY = RatePolicy(requests_per_second=2.0, burst_capacity=2, min_interval_seconds=0.5)

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._tokens: dict[str, float] = {}
        self._last_update: dict[str, float] = {}
        self._last_request_time: dict[str, float] = {}
        self._retry_after_until: dict[str, float] = {}

    def _get_domain(self, target: str) -> str:
        if "://" in target:
            parsed = urlparse(target)
            return (parsed.hostname or "").lower()
        return target.lower()

    def get_policy(self, domain: str) -> RatePolicy:
        dom = domain.lower()
        if dom in self.DOMAIN_POLICIES:
            return self.DOMAIN_POLICIES[dom]
        for known, pol in self.DOMAIN_POLICIES.items():
            if dom.endswith("." + known):
                return pol
        return self.DEFAULT_POLICY

    async def acquire(self, target: str) -> float:
        """Wait until a request is permitted for the given domain or URL.

        Returns the number of seconds waited.
        """
        domain = self._get_domain(target)
        if domain not in self._locks:
            self._locks[domain] = asyncio.Lock()

        async with self._locks[domain]:
            now = time.monotonic()

            # 1. Check if domain is in an active Retry-After backoff window
            blocked_until = self._retry_after_until.get(domain, 0.0)
            if now < blocked_until:
                wait_retry = blocked_until - now
                await asyncio.sleep(wait_retry)
                now = time.monotonic()

            policy = self.get_policy(domain)

            # 2. Enforce minimum interval between consecutive requests
            last_req = self._last_request_time.get(domain, 0.0)
            elapsed_since_last = now - last_req
            if policy.min_interval_seconds > 0 and elapsed_since_last < policy.min_interval_seconds:
                interval_wait = policy.min_interval_seconds - elapsed_since_last
                await asyncio.sleep(interval_wait)
                now = time.monotonic()

            # 3. Token bucket replenishment
            tokens = self._tokens.get(domain, float(policy.burst_capacity))
            last_time = self._last_update.get(domain, now)
            elapsed = now - last_time

            tokens = min(float(policy.burst_capacity), tokens + elapsed * policy.requests_per_second)

            waited = 0.0
            if tokens < 1.0:
                deficit = 1.0 - tokens
                wait_needed = deficit / policy.requests_per_second
                await asyncio.sleep(wait_needed)
                tokens = 0.0
                now = time.monotonic()
                waited = wait_needed
            else:
                tokens -= 1.0

            self._tokens[domain] = tokens
            self._last_update[domain] = now
            self._last_request_time[domain] = now

            return waited

    def handle_rate_limit_response(self, target: str, retry_after_header: str | None = None) -> None:
        """Handle 429 Too Many Requests response and set Retry-After backoff."""
        domain = self._get_domain(target)
        wait_seconds = 5.0
        if retry_after_header:
            try:
                wait_seconds = max(1.0, float(retry_after_header.strip()))
            except ValueError:
                wait_seconds = 10.0

        self._retry_after_until[domain] = time.monotonic() + wait_seconds
