"""Unit tests for TokenBucketLimiter and HttpCache."""

import asyncio
from pathlib import Path

import pytest

from lora.tools.web.cache import HttpCache
from lora.tools.web.limiter import TokenBucketLimiter


@pytest.mark.asyncio
async def test_token_bucket_limiter_basic_acquisition():
    limiter = TokenBucketLimiter()
    # First request should have no or negligible wait
    waited = await limiter.acquire("https://example.com/test")
    assert waited == 0.0


@pytest.mark.asyncio
async def test_token_bucket_limiter_retry_after():
    limiter = TokenBucketLimiter()
    limiter.handle_rate_limit_response("https://example.com/test", retry_after_header="0.1")
    # Next request should pause for at least 0.08 seconds
    start = asyncio.get_event_loop().time()
    await limiter.acquire("https://example.com/test")
    elapsed = asyncio.get_event_loop().time() - start
    assert elapsed >= 0.08


@pytest.mark.asyncio
async def test_http_cache_put_get_expiry(tmp_path: Path):
    db_file = tmp_path / "test_cache.db"
    cache = HttpCache(db_path=db_file)

    url = "https://api.example.com/data"
    headers = {"Content-Type": "application/json"}
    body = '{"value": 42}'

    # 1. Put and immediate get
    await cache.put(url=url, status_code=200, headers=headers, body=body, ttl_seconds=10.0)
    cached = await cache.get(url)
    assert cached is not None
    assert cached.status_code == 200
    assert cached.body == body
    assert cached.headers.get("Content-Type") == "application/json"

    # 2. Put with 0 TTL (expired)
    expired_url = "https://api.example.com/expired"
    await cache.put(url=expired_url, status_code=200, headers={}, body="old", ttl_seconds=-1.0)
    cached_exp = await cache.get(expired_url)
    assert cached_exp is None

    # 3. Clear expired
    deleted = await cache.clear_expired()
    assert deleted >= 1
    await cache.close()
