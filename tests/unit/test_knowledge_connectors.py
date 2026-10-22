"""Unit tests for Phase 4 Knowledge and Search Connectors."""

from pathlib import Path

import httpx
import pytest
import respx

from lora.tools.web.cache import HttpCache
from lora.tools.web.connectors.arxiv import ArxivConnector
from lora.tools.web.connectors.base import ReliabilityTier
from lora.tools.web.connectors.crossref import CrossrefConnector
from lora.tools.web.connectors.duckduckgo import DuckDuckGoConnector
from lora.tools.web.connectors.github import GitHubConnector
from lora.tools.web.connectors.hacker_news import HackerNewsConnector
from lora.tools.web.connectors.open_library import OpenLibraryConnector
from lora.tools.web.connectors.pubmed import PubMedConnector
from lora.tools.web.connectors.wikipedia import WikipediaConnector
from lora.tools.web.limiter import TokenBucketLimiter
from lora.tools.web.search import WebSearchDispatcher


@pytest.fixture
async def mock_env(tmp_path: Path):
    db_file = tmp_path / "conn_cache.db"
    cache = HttpCache(db_path=db_file)
    limiter = TokenBucketLimiter()
    yield cache, limiter
    await cache.close()


@pytest.mark.asyncio
@respx.mock
async def test_wikipedia_connector(mock_env):
    cache, limiter = mock_env
    connector = WikipediaConnector(cache=cache, limiter=limiter)

    # Mock search
    respx.get("https://en.wikipedia.org/w/api.php").mock(
        return_value=httpx.Response(200, json={
            "query": {
                "search": [
                    {"title": "Quantum computing", "pageid": 12345, "snippet": "A field of computing", "wordcount": 5000}
                ]
            }
        })
    )

    hits = await connector.search("quantum computing")
    assert len(hits) == 1
    assert hits[0].title == "Quantum computing"
    assert hits[0].reliability_tier == ReliabilityTier.T2

    # Mock summary fetch
    respx.get("https://en.wikipedia.org/api/rest_v1/page/summary/Quantum_computing").mock(
        return_value=httpx.Response(200, json={
            "title": "Quantum computing",
            "extract": "Quantum computing is a rapidly-emerging technology.",
            "description": "Subfield of quantum information science",
            "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Quantum_computing"}},
        })
    )

    doc = await connector.fetch(hits[0])
    assert doc.title == "Quantum computing"
    assert "rapidly-emerging" in doc.text
    assert doc.license == "CC BY-SA 4.0"


@pytest.mark.asyncio
@respx.mock
async def test_arxiv_connector(mock_env):
    cache, limiter = mock_env
    connector = ArxivConnector(cache=cache, limiter=limiter)

    mock_atom = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>http://arxiv.org/abs/2401.00001v1</id>
        <title>Advances in Quantum Error Correction</title>
        <summary>We present a new fault-tolerant threshold theorem.</summary>
        <published>2024-01-01T00:00:00Z</published>
        <author><name>Alice Smith</name></author>
        <author><name>Bob Jones</name></author>
        <link title="pdf" href="http://arxiv.org/pdf/2401.00001v1"/>
      </entry>
    </feed>
    """

    respx.get("http://export.arxiv.org/api/query").mock(
        return_value=httpx.Response(200, text=mock_atom)
    )

    hits = await connector.search("quantum error correction")
    assert len(hits) == 1
    assert hits[0].title == "Advances in Quantum Error Correction"
    assert hits[0].is_preprint is True
    assert hits[0].reliability_tier == ReliabilityTier.T1_P
    assert "Alice Smith" in hits[0].authors

    doc = await connector.fetch(hits[0])
    assert doc.is_preprint is True
    assert "fault-tolerant" in doc.text


@pytest.mark.asyncio
@respx.mock
async def test_crossref_connector(mock_env):
    cache, limiter = mock_env
    connector = CrossrefConnector(cache=cache, limiter=limiter)

    respx.get("https://api.crossref.org/works").mock(
        return_value=httpx.Response(200, json={
            "message": {
                "items": [
                    {
                        "title": ["Superconducting Quantum Circuits"],
                        "DOI": "10.1103/PhysRevA.100.00000",
                        "author": [{"given": "John", "family": "Doe"}],
                        "created": {"date-time": "2024-02-01T10:00:00Z"},
                        "abstract": "<jats:p>Study of transmons.</jats:p>",
                    }
                ]
            }
        })
    )

    hits = await connector.search("transmon qubits")
    assert len(hits) == 1
    assert hits[0].title == "Superconducting Quantum Circuits"
    assert hits[0].reliability_tier == ReliabilityTier.T1
    assert "John Doe" in hits[0].authors


@pytest.mark.asyncio
@respx.mock
async def test_pubmed_connector(mock_env):
    cache, limiter = mock_env
    connector = PubMedConnector(cache=cache, limiter=limiter)

    respx.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi").mock(
        return_value=httpx.Response(200, json={
            "esearchresult": {"idlist": ["98765432"]}
        })
    )

    respx.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi").mock(
        return_value=httpx.Response(200, json={
            "result": {
                "98765432": {
                    "title": "CRISPR-Cas9 Therapeutics in Human Trials",
                    "authors": [{"name": "Smith J"}],
                    "pubdate": "2024",
                    "source": "Nature Medicine",
                }
            }
        })
    )

    hits = await connector.search("CRISPR therapeutics")
    assert len(hits) == 1
    assert "CRISPR-Cas9" in hits[0].title
    assert hits[0].reliability_tier == ReliabilityTier.T1


@pytest.mark.asyncio
@respx.mock
async def test_github_connector(mock_env):
    cache, limiter = mock_env
    connector = GitHubConnector(cache=cache, limiter=limiter)

    respx.get("https://api.github.com/search/repositories").mock(
        return_value=httpx.Response(200, json={
            "items": [
                {
                    "full_name": "google/jax",
                    "description": "Autograd and XLA in Python",
                    "stargazers_count": 28000,
                    "html_url": "https://github.com/google/jax",
                    "updated_at": "2026-01-01",
                    "owner": {"login": "google"},
                }
            ]
        })
    )

    hits = await connector.search("jax")
    assert len(hits) == 1
    assert hits[0].title == "google/jax"
    assert hits[0].reliability_tier == ReliabilityTier.T3


@pytest.mark.asyncio
@respx.mock
async def test_hacker_news_connector(mock_env):
    cache, limiter = mock_env
    connector = HackerNewsConnector(cache=cache, limiter=limiter)

    respx.get("https://hn.algolia.com/api/v1/search").mock(
        return_value=httpx.Response(200, json={
            "hits": [
                {
                    "title": "Show HN: Fast local inference engine",
                    "author": "dev123",
                    "points": 340,
                    "num_comments": 85,
                    "objectID": "40001",
                    "url": "https://example.com/fast-engine",
                }
            ]
        })
    )

    hits = await connector.search("inference engine")
    assert len(hits) == 1
    assert hits[0].reliability_tier == ReliabilityTier.T4


@pytest.mark.asyncio
@respx.mock
async def test_open_library_connector(mock_env):
    cache, limiter = mock_env
    connector = OpenLibraryConnector(cache=cache, limiter=limiter)

    respx.get("https://openlibrary.org/search.json").mock(
        return_value=httpx.Response(200, json={
            "docs": [
                {
                    "title": "Quantum Computation and Quantum Information",
                    "author_name": ["Michael A. Nielsen", "Isaac L. Chuang"],
                    "first_publish_year": 2000,
                    "key": "/works/OL12345W",
                }
            ]
        })
    )

    hits = await connector.search("quantum computation nielsen chuang")
    assert len(hits) == 1
    assert "Nielsen" in hits[0].snippet


@pytest.mark.asyncio
@respx.mock
async def test_duckduckgo_connector(mock_env):
    cache, limiter = mock_env
    connector = DuckDuckGoConnector(cache=cache, limiter=limiter)

    respx.get("https://api.duckduckgo.com/").mock(
        return_value=httpx.Response(200, json={
            "Heading": "Python (programming language)",
            "AbstractText": "Python is a high-level, general-purpose programming language.",
            "AbstractURL": "https://en.wikipedia.org/wiki/Python_(programming_language)",
            "RelatedTopics": [],
        })
    )

    hits = await connector.search("python programming")
    assert len(hits) == 1
    assert "high-level" in hits[0].snippet


@pytest.mark.asyncio
@respx.mock
async def test_web_search_dispatcher():
    dispatcher = WebSearchDispatcher()

    # Mock Wikipedia response for search
    respx.get("https://en.wikipedia.org/w/api.php").mock(
        return_value=httpx.Response(200, json={
            "query": {"search": [{"title": "Photosynthesis", "pageid": 999, "snippet": "Light to energy"}]}
        })
    )

    res = await dispatcher.search("Photosynthesis", source="wikipedia")
    assert res["status"] == "success"
    assert res["count"] == 1
    assert res["hits"][0]["title"] == "Photosynthesis"
    await dispatcher.close()
