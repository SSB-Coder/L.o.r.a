"""Web research: search -> fetch pages -> numbered sources for a cited answer.

No browser needed: DuckDuckGo's HTML endpoint for search, httpx + trafilatura to read pages.
All fetches go through the egress filter (http/https only, no private/loopback hosts).
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from urllib.parse import parse_qs, unquote, urlparse

import httpx

from lora.net import async_client

logger = logging.getLogger(__name__)

NO_TEXT_DOMAINS = {
    "youtube.com", "youtu.be", "m.youtube.com", "tiktok.com", "instagram.com", "facebook.com", "x.com", "twitter.com",
    "pinterest.com", "linkedin.com",
}  # video / login-walled sites: nothing readable to cite
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) LORA/0.1 (+local research assistant)"
MAX_BYTES = 1_500_000
PAGE_CHARS = 2200


@dataclass
class Source:
    id: str
    title: str
    url: str
    text: str = ""
    snippet: str = ""

    @property
    def domain(self) -> str:
        return (urlparse(self.url).hostname or "").removeprefix("www.")


@dataclass
class ResearchResult:
    query: str
    sources: list[Source] = field(default_factory=list)

    def prompt_block(self) -> str:
        parts = []
        for s in self.sources:
            body = (s.text or s.snippet).strip()
            parts.append(f"[{s.id}] {s.title} ({s.domain})\n{body}")
        return "\n\n".join(parts)

    def for_ui(self) -> list[dict[str, str]]:
        return [{"id": s.id, "title": s.title, "url": s.url, "domain": s.domain} for s in self.sources]


def _egress_ok(url: str) -> bool:
    try:
        from lora.safety.egress import EgressFilter

        return EgressFilter().check_egress(url).allowed
    except Exception:  # if the guard itself is unavailable, fall back to a basic scheme/host check
        p = urlparse(url)
        return p.scheme in ("http", "https") and (p.hostname or "") not in ("localhost", "127.0.0.1", "::1")


def _decode_ddg(href: str) -> str:
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    if "duckduckgo.com" in (parsed.hostname or "") and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        return unquote(target)
    return href


def parse_ddg_html(html: str, limit: int = 8) -> list[Source]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    out: list[Source] = []
    for res in soup.select("div.result, div.web-result"):
        a = res.select_one("a.result__a")
        if not a or not a.get("href"):
            continue
        url = _decode_ddg(a["href"])
        if not url.startswith("http") or "duckduckgo.com/y.js" in url:
            continue  # ads
        snip = res.select_one(".result__snippet")
        out.append(
            Source(
                id=f"S{len(out) + 1}",
                title=a.get_text(" ", strip=True)[:140],
                url=url,
                snippet=snip.get_text(" ", strip=True) if snip else "",
            )
        )
        if len(out) >= limit:
            break
    return out


HEADERS = {
    "User-Agent": UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def parse_ddg_lite(html: str, limit: int = 8) -> list[Source]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    out: list[Source] = []
    for a in soup.select("a.result-link"):
        url = _decode_ddg(a.get("href", ""))
        if not url.startswith("http") or "duckduckgo.com/y.js" in url:
            continue
        row = a.find_parent("tr")
        snip_row = row.find_next_sibling("tr") if row else None
        snip = snip_row.select_one(".result-snippet") if snip_row else None
        out.append(Source(f"S{len(out) + 1}", a.get_text(" ", strip=True)[:140], url, snippet=snip.get_text(" ", strip=True) if snip else ""))
        if len(out) >= limit:
            break
    return out


def _decode_bing(href: str) -> str:
    """Bing wraps result links as /ck/a?...&u=a1<base64url of the real URL>."""
    parsed = urlparse(href)
    if (parsed.hostname or "").endswith("bing.com") and parsed.path.startswith("/ck/a"):
        raw = parse_qs(parsed.query).get("u", [""])[0]
        if raw.startswith("a1"):
            import base64

            try:
                payload = raw[2:] + "=" * (-len(raw[2:]) % 4)
                return base64.urlsafe_b64decode(payload).decode("utf-8")
            except Exception:
                return ""
    return href


def parse_bing_html(html: str, limit: int = 8) -> list[Source]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    out: list[Source] = []
    for li in soup.select("li.b_algo"):
        a = li.select_one("h2 a")
        if not a or not a.get("href"):
            continue
        url = _decode_bing(a["href"])
        if not url.startswith("http") or (urlparse(url).hostname or "").endswith("bing.com"):
            continue
        snip = li.select_one(".b_caption p") or li.select_one("p")
        out.append(Source(f"S{len(out) + 1}", a.get_text(" ", strip=True)[:140], url, snippet=snip.get_text(" ", strip=True) if snip else ""))
        if len(out) >= limit:
            break
    return out


async def _wikipedia(q: str, client: httpx.AsyncClient, limit: int) -> list[Source]:
    res = await client.get(
        "https://en.wikipedia.org/w/api.php",
        params={"action": "query", "list": "search", "srsearch": q, "format": "json", "srlimit": min(limit, 5)},
    )
    hits = []
    for i, item in enumerate(res.json().get("query", {}).get("search", []), 1):
        title = item["title"]
        snippet = re.sub(r"<[^>]+>", "", item.get("snippet", ""))
        hits.append(Source(f"S{i}", title, "https://en.wikipedia.org/wiki/" + title.replace(" ", "_"), snippet=snippet))
    return hits


async def search_with_report(query: str, client: httpx.AsyncClient, limit: int = 8) -> tuple[list[Source], dict]:
    """Try several search backends in turn. The report says which one worked and why others did not."""
    q = query.strip()
    report: dict = {"backend": None, "errors": []}

    async def duckduckgo_html() -> list[Source]:
        res = await client.post("https://html.duckduckgo.com/html/", data={"q": q}, headers=HEADERS)
        if res.status_code != 200:
            raise RuntimeError(f"HTTP {res.status_code} (probably a bot check)")
        return parse_ddg_html(res.text, limit)

    async def duckduckgo_lite() -> list[Source]:
        res = await client.get("https://lite.duckduckgo.com/lite/", params={"q": q}, headers=HEADERS)
        if res.status_code != 200:
            raise RuntimeError(f"HTTP {res.status_code}")
        return parse_ddg_lite(res.text, limit)

    async def bing() -> list[Source]:
        res = await client.get("https://www.bing.com/search", params={"q": q, "setlang": "en"}, headers=HEADERS)
        if res.status_code != 200:
            raise RuntimeError(f"HTTP {res.status_code}")
        return parse_bing_html(res.text, limit)

    async def wikipedia() -> list[Source]:
        return await _wikipedia(q, client, limit)

    for name, fn in (("duckduckgo", duckduckgo_html), ("duckduckgo-lite", duckduckgo_lite), ("bing", bing), ("wikipedia", wikipedia)):
        try:
            hits = await fn()
        except Exception as exc:
            report["errors"].append(f"{name}: {type(exc).__name__}: {exc}")
            logger.info("Search backend %s failed: %s", name, exc)
            continue
        if hits:
            report["backend"] = name
            return hits, report
        report["errors"].append(f"{name}: no results")
    return [], report


async def web_search(query: str, client: httpx.AsyncClient, limit: int = 8) -> list[Source]:
    hits, report = await search_with_report(query, client, limit)
    if not hits:
        logger.warning("Web search found nothing for %r: %s", query, "; ".join(report["errors"]))
    else:
        logger.info("Web search via %s: %d results", report["backend"], len(hits))
    return hits


def extract_text(html: str) -> str:
    try:
        import trafilatura

        text = trafilatura.extract(html, include_comments=False, include_tables=False) or ""
        if text.strip():
            return text.strip()
    except Exception:
        pass
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for t in soup(["script", "style", "nav", "footer", "header", "aside"]):
            t.decompose()
        return re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
    except Exception:
        return ""


async def fetch_text(url: str, client: httpx.AsyncClient) -> str:
    if not _egress_ok(url):
        return ""
    try:
        async with client.stream("GET", url) as res:
            ctype = res.headers.get("content-type", "")
            if res.status_code != 200 or not any(t in ctype for t in ("html", "text/plain", "xml")):
                return ""
            buf = bytearray()
            async for chunk in res.aiter_bytes():
                buf.extend(chunk)
                if len(buf) > MAX_BYTES:
                    break
            html = bytes(buf).decode(res.encoding or "utf-8", errors="replace")
    except Exception as exc:
        logger.info("Fetch failed for %s: %s", url, exc)
        return ""
    return extract_text(html)[:PAGE_CHARS]


async def research(
    query: str, status=lambda _t: None, max_pages: int = 4, queries: list[str] | None = None
) -> ResearchResult:
    """Search the web and read the best few pages. ``queries`` are the prompt engineer's search phrases."""
    phrases = list(dict.fromkeys(q.strip() for q in (queries or [query]) if q and q.strip()))[:2] or [query]
    async with async_client(HEADERS, httpx.Timeout(12.0, connect=6.0)) as client:
        status("Searching the web…")
        lists = [await web_search(q, client) for q in phrases]
        hits: list[Source] = []
        seen_urls: set[str] = set()
        for i in range(max((len(x) for x in lists), default=0)):  # interleave so each query contributes its best hits
            for lst in lists:
                if i < len(lst) and lst[i].url not in seen_urls:
                    seen_urls.add(lst[i].url)
                    hits.append(lst[i])
        picked: list[Source] = []
        seen: set[str] = set()
        for h in hits:
            if h.domain in seen or h.domain in NO_TEXT_DOMAINS or not _egress_ok(h.url):
                continue
            seen.add(h.domain)
            picked.append(h)
            if len(picked) >= max_pages:
                break
        if not picked:
            return ResearchResult(query)
        status(f"Reading {len(picked)} pages…")
        texts = await asyncio.gather(*(fetch_text(s.url, client) for s in picked), return_exceptions=True)
        for s, t in zip(picked, texts, strict=True):
            s.text = t if isinstance(t, str) else ""
        usable = [s for s in picked if len(s.text) >= 200 or s.snippet]
        for i, s in enumerate(usable, 1):
            s.id = f"S{i}"
        return ResearchResult(query, usable)
