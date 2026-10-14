import httpx
import pytest
import respx

from lora.agent.research import extract_text, parse_ddg_html, research

DDG = """
<div class="result"><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fa&rut=x">Alpha page</a>
<a class="result__snippet">Alpha snippet</a></div>
<div class="result"><a class="result__a" href="https://duckduckgo.com/y.js?ad=1">Ad</a></div>
<div class="result"><a class="result__a" href="https://other.net/b">Beta page</a><a class="result__snippet">Beta snippet</a></div>
"""
PAGE = "<html><body><article><h1>Alpha</h1>" + "<p>Real article text about the subject. </p>" * 20 + "</article></body></html>"


def test_parse_ddg_decodes_redirects_and_skips_ads():
    hits = parse_ddg_html(DDG)
    assert [h.url for h in hits] == ["https://example.org/a", "https://other.net/b"]
    assert hits[0].title == "Alpha page" and hits[0].snippet == "Alpha snippet" and hits[0].domain == "example.org"


def test_extract_text_strips_markup():
    assert "Real article text" in extract_text(PAGE) and "<p>" not in extract_text(PAGE)


@pytest.mark.asyncio
@respx.mock
async def test_research_end_to_end_with_fake_web():
    respx.post("https://html.duckduckgo.com/html/").respond(200, text=DDG)
    respx.get("https://example.org/a").respond(200, text=PAGE, headers={"content-type": "text/html"})
    respx.get("https://other.net/b").respond(404)
    seen = []
    res = await research("subject", status=seen.append)
    assert seen[0].startswith("Searching") and "Reading" in seen[1]
    assert [s.id for s in res.sources] == ["S1", "S2"]  # page text for S1, snippet-only for S2
    assert "Real article text" in res.prompt_block()
    assert res.for_ui()[0]["url"] == "https://example.org/a"


@pytest.mark.asyncio
@respx.mock
async def test_private_hosts_are_never_fetched():
    html = '<div class="result"><a class="result__a" href="http://127.0.0.1:8765/api">local</a></div>'
    respx.post("https://html.duckduckgo.com/html/").respond(200, text=html)
    res = await research("x")
    assert res.sources == []


@pytest.mark.asyncio
@respx.mock
async def test_falls_back_to_wikipedia_when_ddg_fails():
    respx.post("https://html.duckduckgo.com/html/").mock(side_effect=httpx.ConnectError("down"))
    respx.get("https://en.wikipedia.org/w/api.php").respond(
        200, json={"query": {"search": [{"title": "Mamba (architecture)", "snippet": "A <b>state space</b> model."}]}}
    )
    respx.get("https://en.wikipedia.org/wiki/Mamba_(architecture)").respond(404)
    res = await research("mamba")
    assert res.sources and res.sources[0].url.endswith("Mamba_(architecture)") and "state space" in res.sources[0].snippet
