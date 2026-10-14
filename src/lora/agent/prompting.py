"""The prompt engineer: turns a messy spoken request into a clean question and good search queries.

Speech-to-text hands us things like "Can you tell me more about, uh, that thing's price?". Searching for that
literally gives junk (your log showed it searching for "Can You, Browse the internet?"). One quick call to
the writer model rewrites it first, using the last turns of chat to resolve "it" / "that".
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

PLAN_SYSTEM = (
    "You rewrite a user's request for a web research assistant. The request may be misheard speech. "
    "Reply with ONLY JSON: {\"question\": \"the request as one clear standalone question (resolve words like it, "
    "that, they using the recent chat)\", \"queries\": [\"2 or 3 short web search queries, keywords only\"]}"
)
_JSON = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class ResearchPlan:
    question: str
    queries: list[str] = field(default_factory=list)
    engineered: bool = False


def parse_plan(raw: str, fallback: str) -> ResearchPlan:
    m = _JSON.search(raw or "")
    if not m:
        return ResearchPlan(fallback, [fallback])
    try:
        data = json.loads(m.group(0))
    except Exception:
        return ResearchPlan(fallback, [fallback])
    question = str(data.get("question") or "").strip() or fallback
    queries = [str(q).strip() for q in (data.get("queries") or []) if str(q).strip() and len(str(q)) < 120][:3]
    if not queries:
        queries = [question]
    return ResearchPlan(question[:300], queries, engineered=True)


async def make_plan(
    collect: Callable[[list[dict[str, str]]], Awaitable[str]], query: str, recent: list[dict[str, str]]
) -> ResearchPlan:
    """Never raises: on any problem the original request is used as is."""
    msgs = [{"role": "system", "content": PLAN_SYSTEM}, *recent[-4:], {"role": "user", "content": query}]
    try:
        return parse_plan(await collect(msgs), query)
    except Exception:
        return ResearchPlan(query, [query])
