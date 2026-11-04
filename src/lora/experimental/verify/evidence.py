"""Select claim-relevant, verbatim evidence without silently dropping its tail."""

import re


def relevant_excerpt(premise: str, claim: str, limit: int = 2400) -> str:
    if len(premise) <= limit:
        return premise
    terms = set(re.findall(r"\b\w{3,}\b", claim.lower())) - {
        "the",
        "and",
        "that",
        "this",
        "with",
        "from",
        "are",
        "was",
        "were",
        "has",
        "have",
    }
    # Overlapping windows retain original whitespace, punctuation and quotes.
    width = min(800, limit)
    windows = [(i, premise[i : i + width]) for i in range(0, len(premise), width // 2)]
    windows.sort(
        key=lambda item: len(terms & set(re.findall(r"\b\w{3,}\b", item[1].lower()))), reverse=True
    )
    selected = []
    for start, text in windows:
        end = start + len(text)
        if any(start < b and end > a for a, b in selected):
            continue
        if sum(b - a for a, b in selected) + len(text) + len(selected) * 2 > limit:
            continue
        selected.append((start, end))
        if len(selected) >= 3:
            break
    return "\n\n".join(premise[a:b] for a, b in sorted(selected))[:limit]
