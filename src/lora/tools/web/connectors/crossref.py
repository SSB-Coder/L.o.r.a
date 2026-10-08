"""Crossref DOI & Scholarly Metadata Connector for L.O.R.A."""


from lora.tools.web.cache import HttpCache
from lora.tools.web.connectors.base import BaseConnector, Document, ReliabilityTier, SourceHit
from lora.tools.web.limiter import TokenBucketLimiter


class CrossrefConnector(BaseConnector):
    """Crossref API Connector for peer-reviewed academic literature."""

    WORKS_API = "https://api.crossref.org/works"

    def __init__(
        self,
        limiter: TokenBucketLimiter | None = None,
        cache: HttpCache | None = None,
    ) -> None:
        super().__init__(
            name="crossref",
            reliability_tier=ReliabilityTier.T1,
            limiter=limiter,
            cache=cache,
        )

    async def search(self, query: str, limit: int = 10) -> list[SourceHit]:
        """Search Crossref works for matching query."""
        params = {
            "query": query.strip(),
            "rows": min(limit, 20),
        }
        headers = {
            "User-Agent": "LORA/0.1.0 (mailto:lora-research@localhost.localdomain)",
        }

        try:
            data = await self.get_json(self.WORKS_API, params=params, headers=headers)
            items = data.get("message", {}).get("items", [])
            hits = []

            for item in items:
                titles = item.get("title", [])
                title = titles[0] if titles else "Untitled Publication"
                doi = item.get("DOI", "")
                url = item.get("URL") or f"https://doi.org/{doi}"

                # Extract authors
                authors = []
                for a in item.get("author", []):
                    given = a.get("given", "")
                    family = a.get("family", "")
                    name = f"{given} {family}".strip()
                    if name:
                        authors.append(name)

                # Extract publish date
                created = item.get("created", {}).get("date-time", "")

                abstract = item.get("abstract", "")
                # Clean simple tags like <jats:p>
                if "<" in abstract and ">" in abstract:
                    import re
                    abstract = re.sub(r"<[^>]+>", "", abstract).strip()

                hits.append(SourceHit(
                    id=f"crossref:{doi}",
                    title=title,
                    url=url,
                    source_name="crossref",
                    reliability_tier=self.reliability_tier,
                    snippet=abstract[:350] + "..." if len(abstract) > 350 else abstract,
                    is_preprint=False,
                    authors=authors,
                    published_date=created,
                    metadata={"doi": doi, "abstract": abstract, "publisher": item.get("publisher", "")},
                ))

            return hits
        except Exception:
            return []

    async def fetch(self, hit: SourceHit) -> Document:
        """Return Document from Crossref metadata."""
        abstract = hit.metadata.get("abstract") or hit.snippet
        return Document(
            id=hit.id,
            url=hit.url,
            title=hit.title,
            authors=hit.authors,
            published_date=hit.published_date,
            source="crossref",
            reliability_tier=self.reliability_tier,
            is_preprint=False,
            license="",
            text=abstract,
            summary=hit.snippet,
            metadata=hit.metadata,
        )
