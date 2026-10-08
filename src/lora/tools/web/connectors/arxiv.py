"""arXiv Preprint Knowledge Connector for L.O.R.A."""

import xml.etree.ElementTree as ET

from lora.tools.web.cache import HttpCache
from lora.tools.web.connectors.base import BaseConnector, Document, ReliabilityTier, SourceHit
from lora.tools.web.limiter import TokenBucketLimiter


class ArxivConnector(BaseConnector):
    """arXiv API Connector with mandatory 1 req / 3s rate limiting."""

    QUERY_API = "http://export.arxiv.org/api/query"
    ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

    def __init__(
        self,
        limiter: TokenBucketLimiter | None = None,
        cache: HttpCache | None = None,
    ) -> None:
        super().__init__(
            name="arxiv",
            reliability_tier=ReliabilityTier.T1_P,  # Unreviewed preprint
            limiter=limiter,
            cache=cache,
        )

    async def search(self, query: str, limit: int = 10) -> list[SourceHit]:
        """Search arXiv preprints for given query."""
        clean_query = query.strip()
        params = {
            "search_query": f"all:{clean_query}",
            "start": 0,
            "max_results": min(limit, 25),
        }

        try:
            xml_text = await self.get_text(self.QUERY_API, params=params)
            root = ET.fromstring(xml_text)
            hits = []

            for entry in root.findall("atom:entry", self.ATOM_NS):
                id_elem = entry.find("atom:id", self.ATOM_NS)
                title_elem = entry.find("atom:title", self.ATOM_NS)
                summary_elem = entry.find("atom:summary", self.ATOM_NS)
                published_elem = entry.find("atom:published", self.ATOM_NS)

                raw_id = id_elem.text.strip() if id_elem is not None and id_elem.text else ""
                title = " ".join((title_elem.text or "").split()) if title_elem is not None else ""
                summary = " ".join((summary_elem.text or "").split()) if summary_elem is not None else ""
                published = published_elem.text.strip() if published_elem is not None and published_elem.text else ""

                authors = []
                for author_elem in entry.findall("atom:author", self.ATOM_NS):
                    name_elem = author_elem.find("atom:name", self.ATOM_NS)
                    if name_elem is not None and name_elem.text:
                        authors.append(name_elem.text.strip())

                # PDF link if present
                pdf_url = ""
                for link_elem in entry.findall("atom:link", self.ATOM_NS):
                    if link_elem.attrib.get("title") == "pdf":
                        pdf_url = link_elem.attrib.get("href", "")

                arxiv_id = raw_id.split("/abs/")[-1] if "/abs/" in raw_id else raw_id

                hits.append(SourceHit(
                    id=f"arxiv:{arxiv_id}",
                    title=title,
                    url=raw_id,
                    source_name="arxiv",
                    reliability_tier=self.reliability_tier,
                    snippet=summary[:350] + "..." if len(summary) > 350 else summary,
                    is_preprint=True,
                    authors=authors,
                    published_date=published,
                    metadata={"summary": summary, "pdf_url": pdf_url, "arxiv_id": arxiv_id},
                ))

            return hits
        except Exception:
            return []

    async def fetch(self, hit: SourceHit) -> Document:
        """Return Document from hit metadata."""
        full_summary = hit.metadata.get("summary") or hit.snippet
        return Document(
            id=hit.id,
            url=hit.url,
            title=hit.title,
            authors=hit.authors,
            published_date=hit.published_date,
            source="arxiv",
            reliability_tier=self.reliability_tier,
            is_preprint=True,
            license="arXiv.org perpetual non-exclusive license",
            text=full_summary,
            summary=hit.snippet,
            metadata=hit.metadata,
        )
