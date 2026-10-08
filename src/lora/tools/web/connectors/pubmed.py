"""PubMed / NCBI E-Utilities Biomedical Connector for L.O.R.A."""


from lora.tools.web.cache import HttpCache
from lora.tools.web.connectors.base import BaseConnector, Document, ReliabilityTier, SourceHit
from lora.tools.web.limiter import TokenBucketLimiter


class PubMedConnector(BaseConnector):
    """PubMed / NCBI E-utilities Connector for biomedical citations."""

    ESEARCH_API = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    ESUMMARY_API = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

    def __init__(
        self,
        limiter: TokenBucketLimiter | None = None,
        cache: HttpCache | None = None,
    ) -> None:
        super().__init__(
            name="pubmed",
            reliability_tier=ReliabilityTier.T1,
            limiter=limiter,
            cache=cache,
        )

    async def search(self, query: str, limit: int = 10) -> list[SourceHit]:
        """Search PubMed IDs and fetch bibliographic summaries."""
        search_params = {
            "db": "pubmed",
            "term": query.strip(),
            "retmode": "json",
            "retmax": min(limit, 15),
        }

        try:
            search_data = await self.get_json(self.ESEARCH_API, params=search_params)
            id_list = search_data.get("esearchresult", {}).get("idlist", [])
            if not id_list:
                return []

            summary_params = {
                "db": "pubmed",
                "id": ",".join(id_list),
                "retmode": "json",
            }
            summary_data = await self.get_json(self.ESUMMARY_API, params=summary_params)
            result_map = summary_data.get("result", {})
            hits = []

            for pmid in id_list:
                item = result_map.get(pmid)
                if not item:
                    continue

                title = item.get("title", "Untitled PubMed Article")
                authors = [a.get("name", "") for a in item.get("authors", []) if a.get("name")]
                pubdate = item.get("pubdate", "")
                source_journal = item.get("source", "")
                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

                hits.append(SourceHit(
                    id=f"pmid:{pmid}",
                    title=title,
                    url=url,
                    source_name="pubmed",
                    reliability_tier=self.reliability_tier,
                    snippet=f"Journal: {source_journal}. Published: {pubdate}.",
                    is_preprint=False,
                    authors=authors,
                    published_date=pubdate,
                    metadata={"pmid": pmid, "journal": source_journal},
                ))

            return hits
        except Exception:
            return []

    async def fetch(self, hit: SourceHit) -> Document:
        """Fetch summary representation of PubMed record."""
        return Document(
            id=hit.id,
            url=hit.url,
            title=hit.title,
            authors=hit.authors,
            published_date=hit.published_date,
            source="pubmed",
            reliability_tier=self.reliability_tier,
            is_preprint=False,
            license="NLM Open Access",
            text=f"{hit.title}\n{hit.snippet}",
            summary=hit.snippet,
            metadata=hit.metadata,
        )
