"""GitHub REST API Connector for L.O.R.A."""

import base64

from lora.tools.web.cache import HttpCache
from lora.tools.web.connectors.base import BaseConnector, Document, ReliabilityTier, SourceHit
from lora.tools.web.limiter import TokenBucketLimiter


class GitHubConnector(BaseConnector):
    """GitHub REST API connector for repository and documentation discovery."""

    SEARCH_API = "https://api.github.com/search/repositories"

    def __init__(
        self,
        token: str | None = None,
        limiter: TokenBucketLimiter | None = None,
        cache: HttpCache | None = None,
    ) -> None:
        super().__init__(
            name="github",
            reliability_tier=ReliabilityTier.T3,
            limiter=limiter,
            cache=cache,
        )
        self.token = token

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/vnd.github.v3+json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    async def search(self, query: str, limit: int = 10) -> list[SourceHit]:
        """Search GitHub repositories."""
        params = {
            "q": query.strip(),
            "per_page": min(limit, 15),
        }

        try:
            data = await self.get_json(self.SEARCH_API, params=params, headers=self._headers())
            items = data.get("items", [])
            hits = []

            for repo in items:
                full_name = repo.get("full_name", "")
                description = repo.get("description", "") or "No description provided."
                stars = repo.get("stargazers_count", 0)
                html_url = repo.get("html_url", "")
                updated_at = repo.get("updated_at", "")

                hits.append(SourceHit(
                    id=f"gh:{full_name}",
                    title=full_name,
                    url=html_url,
                    source_name="github",
                    reliability_tier=self.reliability_tier,
                    snippet=f"{description} (Stars: {stars})",
                    is_preprint=False,
                    authors=[repo.get("owner", {}).get("login", "")],
                    published_date=updated_at,
                    metadata={"full_name": full_name, "stars": stars, "description": description},
                ))

            return hits
        except Exception:
            return []

    async def fetch(self, hit: SourceHit) -> Document:
        """Fetch repository README content."""
        full_name = hit.metadata.get("full_name") or hit.title
        readme_url = f"https://api.github.com/repos/{full_name}/readme"

        try:
            readme_data = await self.get_json(readme_url, headers=self._headers())
            content_b64 = readme_data.get("content", "")
            if content_b64:
                readme_text = base64.b64decode(content_b64).decode("utf-8", errors="ignore")
            else:
                readme_text = hit.snippet

            return Document(
                id=hit.id,
                url=hit.url,
                title=hit.title,
                authors=hit.authors,
                published_date=hit.published_date,
                source="github",
                reliability_tier=self.reliability_tier,
                is_preprint=False,
                license="Open Source",
                text=readme_text,
                summary=hit.snippet,
                metadata=hit.metadata,
            )
        except Exception:
            return Document(
                id=hit.id,
                url=hit.url,
                title=hit.title,
                authors=hit.authors,
                source="github",
                reliability_tier=self.reliability_tier,
                is_preprint=False,
                text=hit.snippet,
                summary=hit.snippet,
            )
