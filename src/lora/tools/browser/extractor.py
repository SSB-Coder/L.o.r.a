"""HTML Cleaning and Trafilatura Article Extraction for L.O.R.A."""

import re
from typing import Any

import trafilatura
from bs4 import BeautifulSoup

from lora.tools.web.spotlight import SpotlightedContent, Spotlighter


class ContentExtractor:
    """Extracts and normalizes clean readable text from HTML documents."""

    def __init__(self, spotlighter: Spotlighter | None = None) -> None:
        self.spotlighter = spotlighter or Spotlighter()

    def extract_from_html(
        self,
        html_content: str,
        source_url: str = "",
        frame_safety: bool = True,
    ) -> dict[str, Any]:
        """Extract clean text content, title, and metadata from raw HTML.

        Uses Trafilatura for primary main-text extraction, falling back to BeautifulSoup.
        """
        if not html_content or not html_content.strip():
            return {
                "title": "",
                "text": "",
                "framed_content": None,
                "is_suspicious": False,
                "flagged_patterns": [],
            }

        # 1. Parse Title and basic metadata with BeautifulSoup
        soup = BeautifulSoup(html_content, "html.parser")

        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        elif soup.find("h1"):
            title = soup.find("h1").get_text(strip=True)

        # 2. Extract clean main text via Trafilatura
        extracted_text = trafilatura.extract(
            html_content,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
        )

        # 3. Fallback to BeautifulSoup if Trafilatura returns empty
        if not extracted_text:
            # Remove scripts, styles, navigations, footers
            for elem in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
                elem.decompose()
            extracted_text = soup.get_text(separator="\n", strip=True)
            # Collapse excess blank lines
            extracted_text = re.sub(r"\n{3,}", "\n\n", extracted_text)

        # 4. Spotlighting & Injection Framing
        spotlight_res: SpotlightedContent | None = None
        if frame_safety and extracted_text:
            spotlight_res = self.spotlighter.frame_untrusted_data(
                text=extracted_text,
                source=source_url or "web",
                datamark=False,
            )

        return {
            "title": title,
            "text": extracted_text or "",
            "framed_content": spotlight_res.framed_text if spotlight_res else (extracted_text or ""),
            "is_suspicious": spotlight_res.is_suspicious if spotlight_res else False,
            "flagged_patterns": spotlight_res.flagged_patterns if spotlight_res else [],
        }
