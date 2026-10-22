"""HTML Cleaning and Trafilatura Article Extraction for L.O.R.A."""

from typing import Any

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

        from lora.safety.parsers import parse_document
        parsed = parse_document("html", html_content.encode("utf-8"))
        title, extracted_text = parsed["title"], parsed["text"]

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
