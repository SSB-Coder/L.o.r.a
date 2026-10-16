"""Quarantined PDF Fetcher and Text Extraction for L.O.R.A."""

import tempfile
from pathlib import Path
from typing import Any

try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz
    except ImportError:
        fitz = None  # type: ignore

import httpx

from lora.safety.egress import EgressFilter
from lora.tools.web.spotlight import Spotlighter


class PdfFetcher:
    """Safely downloads and extracts text from external PDFs in a quarantined scratch location."""

    MAX_BYTES = 25_000_000  # 25 MB cap

    def __init__(
        self,
        egress_filter: EgressFilter | None = None,
        spotlighter: Spotlighter | None = None,
    ) -> None:
        self.egress_filter = egress_filter or EgressFilter()
        self.spotlighter = spotlighter or Spotlighter()

    async def fetch_pdf(self, url: str) -> dict[str, Any]:
        """Fetch remote PDF, parse text with PyMuPDF (or pypdf fallback), and remove scratch file."""
        egress_res = self.egress_filter.check_egress(url)
        if not egress_res.allowed:
            return {
                "status": "error",
                "message": f"PDF download blocked by policy: {egress_res.reason}",
                "content": "",
            }

        target_url = egress_res.cleaned_url

        temp_file = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
                temp_file = Path(tf.name)

            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                async with client.stream("GET", target_url) as resp:
                    resp.raise_for_status()
                    total_bytes = 0
                    with open(temp_file, "wb") as f:
                        async for chunk in resp.aiter_bytes():
                            total_bytes += len(chunk)
                            if total_bytes > self.MAX_BYTES:
                                return {
                                    "status": "error",
                                    "message": f"PDF file size exceeds {self.MAX_BYTES} bytes limit.",
                                    "content": "",
                                }
                            f.write(chunk)

            pages = []
            full_text_parts = []
            if fitz is not None:
                with fitz.open(str(temp_file)) as doc:
                    total_pages = len(doc)
                    for page_idx in range(min(total_pages, 50)):
                        text = doc[page_idx].get_text()
                        pages.append({"page": page_idx + 1, "text": text})
                        full_text_parts.append(text)
            else:
                from pypdf import PdfReader

                reader = PdfReader(str(temp_file))
                total_pages = len(reader.pages)
                for page_idx in range(min(total_pages, 50)):
                    text = reader.pages[page_idx].extract_text() or ""
                    pages.append({"page": page_idx + 1, "text": text})
                    full_text_parts.append(text)

            raw_content = "\n\n".join(full_text_parts)
            spotlight_res = self.spotlighter.frame_untrusted_data(
                text=raw_content,
                source=target_url,
            )

            return {
                "status": "success",
                "url": target_url,
                "total_pages": total_pages,
                "pages_read": len(pages),
                "pages": pages,
                "content": raw_content,
                "framed_content": spotlight_res.framed_text,
                "is_suspicious": spotlight_res.is_suspicious,
                "flagged_patterns": spotlight_res.flagged_patterns,
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to fetch or parse PDF from '{url}': {e}",
                "content": "",
            }
        finally:
            if temp_file and temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception:
                    pass
