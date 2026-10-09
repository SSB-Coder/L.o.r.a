import datetime
import html
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


class ReportExporter:
    """Exports research notebooks and grounded answers to markdown or HTML documents."""

    def __init__(self, outputs_dir: Path) -> None:
        self._outputs_dir = outputs_dir
        self._outputs_dir.mkdir(parents=True, exist_ok=True)

    @property
    def outputs_dir(self) -> Path:
        return self._outputs_dir

    def export_markdown(
        self,
        title: str,
        content: str,
        filename: str | None = None,
    ) -> Path:
        """Exports formatted research text to a .md file in the outputs directory."""
        safe_fname = self._make_safe_filename(title, filename, ext=".md")
        target_path = self._outputs_dir / safe_fname

        header = f"# {title}\n*Exported by L.O.R.A on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
        full_content = header + content.strip() + "\n"

        target_path.write_text(full_content, encoding="utf-8")
        logger.info("Exported markdown report to %s", target_path)
        return target_path

    def export_html(
        self,
        title: str,
        markdown_content: str,
        filename: str | None = None,
    ) -> Path:
        """Exports research report wrapped in an elegant dark-aqueous HTML template."""
        safe_fname = self._make_safe_filename(title, filename, ext=".html")
        target_path = self._outputs_dir / safe_fname

        # Simple HTML conversion preserving line breaks and basic formatting
        escaped_content = html.escape(markdown_content)
        # Convert markdown headers
        escaped_content = re.sub(r"^### (.*)$", r"<h3>\1</h3>", escaped_content, flags=re.MULTILINE)
        escaped_content = re.sub(r"^## (.*)$", r"<h2>\1</h2>", escaped_content, flags=re.MULTILINE)
        escaped_content = re.sub(r"^# (.*)$", r"<h1>\1</h1>", escaped_content, flags=re.MULTILINE)
        # Convert newlines to paragraphs/breaks
        body_html = escaped_content.replace("\n\n", "</p><p>").replace("\n", "<br/>")

        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(title)} - L.O.R.A Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #0b0f17;
            color: #d1d5db;
            line-height: 1.6;
            margin: 0;
            padding: 2rem;
        }}
        .container {{
            max-width: 800px;
            margin: 0 auto;
            background: #111827;
            padding: 2.5rem;
            border-radius: 12px;
            border: 1px solid #1f2937;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
        }}
        h1, h2, h3 {{
            color: #38bdf8;
            border-bottom: 1px solid #1e293b;
            padding-bottom: 0.3rem;
        }}
        .meta {{
            color: #6b7280;
            font-size: 0.85rem;
            margin-bottom: 2rem;
        }}
        code {{
            background: #1e293b;
            color: #f472b6;
            padding: 0.2rem 0.4rem;
            border-radius: 4px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{html.escape(title)}</h1>
        <div class="meta">Exported by L.O.R.A on {now_str}</div>
        <div class="content">
            <p>{body_html}</p>
        </div>
    </div>
</body>
</html>"""

        target_path.write_text(html_doc, encoding="utf-8")
        logger.info("Exported HTML report to %s", target_path)
        return target_path

    def _make_safe_filename(self, title: str, filename: str | None, ext: str) -> str:
        if filename:
            raw = filename
        else:
            raw = title.lower()

        clean = re.sub(r"[^\w\-_]", "_", raw).strip("_")
        if not clean:
            clean = "lora_report"
        if not clean.endswith(ext):
            clean += ext
        return clean
