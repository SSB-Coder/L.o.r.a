"""create_note (.md/.txt) plus Artifact Studio (create_artifact, generate_visual) and Multimodal Attachment ingestion."""

from __future__ import annotations

import base64
import html
import io
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

ALLOWED_EXTENSIONS = {"md", "txt"}
SAFE_ARTIFACT_EXTENSIONS = {"md", "txt", "py", "html", "json", "csv", "svg", "yaml", "yml"}
MAX_CHARS = 200_000
MAX_ARTIFACT_CHARS = 300_000


def default_notes_dir() -> Path:
    env = os.environ.get("LORA_NOTES_DIR")
    return Path(env).expanduser() if env else Path.home() / "Documents" / "LORA"


def _safe_stem(title: str) -> str:
    stem = re.sub(r"[^\w\- ]+", "", title, flags=re.UNICODE).strip().replace("  ", " ")
    stem = re.sub(r"\s+", " ", stem)[:60].strip(" .-_")
    return stem or "Note " + datetime.now().strftime("%Y-%m-%d %H%M")


def create_note(
    title: str,
    content: str,
    ext: str = "md",
    folder: str | Path | None = None,
    policy: Any = None,
) -> dict[str, Any]:
    """Create a new note file (.md or .txt). ``policy`` (a PolicyEngine) is consulted when given."""
    ext = (ext or "md").lower().lstrip(".")
    if ext not in ALLOWED_EXTENSIONS:
        return {"status": "error", "message": f"I can only create .md or .txt files, not .{ext}."}
    body = content or ""
    if not body.strip():
        return {"status": "error", "message": "There is nothing to save yet."}
    if len(body) > MAX_CHARS:
        return {"status": "error", "message": "That note is too large to save."}

    directory = Path(folder).expanduser() if folder else default_notes_dir()
    stem = _safe_stem(title)
    target = directory / f"{stem}.{ext}"
    n = 2
    while target.exists():  # never overwrite
        target = directory / f"{stem} ({n}).{ext}"
        n += 1

    if policy is not None:
        verdict = policy.check_path(str(target), origin="desktop", mode="write")
        if not verdict.allowed:
            return {"status": "denied", "message": f"I'm not allowed to write there: {verdict.reason}"}

    try:
        directory.mkdir(parents=True, exist_ok=True)
        with open(target, "x", encoding="utf-8") as fh:
            fh.write(body if body.endswith("\n") else body + "\n")
    except Exception as exc:
        return {"status": "error", "message": f"Couldn't save the file: {exc}"}
    return {"status": "success", "path": str(target), "message": f"Saved {target.name}"}


def create_artifact(
    title: str,
    content: str,
    ext: str = "md",
    folder: str | Path | None = None,
    policy: Any = None,
) -> dict[str, Any]:
    """Create a new code, document, data, or SVG file in the user's LORA folder without overwriting existing files."""
    clean_ext = (ext or "md").lower().lstrip(".")
    if clean_ext not in SAFE_ARTIFACT_EXTENSIONS:
        return {
            "status": "error",
            "message": f"Unsupported artifact extension '.{clean_ext}'. Allowed: {', '.join(sorted(SAFE_ARTIFACT_EXTENSIONS))}.",
        }
    body = content or ""
    if not body.strip():
        return {"status": "error", "message": "Artifact content cannot be empty."}
    if len(body) > MAX_ARTIFACT_CHARS:
        return {"status": "error", "message": "Artifact content exceeds maximum size limit."}

    directory = Path(folder).expanduser() if folder else default_notes_dir()
    stem = _safe_stem(title)
    target = directory / f"{stem}.{clean_ext}"
    n = 2
    while target.exists():
        target = directory / f"{stem} ({n}).{clean_ext}"
        n += 1

    if policy is not None:
        verdict = policy.check_path(str(target), origin="desktop", mode="write")
        if not verdict.allowed:
            return {"status": "denied", "message": f"I'm not allowed to write there: {verdict.reason}"}

    try:
        directory.mkdir(parents=True, exist_ok=True)
        with open(target, "x", encoding="utf-8") as fh:
            fh.write(body if body.endswith("\n") else body + "\n")
    except Exception as exc:
        return {"status": "error", "message": f"Couldn't save the artifact: {exc}"}

    artifact_meta: dict[str, Any] = {
        "kind": "image" if clean_ext == "svg" else "file",
        "name": target.name,
        "path": str(target),
        "ext": clean_ext,
        "preview": body[:240],
    }
    if clean_ext == "svg" and "<svg" in body:
        b64 = base64.b64encode(body.encode("utf-8")).decode("ascii")
        artifact_meta["data_url"] = f"data:image/svg+xml;base64,{b64}"

    return {
        "status": "success",
        "path": str(target),
        "name": target.name,
        "message": f"Saved {target.name} to {target}",
        "artifact": artifact_meta,
    }


def _build_default_svg_diagram(title: str, spec: str) -> str:
    """Render a clean, high-contrast vector diagram/card when the model provides a textual specification."""
    lines = [ln.strip(" -•*\t") for ln in (spec or title).splitlines() if ln.strip(" -•*\t")]
    if not lines:
        lines = [title or "L.O.R.A Visual Artifact"]
    steps = lines[:6]
    height = max(280, 110 + len(steps) * 58)
    boxes = []
    for idx, step_text in enumerate(steps):
        y = 84 + idx * 54
        clipped = html.escape(step_text[:68])
        boxes.append(
            f'<rect x="44" y="{y}" width="552" height="40" rx="10" fill="#132634" stroke="#6cc4e0" stroke-opacity="0.55" stroke-width="1.5"/>'
            f'<circle cx="68" cy="{y + 20}" r="11" fill="#6cc4e0" fill-opacity="0.2"/>'
            f'<text x="68" y="{y + 24}" text-anchor="middle" fill="#6cc4e0" font-family="Segoe UI, sans-serif" font-size="12" font-weight="600">{idx + 1}</text>'
            f'<text x="92" y="{y + 25}" fill="#e6eef3" font-family="Segoe UI, sans-serif" font-size="14">{clipped}</text>'
        )
    safe_title = html.escape((title or "Visual Diagram")[:56])
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 {height}" width="640" height="{height}">'
        f'<defs><linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">'
        f'<stop offset="0%" stop-color="#0a1520"/><stop offset="100%" stop-color="#102433"/></linearGradient></defs>'
        f'<rect width="640" height="{height}" rx="16" fill="url(#bg)" stroke="#6cc4e0" stroke-opacity="0.35"/>'
        f'<text x="44" y="50" fill="#6cc4e0" font-family="Segoe UI, sans-serif" font-size="18" font-weight="600">{safe_title}</text>'
        + "".join(boxes)
        + "</svg>"
    )


def generate_visual(
    title: str,
    spec: str,
    svg_code: str = "",
    source_image_path: str = "",
    folder: str | Path | None = None,
    policy: Any = None,
) -> dict[str, Any]:
    """Generate a visual diagram (SVG/PNG) or annotate/modify an existing image and return an inline-renderable artifact."""
    directory = Path(folder).expanduser() if folder else default_notes_dir()
    stem = _safe_stem(title or "Visual")

    if source_image_path and Path(source_image_path).is_file():
        try:
            from PIL import Image, ImageDraw, ImageOps

            src_p = Path(source_image_path)
            with Image.open(src_p) as im:
                im_rgb = im.convert("RGB")
                low_spec = (spec or "").lower()
                if "grayscale" in low_spec or "black and white" in low_spec:
                    im_rgb = ImageOps.grayscale(im_rgb).convert("RGB")
                elif "invert" in low_spec:
                    im_rgb = ImageOps.invert(im_rgb)
                draw = ImageDraw.Draw(im_rgb)
                caption = (title or spec)[:60].strip()
                if caption:
                    w, h = im_rgb.size
                    draw.rectangle([(0, max(0, h - 36)), (w, h)], fill=(10, 20, 30))
                    draw.text((12, max(4, h - 26)), caption, fill=(108, 196, 224))
                out_target = directory / f"{stem}.png"
                n = 2
                while out_target.exists():
                    out_target = directory / f"{stem} ({n}).png"
                    n += 1
                if policy is not None:
                    verdict = policy.check_path(str(out_target), origin="desktop", mode="write")
                    if not verdict.allowed:
                        return {"status": "denied", "message": f"I'm not allowed to write there: {verdict.reason}"}
                directory.mkdir(parents=True, exist_ok=True)
                buf = io.BytesIO()
                im_rgb.save(buf, format="PNG")
                png_bytes = buf.getvalue()
                out_target.write_bytes(png_bytes)
                b64 = base64.b64encode(png_bytes).decode("ascii")
                return {
                    "status": "success",
                    "path": str(out_target),
                    "name": out_target.name,
                    "message": f"Created edited image {out_target.name}",
                    "artifact": {
                        "kind": "image",
                        "name": out_target.name,
                        "path": str(out_target),
                        "ext": "png",
                        "data_url": f"data:image/png;base64,{b64}",
                    },
                }
        except Exception:
            pass

    raw_svg = (svg_code or "").strip()
    if "<svg" not in raw_svg:
        m = re.search(r"<svg[\s\S]*?</svg>", spec or "", re.IGNORECASE)
        raw_svg = m.group(0) if m else _build_default_svg_diagram(title, spec)

    return create_artifact(title or "Visual Diagram", raw_svg, ext="svg", folder=directory, policy=policy)


def process_attachments(attachments: list[dict[str, Any]] | None) -> tuple[list[str], str, list[str]]:
    """Extract base64 images for vision models, readable text from attached files, and local image paths."""
    if not attachments:
        return [], "", []

    images_b64: list[str] = []
    context_parts: list[str] = []
    image_paths: list[str] = []

    for att in attachments[:6]:
        if not isinstance(att, dict):
            continue
        name = str(att.get("name") or "attachment").strip()
        mime = str(att.get("mime") or "").lower()
        data_url = str(att.get("data_url") or "")
        path_str = str(att.get("path") or "").strip()
        text_content = str(att.get("text") or "")

        if mime.startswith("image/") or data_url.startswith("data:image/") or name.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")):
            b64_payload = ""
            if ";base64," in data_url:
                b64_payload = data_url.split(";base64,", 1)[1].strip()
            elif path_str and Path(path_str).is_file():
                try:
                    b64_payload = base64.b64encode(Path(path_str).read_bytes()).decode("ascii")
                    image_paths.append(path_str)
                except Exception:
                    pass
            if b64_payload:
                images_b64.append(b64_payload)
                img_desc = f"[Attached Image: {name}]"
                try:
                    from PIL import Image

                    raw_bytes = base64.b64decode(b64_payload)
                    with Image.open(io.BytesIO(raw_bytes)) as im:
                        img_desc = f"[Attached Image: {name} ({im.format or 'IMG'} {im.width}x{im.height}px, mode={im.mode})]"
                        if not path_str:
                            tmp_dir = default_notes_dir() / "uploads"
                            tmp_dir.mkdir(parents=True, exist_ok=True)
                            saved_p = tmp_dir / _safe_stem(Path(name).stem or "pasted_image")
                            saved_p = saved_p.with_suffix(".png")
                            im.save(saved_p, format="PNG")
                            image_paths.append(str(saved_p))
                except Exception:
                    pass
                context_parts.append(img_desc)
            continue

        if text_content:
            context_parts.append(f"--- File: {name} ---\n{text_content[:6000]}")
        elif ";base64," in data_url:
            try:
                raw_bytes = base64.b64decode(data_url.split(";base64,", 1)[1])
                decoded = raw_bytes.decode("utf-8", errors="replace")
                context_parts.append(f"--- File: {name} ---\n{decoded[:6000]}")
            except Exception:
                context_parts.append(f"[Attached Binary File: {name}]")
        elif path_str and Path(path_str).is_file():
            try:
                from lora.tools.files.read import read_file

                res = read_file(path_str)
                if res.get("status") == "success":
                    body = str(res.get("content") or res.get("text") or "")[:6000]
                    context_parts.append(f"--- File: {name} ({path_str}) ---\n{body}")
            except Exception:
                pass

    return images_b64, "\n\n".join(context_parts), image_paths
