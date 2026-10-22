"""Bounded parser subprocess interface; only text/metadata crosses the boundary."""

import base64
import json
import os
import subprocess
import sys
from pathlib import Path

MAX_INPUT_BYTES = 25_000_000


def parse_document(kind: str, data: bytes, timeout: float = 20) -> dict:
    if kind not in {"html", "pdf", "docx"}:
        raise ValueError("Unsupported parser type")
    if len(data) > MAX_INPUT_BYTES:
        raise ValueError("Parser input exceeds limit")
    worker = Path(__file__).with_name("parser_worker.py")
    env = {k: v for k, v in os.environ.items() if k in {"SYSTEMROOT", "WINDIR", "PATH", "TEMP", "TMP"}}
    proc = subprocess.run(
        [sys.executable, "-I", str(worker)],
        input=json.dumps({"kind": kind, "data": base64.b64encode(data).decode("ascii")}),
        capture_output=True, text=True, encoding="utf-8", timeout=timeout, env=env,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), check=False,
    )
    if proc.returncode or len(proc.stdout) > 8_000_000:
        raise ValueError("Isolated parser failed or exceeded output limit")
    result = json.loads(proc.stdout)
    if not isinstance(result.get("text"), str) or not isinstance(result.get("pages"), list):
        raise ValueError("Invalid parser response")
    return result
