"""Private subprocess entry point. No agent actions or network access."""

import base64
import io
import json
import os
import socket
import sys
import xml.etree.ElementTree as ET
import zipfile

_WINDOWS_JOB = None


def restrict_windows_resources():
    import ctypes
    from ctypes import wintypes

    class BasicLimits(ctypes.Structure):
        _fields_ = [
            ("process_time", ctypes.c_longlong),
            ("job_time", ctypes.c_longlong),
            ("flags", wintypes.DWORD),
            ("min_ws", ctypes.c_size_t),
            ("max_ws", ctypes.c_size_t),
            ("processes", wintypes.DWORD),
            ("affinity", ctypes.c_size_t),
            ("priority", wintypes.DWORD),
            ("scheduling", wintypes.DWORD),
        ]

    class IoCounters(ctypes.Structure):
        _fields_ = [
            (name, ctypes.c_ulonglong)
            for name in ("reads", "writes", "other", "read_bytes", "write_bytes", "other_bytes")
        ]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [
            ("basic", BasicLimits),
            ("io", IoCounters),
            ("process_memory", ctypes.c_size_t),
            ("job_memory", ctypes.c_size_t),
            ("peak_process", ctypes.c_size_t),
            ("peak_job", ctypes.c_size_t),
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    job = kernel.CreateJobObjectW(None, None)
    if not job:
        raise OSError(ctypes.get_last_error(), "Cannot create parser resource job")
    limits = ExtendedLimits()
    limits.basic.flags = 0x100 | 0x8 | 0x2000 | 0x2
    limits.basic.process_time = 15 * 10_000_000
    limits.basic.processes = 1
    limits.process_memory = 768 * 1024**2
    if not kernel.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
        raise OSError(ctypes.get_last_error(), "Cannot limit parser memory")
    if not kernel.AssignProcessToJobObject(job, kernel.GetCurrentProcess()):
        raise OSError(ctypes.get_last_error(), "Cannot isolate parser resources")
    global _WINDOWS_JOB
    _WINDOWS_JOB = job


def restrict_resources():
    if os.name == "nt":
        restrict_windows_resources()
    else:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (768 * 1024**2, 768 * 1024**2))
        resource.setrlimit(resource.RLIMIT_CPU, (15, 15))
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))

    # Parsers never need network connections, including external DOCX references.
    def denied(*args, **kwargs):
        raise PermissionError("Parser network access is disabled")

    original_socket = socket.socket

    class RestrictedSocket(original_socket):
        connect = denied
        connect_ex = denied

    socket.socket = RestrictedSocket
    socket.create_connection = denied


def main():
    restrict_resources()
    payload = json.loads(sys.stdin.buffer.read(35_000_000))
    data = base64.b64decode(payload["data"], validate=True)
    if len(data) > 25_000_000:
        raise ValueError("Input too large")
    pages, title = [], ""
    if payload["kind"] == "pdf":
        import pymupdf

        with pymupdf.open(stream=data, filetype="pdf") as doc:
            for i in range(min(len(doc), 200)):
                pages.append({"page": i + 1, "text": doc[i].get_text()[:100_000]})
    elif payload["kind"] == "docx":
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries = archive.infolist()
            if len(entries) > 3000 or sum(e.file_size for e in entries) > 64_000_000:
                raise ValueError("DOCX expanded content exceeds limit")

            def xml(name):
                content = archive.read(name)
                if (
                    len(content) > 16_000_000
                    or b"<!DOCTYPE" in content.upper()
                    or b"<!ENTITY" in content.upper()
                ):
                    raise ValueError("DOCX XML exceeds limits or declares entities")
                return ET.fromstring(content)

            document = xml("word/document.xml")
            ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
            prefix = "{" + ns["w"] + "}"
            styles = {}
            if "word/styles.xml" in archive.namelist():
                for style in xml("word/styles.xml").findall("w:style", ns):
                    name = style.find("w:name", ns)
                    if name is not None:
                        styles[style.get(prefix + "styleId")] = name.get(prefix + "val", "")

            def blocks_in(parent):
                blocks = []
                for element in parent:
                    if element.tag == prefix + "p":
                        parts = []
                        for node in element.iter():
                            if node.tag == prefix + "t":
                                parts.append(node.text or "")
                            elif node.tag in (prefix + "br", prefix + "cr"):
                                parts.append("\n")
                            elif node.tag == prefix + "tab":
                                parts.append("\t")
                        text = "".join(parts)
                        style = element.find("w:pPr/w:pStyle", ns)
                        if style is not None:
                            identifier = style.get(prefix + "val", "")
                            heading = styles.get(identifier, identifier).lower().replace(" ", "")
                            if heading.startswith("heading") and heading[7:].isdigit():
                                text = "#" * max(1, min(6, int(heading[7:]))) + " " + text
                    elif element.tag == prefix + "tbl":
                        text = "\n".join(
                            " | ".join(
                                "\n".join(blocks_in(cell)) for cell in row.findall("w:tc", ns)
                            )
                            for row in element.findall("w:tr", ns)
                        )
                    else:
                        continue
                    if text.strip():
                        blocks.append(text)
                return blocks

            body = document.find("w:body", ns)
            if body is None:
                raise ValueError("DOCX contains no document body")
            blocks = blocks_in(body)
        pages = [{"page": None, "text": "\n\n".join(blocks)}]
    elif payload["kind"] == "html":
        import trafilatura
        from bs4 import BeautifulSoup

        html = data.decode("utf-8", errors="replace")
        soup = BeautifulSoup(html, "html.parser")
        title = soup.title.get_text(strip=True) if soup.title else ""
        text = trafilatura.extract(html, include_comments=False, include_tables=True)
        if not text:
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()
            text = soup.get_text("\n", strip=True)
        pages = [{"page": 1, "text": text[:2_000_000]}]
    else:
        raise ValueError("Unsupported parser")
    text = "\n\n".join(p["text"] for p in pages)
    if len(text) > 2_000_000:
        raise ValueError("Extracted text too large")
    print(json.dumps({"text": text, "title": title, "pages": pages}), flush=True)


if __name__ == "__main__":
    main()
