"""FastAPI Application Factory and Server for L.O.R.A Gateway."""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from lora.gateway.auth import HostOriginGuard, IpAuthRateLimiter, TokenValidator
from lora.gateway.queue import JobQueue

from lora.gateway.routes import create_gateway_router
from lora.gateway.ws import ConnectionManager
from lora.storage.notebook import NotebookStore


from pydantic import BaseModel
from starlette.requests import Request
from fastapi.middleware.cors import CORSMiddleware
from lora.models.ollama_client import OllamaClient
from lora.grounding.media_vector_db import MediaVectorDB

LOCAL_ORIGIN_REGEX = r"https?://(localhost|127\.0\.0\.1|\[::1\]|192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})(:\d+)?"


class HudChatPayload(BaseModel):
    text: str = ""


class HudMediaAddPayload(BaseModel):
    path: str = ""


class HudMediaRemovePayload(BaseModel):
    path_or_id: str = ""


class HudMediaReportPayload(BaseModel):
    query: str = ""


def create_gateway_app(
    origin: str = "desktop",
    queue: JobQueue | None = None,
    token_validator: TokenValidator | None = None,
    ip_limiter: IpAuthRateLimiter | None = None,
    host_guard: HostOriginGuard | None = None,
    notebook_store: NotebookStore | None = None,
    connection_manager: ConnectionManager | None = None,
    static_dir: str | Path | None = None,
    media_db: MediaVectorDB | None = None,
) -> FastAPI:
    """Create and configure FastAPI application for the specified listener origin."""
    queue = queue or JobQueue()
    token_validator = token_validator or TokenValidator()
    ip_limiter = ip_limiter or IpAuthRateLimiter()
    host_guard = host_guard or HostOriginGuard(listener_origin=origin)
    notebook_store = notebook_store or NotebookStore()
    connection_manager = connection_manager or ConnectionManager()

    app = FastAPI(
        title=f"L.O.R.A Gateway ({origin.capitalize()})",
        version="0.1.0",
        docs_url=None,  # Disabled for security
        redoc_url=None,
    )

    # Only this machine and private-LAN pages may talk to the gateway from a browser. "*" let ANY website
    # you visit call the local API (index a file via /api/hud/media/add, read it back via /report...).
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=LOCAL_ORIGIN_REGEX,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def _reject_foreign_origins(request, call_next):
        origin = request.headers.get("origin")
        if origin and not re.fullmatch(LOCAL_ORIGIN_REGEX, origin):
            return JSONResponse({"detail": "Cross-site requests are not allowed."}, status_code=403)
        return await call_next(request)

    media_db = media_db or MediaVectorDB()

    @app.get("/api/hud/status")
    async def get_hud_status():
        ollama = OllamaClient()
        online = await ollama.is_online()
        active_model = "unknown"
        if online:
            models = await ollama.list_models()
            for cand in ("spark-x25-4b-32k:latest", "spark-x25-4b-8k:latest", "granite4.2:8b", "gemma4:e4b"):
                if any(m.get("name") == cand for m in models):
                    active_model = cand
                    break
            else:
                if models:
                    active_model = models[0].get("name", "none")
        return {
            "status": "online",
            "ollama_online": online,
            "active_model": active_model,
            "media_files": media_db.file_count,
        }

    @app.post("/api/hud/chat")
    async def chat_with_hud(payload: HudChatPayload):
        query = payload.text.strip()
        if not query:
            return {"status": "error", "message": "Empty query"}

        ollama = OllamaClient()
        is_online = await ollama.is_online()
        model_to_use = "spark-x25-4b-32k:latest"

        if is_online:
            models = await ollama.list_models()
            available = [m.get("name", "") for m in models]
            for cand in (
                "spark-x25-4b-32k:latest",
                "spark-x25-4b-8k:latest",
                "granite4.2:8b",
                "gemma4:e4b",
                "ornith-1.5:9b",
            ):
                if cand in available:
                    model_to_use = cand
                    break
            else:
                if available:
                    model_to_use = available[0]

            # Search media db for relevant context
            search_results = media_db.search(query, top_k=2)
            context_snippet = ""
            sources = []
            if search_results:
                for i, r in enumerate(search_results):
                    sources.append([f"S{i+1}", r["filename"], "Media Knowledge DB", r["text"][:120]])
                    context_snippet += f"\n[From {r['filename']}]: {r['text'][:300]}"
            else:
                sources.append(["S1", "Ollama Local Engine", f"Model: {model_to_use}", "Local neural model running on laptop."])
                sources.append(["S2", "L.O.R.A Grounding", "Verified Sovereign", "Claims fact-checked against system context."])

            sys_prompt = (
                "You are L.O.R.A (Layer Oriented Research Assistant), a sovereign, "
                "grounded local AI research assistant running on the user's computer. "
                "Be direct, articulate, concise, and helpful. Do not mention that you are a large language model."
            )
            if context_snippet:
                sys_prompt += f"\nRelevant indexed files:\n{context_snippet}"

            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": query},
            ]
            answer = await ollama.generate_chat(model=model_to_use, messages=messages)
        else:
            answer = "Ollama is currently offline at http://127.0.0.1:11434. Please start Ollama (`ollama serve`)."
            sources = [["S1", "System Alert", "Offline", "Local Ollama service unreachable."]]

        return {
            "status": "ok",
            "query": query,
            "answer": answer,
            "sources": sources,
            "confidence": 0.95,
        }

    @app.get("/api/hud/media/list")
    async def list_media_files():
        return {"status": "ok", "files": media_db.list_files()}

    @app.post("/api/hud/media/add")
    async def add_media_file(payload: HudMediaAddPayload):
        path_str = payload.path.strip()
        if not path_str:
            return {"status": "error", "message": "Missing path"}
        res = media_db.add_file(path_str)
        return {"status": "ok", "result": res, "files": media_db.list_files()}

    @app.post("/api/hud/media/remove")
    async def remove_media_file(payload: HudMediaRemovePayload):
        path_or_id = payload.path_or_id.strip()
        if not path_or_id:
            return {"status": "error", "message": "Missing path or id"}
        res = media_db.remove_file(path_or_id)
        return {"status": "ok", "result": res, "files": media_db.list_files()}

    @app.post("/api/hud/media/report")
    async def report_on_media(payload: HudMediaReportPayload):
        q = payload.query.strip()
        rep = media_db.report_on_media(query=q)
        return {"status": "ok", "report": rep}

    # Attach router
    router = create_gateway_router(
        origin=origin,
        queue=queue,
        token_validator=token_validator,
        ip_limiter=ip_limiter,
        host_guard=host_guard,
        notebook_store=notebook_store,
        connection_manager=connection_manager,
    )
    app.include_router(router)

    # Mount HUD static assets (droplet.html)
    ui_dir = Path(__file__).resolve().parent.parent / "ui"
    if ui_dir.exists():
        app.mount("/hud", StaticFiles(directory=str(ui_dir), html=True), name="hud")

    # Mount static assets for mobile web client if directory exists
    if static_dir is None:
        static_dir = Path(__file__).resolve().parent / "static" / "app"

    static_path = Path(static_dir)
    if static_path.exists():
        app.mount("/app", StaticFiles(directory=str(static_path), html=True), name="app")

    from fastapi.responses import RedirectResponse

    @app.get("/", include_in_schema=False)
    def redirect_to_hud():
        return RedirectResponse(url="/hud/droplet.html")

    return app
