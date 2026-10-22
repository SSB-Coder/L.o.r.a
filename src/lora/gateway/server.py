"""Gateway composition. All inference enters the shared, origin-tagged queue."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.requests import Request

from lora.gateway.auth import HostOriginGuard, IpAuthRateLimiter, TokenValidator
from lora.gateway.queue import JobItem, JobOrigin, JobQueue
from lora.gateway.routes import create_gateway_router
from lora.gateway.ws import ConnectionManager
from lora.storage.notebook import NotebookStore


class HudChatPayload(BaseModel):
    text: str = Field(min_length=1, max_length=16000)


def create_gateway_app(
    origin: str = "desktop",
    queue: JobQueue | None = None,
    token_validator: TokenValidator | None = None,
    ip_limiter: IpAuthRateLimiter | None = None,
    host_guard: HostOriginGuard | None = None,
    notebook_store: NotebookStore | None = None,
    connection_manager: ConnectionManager | None = None,
    static_dir: str | Path | None = None,
    media_db=None,
) -> FastAPI:
    if origin not in {"desktop", "mobile"}:
        raise ValueError("Unknown gateway listener origin")
    queue = queue or JobQueue()
    token_validator = token_validator or TokenValidator()
    ip_limiter = ip_limiter or IpAuthRateLimiter()
    host_guard = host_guard or HostOriginGuard(listener_origin=origin)
    app = FastAPI(title="L.O.R.A Gateway", docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def secure_listener(request: Request, call_next):
        if not host_guard.validate_host(request.headers.get("host")):
            return JSONResponse({"detail": "Host rejected"}, status_code=403)
        if not host_guard.validate_origin(request.headers.get("origin")):
            return JSONResponse({"detail": "Origin rejected"}, status_code=403)
        # Public companion shell has no data. Fragment tokens never reach HTTP logs.
        public = request.url.path == "/healthz" or request.url.path in {
            "/app", "/app/", "/app/index.html", "/app/app.js", "/app/orb.js", "/app/style.css",
        }
        if origin == "mobile" and request.url.path.startswith("/api/hud/media"):
            return JSONResponse({"detail": "Local media is desktop only"}, status_code=403)
        if not public:
            ip = request.client.host if request.client else "unknown"
            banned, remaining = ip_limiter.is_banned(ip)
            if banned:
                return JSONResponse({"detail": "Too many authentication failures"}, status_code=429,
                                    headers={"Retry-After": str(int(remaining) + 1)})
            auth = request.headers.get("authorization", "")
            token = auth[7:].strip() if auth.startswith("Bearer ") else ""
            if not token_validator.validate_token(token, origin):
                ip_limiter.record_failure(ip)
                return JSONResponse({"detail": "Unauthorized"}, status_code=401)
            ip_limiter.record_success(ip)
        response = await call_next(request)
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/api/hud/status")
    async def hud_status():
        return {"status": "ok"}

    @app.post("/api/hud/chat", status_code=202)
    async def hud_chat(payload: HudChatPayload):
        import uuid

        job = JobItem(job_id=f"job-{uuid.uuid4().hex[:12]}", origin=JobOrigin(origin), text=payload.text)
        pos = await queue.enqueue(job)
        return {"job_id": job.job_id, "queue_position": pos, "status": "queued"}

    # Retire out-of-plan HTTP media operations; native HUD owns explicit selection.
    @app.api_route("/api/hud/media/{operation}", methods=["GET", "POST"])
    async def retired_media(operation: str):
        return JSONResponse({"detail": "Use native desktop media controls"}, status_code=410)

    app.include_router(create_gateway_router(
        origin=origin, queue=queue, token_validator=token_validator,
        ip_limiter=ip_limiter, host_guard=host_guard,
        notebook_store=notebook_store or NotebookStore(),
        connection_manager=connection_manager or ConnectionManager(),
    ))
    ui_dir = Path(__file__).resolve().parent.parent / "ui"
    app.mount("/hud", StaticFiles(directory=str(ui_dir), html=True), name="hud")
    static_path = Path(static_dir) if static_dir else Path(__file__).parent / "static" / "app"
    if static_path.exists():
        app.mount("/app", StaticFiles(directory=str(static_path), html=True), name="app")

    @app.get("/")
    async def index():
        return RedirectResponse("/app/")

    return app
