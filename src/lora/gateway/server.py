"""FastAPI Application Factory and Server for L.O.R.A Gateway."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from lora.gateway.auth import HostOriginGuard, IpAuthRateLimiter, TokenValidator
from lora.gateway.queue import JobQueue
from lora.gateway.routes import create_gateway_router
from lora.gateway.ws import ConnectionManager
from lora.storage.notebook import NotebookStore


def create_gateway_app(
    origin: str = "desktop",
    queue: JobQueue | None = None,
    token_validator: TokenValidator | None = None,
    ip_limiter: IpAuthRateLimiter | None = None,
    host_guard: HostOriginGuard | None = None,
    notebook_store: NotebookStore | None = None,
    connection_manager: ConnectionManager | None = None,
    static_dir: str | Path | None = None,
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

    # Mount static assets for mobile web client if directory exists
    if static_dir is None:
        static_dir = Path(__file__).resolve().parent / "static" / "app"

    static_path = Path(static_dir)
    if static_path.exists():
        app.mount("/app", StaticFiles(directory=str(static_path), html=True), name="app")

    return app
