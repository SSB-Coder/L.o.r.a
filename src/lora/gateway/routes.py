"""REST and WebSocket route definitions for L.O.R.A Gateway."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    Header,
    HTTPException,
    Request,
    WebSocket,
    status,
)

from lora.gateway.auth import HostOriginGuard, IpAuthRateLimiter, TokenValidator
from lora.gateway.models import (
    AskRequest,
    AskResponse,
    JobResponse,
    NotebookListResponse,
    NotebookSummary,
    RewriteRequest,
    StopRequest,
    StopResponse,
)
from lora.gateway.queue import JobItem, JobOrigin, JobQueue
from lora.gateway.ws import ConnectionManager, handle_websocket_connection
from lora.storage.notebook import NotebookStore


def create_gateway_router(
    origin: str,
    queue: JobQueue,
    token_validator: TokenValidator,
    ip_limiter: IpAuthRateLimiter,
    host_guard: HostOriginGuard,
    notebook_store: NotebookStore,
    connection_manager: ConnectionManager,
) -> APIRouter:
    """Create configured FastAPI router for the specified listener origin."""
    router = APIRouter()

    async def verify_auth_and_security(
        request: Request,
        authorization: Annotated[str | None, Header()] = None,
        x_lora_token: Annotated[str | None, Header(alias="X-Lora-Token")] = None,
    ) -> str:
        """Verify Host/Origin rebinding defense, IP rate limiting, and scoped Bearer token."""
        client_ip = request.client.host if request.client else "unknown"

        # 1. Defend against CSRF and DNS-rebinding
        host_header = request.headers.get("host")
        if not host_guard.validate_host(host_header):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Host header rejected by rebinding guard",
            )

        origin_header = request.headers.get("origin")
        if not host_guard.validate_origin(origin_header):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Origin header rejected by CSRF guard",
            )

        # 2. Check IP rate-limit ban
        banned, remaining = ip_limiter.is_banned(client_ip)
        if banned:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Too many failed attempts. Banned for {int(remaining)}s",
                headers={"Retry-After": str(int(remaining))},
            )

        # 3. Extract and validate token (support FastAPI Header DI, raw request.headers, and /app pairing cookie)
        auth_val = authorization or request.headers.get("authorization")
        xlora_val = x_lora_token or request.headers.get("x-lora-token")
        token: str | None = None
        if auth_val and auth_val.startswith("Bearer "):
            token = auth_val[7:].strip()
        elif xlora_val:
            token = xlora_val.strip()

        if not token or not token_validator.validate_token(token, origin):
            ip_limiter.record_failure(client_ip)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing authentication token for this listener",
            )

        ip_limiter.record_success(client_ip)
        return token

    # 1. Unauthenticated Healthz
    @router.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    # 2. Ask endpoint
    @router.post(
        "/v1/ask",
        response_model=AskResponse,
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(verify_auth_and_security)],
    )
    async def ask(payload: AskRequest):
        # Mobile clients are strictly restricted to quick or standard run modes
        if origin == "mobile" and payload.run_mode.lower() in ("deep", "deep_with_refinement"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Mobile origin is restricted to quick or standard research modes",
            )

        # Enforce notebook access gating for mobile clients
        if payload.notebook_id:
            nb = await notebook_store.get_notebook(payload.notebook_id)
            if nb is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Notebook '{payload.notebook_id}' not found",
                )
            if origin == "mobile" and getattr(nb, "visibility", "public") == "desktop_only":
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Notebook not accessible from mobile origin",
                )

        job_id = f"job-{uuid.uuid4().hex[:12]}"
        job = JobItem(
            job_id=job_id,
            origin=JobOrigin(origin),
            text=payload.text,
            notebook_id=payload.notebook_id,
            depth=payload.depth,
            run_mode=payload.run_mode,
        )
        pos = await queue.enqueue(job)
        return AskResponse(job_id=job_id, queue_position=pos, status="queued")

    # 3. Job status endpoint
    @router.get(
        "/v1/jobs/{job_id}",
        response_model=JobResponse,
        dependencies=[Depends(verify_auth_and_security)],
    )
    async def get_job_status(job_id: str):
        job = queue.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
        if origin == "mobile" and job.origin != JobOrigin.MOBILE:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

        pos = queue.get_queue_position(job_id)
        return JobResponse(
            job_id=job.job_id,
            status=job.status.value,
            queue_position=pos,
            origin=job.origin.value,
            result=job.result,
            error=job.error,
        )

    # 4. Stop endpoint (Scoped cancellation)
    @router.post(
        "/v1/stop",
        response_model=StopResponse,
        dependencies=[Depends(verify_auth_and_security)],
    )
    async def stop_job(payload: StopRequest):
        ok, msg = await queue.request_stop(payload.job_id, requester_origin=JobOrigin(origin))
        if not ok:
            if "mobile origin cannot cancel" in msg.lower():
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=msg)
            elif "not found" in msg.lower():
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=msg)
            else:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg)

        return StopResponse(stopped=True, job_id=payload.job_id, message=msg)

    # 5. Notebooks listing (with mobile visibility filtering)
    @router.get(
        "/v1/notebooks",
        response_model=NotebookListResponse,
        dependencies=[Depends(verify_auth_and_security)],
    )
    async def list_notebooks():
        all_nbs = await notebook_store.list_notebooks()
        summaries: list[NotebookSummary] = []
        for nb in all_nbs:
            nb_id = nb["id"] if isinstance(nb, dict) else nb.id
            title = nb["title"] if isinstance(nb, dict) else nb.title
            created_at = str(nb["created_at"]) if isinstance(nb, dict) else str(nb.created_at)
            visibility = nb.get("visibility", "public") if isinstance(nb, dict) else getattr(nb, "visibility", "public")
            depth_mode = nb.get("depth_mode", "standard") if isinstance(nb, dict) else getattr(nb, "depth_mode", "standard")
            source_count = 0
            if isinstance(nb, dict):
                source_count = nb.get("source_count", 0)
            elif hasattr(nb, "chunks"):
                source_count = len(nb.chunks)
            elif hasattr(nb, "sources"):
                source_count = len(nb.sources)

            # Hide desktop_only notebooks from mobile clients
            if origin == "mobile" and visibility == "desktop_only":
                continue

            summaries.append(
                NotebookSummary(
                    id=nb_id,
                    title=title,
                    depth_mode=depth_mode,
                    source_count=source_count,
                    visibility=visibility,
                    created_at=created_at,
                )
            )
        return NotebookListResponse(notebooks=summaries)

    # 6. Notebook rewrite endpoint
    @router.post(
        "/v1/notebooks/{notebook_id}/rewrite",
        response_model=AskResponse,
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(verify_auth_and_security)],
    )
    async def rewrite_notebook(notebook_id: str, payload: RewriteRequest):
        nb = await notebook_store.get_notebook(notebook_id)
        if nb is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Notebook not found")

        if origin == "mobile" and getattr(nb, "visibility", "public") == "desktop_only":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Notebook not accessible from mobile origin",
            )

        job_id = f"job-{uuid.uuid4().hex[:12]}"
        job = JobItem(
            job_id=job_id,
            origin=JobOrigin(origin),
            text=f"Rewrite notebook '{nb.title}' at depth {payload.depth}",
            notebook_id=notebook_id,
            depth=payload.depth,
            run_mode="standard",
        )
        pos = await queue.enqueue(job)
        return AskResponse(job_id=job_id, queue_position=pos, status="queued")

    # 7. WebSocket events endpoint
    @router.websocket("/v1/events")
    async def websocket_events(websocket: WebSocket):
        await handle_websocket_connection(
            websocket=websocket,
            token_validator=token_validator,
            listener_origin=origin,
            ip_limiter=ip_limiter,
            manager=connection_manager,
            host_guard=host_guard,
            handshake_timeout=3.0,
        )

    return router
