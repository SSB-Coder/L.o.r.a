"""WebSocket connection manager and event streaming for L.O.R.A Gateway."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect, status

from lora.gateway.auth import HostOriginGuard, IpAuthRateLimiter, TokenValidator

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages active WebSocket connections and handles event broadcasting."""

    def __init__(self) -> None:
        self.active_connections: set[WebSocket] = set()
        self._lock = threading.Lock()
        self._loops: dict[WebSocket, asyncio.AbstractEventLoop] = {}

    async def connect(self, websocket: WebSocket) -> None:
        """Register an authenticated connection."""
        with self._lock:
            self.active_connections.add(websocket)
            self._loops[websocket] = asyncio.get_running_loop()

    async def disconnect(self, websocket: WebSocket) -> None:
        """Remove a closed or disconnected connection."""
        with self._lock:
            self.active_connections.discard(websocket)
            self._loops.pop(websocket, None)

    async def broadcast(self, event: dict[str, Any]) -> None:
        """Broadcast an event payload to all active connections."""
        current = asyncio.get_running_loop()
        with self._lock:
            connections = [(connection, self._loops.get(connection, current))
                           for connection in self.active_connections]

        async def send(connection, owner):
            try:
                if owner is current:
                    await asyncio.wait_for(connection.send_json(event), timeout=2)
                else:
                    # ASGI sockets belong to the server loop, not the job loop.
                    if not owner.is_running():
                        await self.disconnect(connection)
                        return
                    async def on_owner():
                        await asyncio.wait_for(connection.send_json(event), timeout=2)
                    coroutine = on_owner()
                    try:
                        future = asyncio.run_coroutine_threadsafe(coroutine, owner)
                    except RuntimeError:
                        coroutine.close()
                        raise
                    await asyncio.wait_for(asyncio.wrap_future(future), timeout=2.5)
            except Exception:
                await self.disconnect(connection)

        await asyncio.gather(*(send(connection, owner) for connection, owner in connections))


async def handle_websocket_connection(
    websocket: WebSocket,
    token_validator: TokenValidator,
    listener_origin: str,
    ip_limiter: IpAuthRateLimiter,
    manager: ConnectionManager,
    host_guard: HostOriginGuard | None = None,
    handshake_timeout: float = 3.0,
) -> None:
    """Handles WebSocket connection lifecycle with mandatory 3-second authentication handshake."""
    # Defend against CSRF and DNS-rebinding before accepting socket
    if host_guard is not None:
        host_header = websocket.headers.get("host")
        if not host_guard.validate_host(host_header):
            await websocket.close(
                code=status.WS_1008_POLICY_VIOLATION,
                reason="Host header rejected by rebinding guard",
            )
            return

        origin_header = websocket.headers.get("origin")
        if not host_guard.validate_origin(origin_header):
            await websocket.close(
                code=status.WS_1008_POLICY_VIOLATION,
                reason="Origin header rejected by CSRF guard",
            )
            return

    await websocket.accept()
    client_ip = websocket.client.host if websocket.client else "unknown"

    # Check if IP is already banned
    banned, _ = ip_limiter.is_banned(client_ip)
    if banned:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="IP is temporarily banned")
        return

    # First-frame authentication; URL/cookie credentials are deliberately refused.
    try:
        raw_msg = await asyncio.wait_for(
            websocket.receive_text(), timeout=handshake_timeout
        )
        data = json.loads(raw_msg)
        token = str(data.get("token") or "").strip()
        msg_type = data.get("type")

        if msg_type != "auth" or not token or not token_validator.validate_token(token, listener_origin):
            ip_limiter.record_failure(client_ip)
            await websocket.close(
                code=status.WS_1008_POLICY_VIOLATION,
                reason="Invalid authentication token or format",
            )
            return

        ip_limiter.record_success(client_ip)
        await websocket.send_json({"type": "auth_ok", "origin": listener_origin})
    except (TimeoutError, json.JSONDecodeError, Exception):
        ip_limiter.record_failure(client_ip)
        try:
            await websocket.close(
                code=status.WS_1008_POLICY_VIOLATION,
                reason="Auth handshake failed or timed out",
            )
        except Exception:
            pass
        return

    # Authenticated: Register connection and listen for client messages or disconnects
    await manager.connect(websocket)
    try:
        while True:
            text = await websocket.receive_text()
            # Respond to client ping/keepalive if needed
            try:
                msg = json.loads(text)
                if msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
            except Exception:
                pass
    except WebSocketDisconnect:
        pass
    finally:
        await manager.disconnect(websocket)
