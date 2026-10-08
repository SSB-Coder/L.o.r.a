"""WebSocket connection manager and event streaming for L.O.R.A Gateway."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect, status

from lora.gateway.auth import HostOriginGuard, IpAuthRateLimiter, TokenValidator

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages active WebSocket connections and handles event broadcasting."""

    def __init__(self) -> None:
        self.active_connections: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        """Register an authenticated connection."""
        async with self._lock:
            self.active_connections.add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        """Remove a closed or disconnected connection."""
        async with self._lock:
            self.active_connections.discard(websocket)

    async def broadcast(self, event: dict[str, Any]) -> None:
        """Broadcast an event payload to all active connections."""
        async with self._lock:
            dead_connections = set()
            for connection in self.active_connections:
                try:
                    await connection.send_json(event)
                except Exception:
                    dead_connections.add(connection)
            for dead in dead_connections:
                self.active_connections.discard(dead)


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

    # 3-Second Authentication Handshake
    try:
        raw_msg = await asyncio.wait_for(
            websocket.receive_text(), timeout=handshake_timeout
        )
        data = json.loads(raw_msg)
        token = data.get("token")
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
    except (TimeoutError, json.JSONDecodeError, Exception) as exc:
        ip_limiter.record_failure(client_ip)
        try:
            await websocket.close(
                code=status.WS_1008_POLICY_VIOLATION,
                reason=f"Auth handshake failed or timed out: {exc}",
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
