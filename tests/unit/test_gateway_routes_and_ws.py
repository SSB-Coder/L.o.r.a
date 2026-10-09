"""Unit and integration tests for Gateway REST endpoints, auth, and WebSockets (tests/unit/test_gateway_routes_and_ws.py)."""

import asyncio

import pytest
from fastapi.testclient import TestClient

from lora.gateway.auth import IpAuthRateLimiter, TokenValidator
from lora.gateway.queue import JobQueue
from lora.gateway.server import create_gateway_app
from lora.storage.notebook import NotebookStore, SavedNotebook


@pytest.fixture
def test_setup(tmp_path):
    token_validator = TokenValidator(desktop_token="desk_secret_token", mobile_token="mob_secret_token")
    queue = JobQueue()
    ip_limiter = IpAuthRateLimiter(max_attempts=5, window_seconds=60.0, ban_seconds=300.0)
    db_path = tmp_path / "test_gateway_nb.db"
    notebook_store = NotebookStore(db_path=db_path)
    asyncio.run(notebook_store.initialize())

    import time
    now = time.time()
    # Pre-populate notebooks
    asyncio.run(
        notebook_store.save_notebook(
            SavedNotebook(
                id="nb-public",
                title="Quantum Physics",
                depth_mode="standard",
                answer="Quantum mechanics overview",
                created_at=now,
                visibility="public",
            )
        )
    )
    asyncio.run(
        notebook_store.save_notebook(
            SavedNotebook(
                id="nb-desktop",
                title="Local Tax Returns",
                depth_mode="detailed",
                answer="Private files",
                created_at=now,
                visibility="desktop_only",
            )
        )
    )

    return {
        "validator": token_validator,
        "queue": queue,
        "limiter": ip_limiter,
        "notebook_store": notebook_store,
    }


def test_healthz_unauthenticated(test_setup):
    app = create_gateway_app(
        origin="desktop",
        queue=test_setup["queue"],
        token_validator=test_setup["validator"],
        ip_limiter=test_setup["limiter"],
        notebook_store=test_setup["notebook_store"],
    )
    client = TestClient(app)
    res = client.get("/healthz")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}
    # Verify no internals leaked
    assert "version" not in res.json()
    assert "uptime" not in res.json()


def test_root_redirects_to_app(test_setup):
    app = create_gateway_app(
        origin="desktop",
        queue=test_setup["queue"],
        token_validator=test_setup["validator"],
        ip_limiter=test_setup["limiter"],
        notebook_store=test_setup["notebook_store"],
    )
    client = TestClient(app)
    res = client.get("/", follow_redirects=False)
    assert res.status_code in (302, 307)
    assert res.headers["location"] == "/app"


def test_desktop_auth_enforcement(test_setup):
    app = create_gateway_app(
        origin="desktop",
        queue=test_setup["queue"],
        token_validator=test_setup["validator"],
        ip_limiter=test_setup["limiter"],
        notebook_store=test_setup["notebook_store"],
    )
    client = TestClient(app)

    # Missing token -> 401
    res = client.get("/v1/notebooks")
    assert res.status_code == 401

    # Invalid token -> 401
    res = client.get("/v1/notebooks", headers={"Authorization": "Bearer wrong_token"})
    assert res.status_code == 401

    # Mobile token on desktop listener -> 401 (token isolation)
    res = client.get("/v1/notebooks", headers={"Authorization": "Bearer mob_secret_token"})
    assert res.status_code == 401

    # Desktop token on desktop listener -> 200
    res = client.get("/v1/notebooks", headers={"Authorization": "Bearer desk_secret_token"})
    assert res.status_code == 200
    nbs = res.json()["notebooks"]
    assert len(nbs) == 2  # Desktop sees both public and desktop_only notebooks


def test_mobile_notebooks_filtering(test_setup):
    app = create_gateway_app(
        origin="mobile",
        queue=test_setup["queue"],
        token_validator=test_setup["validator"],
        ip_limiter=test_setup["limiter"],
        notebook_store=test_setup["notebook_store"],
    )
    client = TestClient(app, base_url="http://192.168.50.10:8765")

    # Mobile token on mobile listener
    res = client.get(
        "/v1/notebooks",
        headers={
            "Authorization": "Bearer mob_secret_token",
            "Host": "192.168.50.10:8765",
        },
    )
    assert res.status_code == 200
    nbs = res.json()["notebooks"]
    # Mobile MUST NOT see desktop_only notebooks!
    assert len(nbs) == 1
    assert nbs[0]["id"] == "nb-public"


def test_mobile_run_mode_restriction(test_setup):
    app = create_gateway_app(
        origin="mobile",
        queue=test_setup["queue"],
        token_validator=test_setup["validator"],
        ip_limiter=test_setup["limiter"],
        notebook_store=test_setup["notebook_store"],
    )
    client = TestClient(app, base_url="http://192.168.50.10:8765")
    headers = {"Authorization": "Bearer mob_secret_token", "Host": "192.168.50.10:8765"}

    # Allowed: quick or standard
    res_quick = client.post("/v1/ask", json={"text": "What is gravity?", "run_mode": "quick"}, headers=headers)
    assert res_quick.status_code == 202
    assert "job_id" in res_quick.json()

    # Denied on mobile: deep or deep_with_refinement -> 403 Forbidden
    res_deep = client.post("/v1/ask", json={"text": "Deep research", "run_mode": "deep"}, headers=headers)
    assert res_deep.status_code == 403
    assert "forbidden" in res_deep.json()["detail"].lower() or "mobile" in res_deep.json()["detail"].lower()


def test_mobile_ask_notebook_authorization(test_setup):
    app = create_gateway_app(
        origin="mobile",
        queue=test_setup["queue"],
        token_validator=test_setup["validator"],
        ip_limiter=test_setup["limiter"],
        notebook_store=test_setup["notebook_store"],
    )
    client = TestClient(app, base_url="http://192.168.50.10:8765")
    headers = {"Authorization": "Bearer mob_secret_token", "Host": "192.168.50.10:8765"}

    # 1. Allowed notebook append to public notebook
    res_pub = client.post(
        "/v1/ask",
        json={"text": "Followup on quantum", "notebook_id": "nb-public"},
        headers=headers,
    )
    assert res_pub.status_code == 202

    # 2. Denied attempt to append to desktop_only notebook -> 403 Forbidden
    res_desk = client.post(
        "/v1/ask",
        json={"text": "Read tax details", "notebook_id": "nb-desktop"},
        headers=headers,
    )
    assert res_desk.status_code == 403
    assert "desktop origin" in res_desk.json()["detail"].lower() or "not accessible" in res_desk.json()["detail"].lower()

    # 3. Nonexistent notebook -> 404 Not Found
    res_nonexistent = client.post(
        "/v1/ask",
        json={"text": "Query", "notebook_id": "nb-unknown"},
        headers=headers,
    )
    assert res_nonexistent.status_code == 404


def test_stop_endpoint_scoped_by_origin(test_setup):
    queue = test_setup["queue"]
    app_mob = create_gateway_app(
        origin="mobile",
        queue=queue,
        token_validator=test_setup["validator"],
        ip_limiter=test_setup["limiter"],
        notebook_store=test_setup["notebook_store"],
    )
    client_mob = TestClient(app_mob, base_url="http://192.168.50.10:8765")
    mob_headers = {"Authorization": "Bearer mob_secret_token", "Host": "192.168.50.10:8765"}

    # Create a desktop job directly in queue
    desk_app = create_gateway_app(
        origin="desktop",
        queue=queue,
        token_validator=test_setup["validator"],
        ip_limiter=test_setup["limiter"],
        notebook_store=test_setup["notebook_store"],
    )
    client_desk = TestClient(desk_app)
    desk_headers = {"Authorization": "Bearer desk_secret_token"}

    res_desk_job = client_desk.post("/v1/ask", json={"text": "Desktop command"}, headers=desk_headers)
    assert res_desk_job.status_code == 202
    desk_job_id = res_desk_job.json()["job_id"]

    # Mobile attempts to cancel desktop job -> 403 Forbidden
    stop_attempt = client_mob.post("/v1/stop", json={"job_id": desk_job_id}, headers=mob_headers)
    assert stop_attempt.status_code == 403

    # Desktop cancels its own job -> 200 OK
    stop_desk = client_desk.post("/v1/stop", json={"job_id": desk_job_id}, headers=desk_headers)
    assert stop_desk.status_code == 200
    assert stop_desk.json()["stopped"] is True


def test_websocket_auth_handshake(test_setup):
    app = create_gateway_app(
        origin="desktop",
        queue=test_setup["queue"],
        token_validator=test_setup["validator"],
        ip_limiter=test_setup["limiter"],
        notebook_store=test_setup["notebook_store"],
    )
    client = TestClient(app)

    # Valid handshake
    with client.websocket_connect("/v1/events") as websocket:
        websocket.send_json({"type": "auth", "token": "desk_secret_token"})
        ack = websocket.receive_json()
        assert ack["type"] == "auth_ok"
        assert ack["origin"] == "desktop"


def test_websocket_auth_handshake_failure(test_setup):
    app = create_gateway_app(
        origin="desktop",
        queue=test_setup["queue"],
        token_validator=test_setup["validator"],
        ip_limiter=test_setup["limiter"],
        notebook_store=test_setup["notebook_store"],
    )
    client = TestClient(app)

    # Invalid token closes with 1008 / 4401
    with pytest.raises(Exception):
        with client.websocket_connect("/v1/events") as websocket:
            websocket.send_json({"type": "auth", "token": "invalid_token"})
            websocket.receive_json()


def test_static_app_mounted(test_setup):
    app = create_gateway_app(
        origin="mobile",
        queue=test_setup["queue"],
        token_validator=test_setup["validator"],
        ip_limiter=test_setup["limiter"],
        notebook_store=test_setup["notebook_store"],
    )
    client = TestClient(app, base_url="http://192.168.50.10:8765")

    res = client.get("/app/")
    assert res.status_code == 200
    assert "L.O.R.A Mobile" in res.text
    assert "hydro-orb-canvas" in res.text

