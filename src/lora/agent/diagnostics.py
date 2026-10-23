"""'run diagnostics': a plain-language health check of everything LORA depends on.

The chat gets a short pass/fail list with the fix for anything broken. Technical detail (model names,
versions, paths) goes to the terminal log only.
"""

from __future__ import annotations

import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from lora.net import async_client

logger = logging.getLogger(__name__)


@dataclass
class Check:
    name: str
    ok: bool
    summary: str
    fix: str = ""
    detail: str = ""

    def line(self) -> str:
        mark = "✅" if self.ok else "⚠️"
        fix = f" → {self.fix}" if (self.fix and not self.ok) else ""
        return f"{mark} {self.name}: {self.summary}{fix}"


async def _check_engine(rt) -> Check:
    from lora.models.ollama_client import OllamaClient, match_models
    from lora.models.selection import PREFERRED_MODEL, plan_chat

    client = OllamaClient(base_url=rt.ollama_url)
    try:
        installed = await client.installed_models()
    except Exception as exc:
        return Check("Thinking", False, "I can't reach my local engine.", "start the Ollama app", f"{type(exc).__name__}: {exc}")
    if not installed:
        return Check("Thinking", False, "the engine has no models installed.", "pull your base model with `ollama pull`", "no models")
    plan = await plan_chat(client, user_default=rt.user_default_model)
    has_preferred = bool(match_models(PREFERRED_MODEL, installed))
    detail = f"installed={installed} chosen={plan.model if plan else None} preferred_present={has_preferred}"
    if not has_preferred:
        return Check("Thinking", True, "ready, but my preferred base model isn't installed so I'm using another one.", detail=detail)
    return Check("Thinking", True, "ready.", detail=detail)


async def _check_web() -> Check:
    from lora.agent.research import HEADERS, search_with_report

    try:
        async with async_client(HEADERS, httpx.Timeout(10.0, connect=5.0)) as client:
            hits, report = await search_with_report("python programming language", client)
    except Exception as exc:
        return Check("Browsing", False, "the web search crashed.", "check your internet connection", f"{type(exc).__name__}: {exc}")
    detail = f"backend={report['backend']} errors={report['errors']} hits={len(hits)}"
    if not hits:
        return Check("Browsing", False, "I can't reach any search engine.", "check your internet connection, VPN or firewall", detail)
    if report["backend"] == "wikipedia":
        return Check("Browsing", True, "limited: the main search engines are blocking me, using Wikipedia only.", detail=detail)
    return Check("Browsing", True, "working.", detail=detail)


def _exe_found(command: str) -> bool:
    if shutil.which(command):
        return True
    if sys.platform == "win32":
        try:
            import winreg

            for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
                try:
                    winreg.CloseKey(winreg.OpenKey(hive, rf"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\{command}"))
                    return True
                except OSError:
                    continue
        except Exception:
            pass
    return False


def _check_apps(rt) -> list[Check]:
    out: list[Check] = []
    try:
        reg = rt._get_registry()
    except Exception as exc:
        return [Check("Opening things", False, "my tool layer failed to start.", "see the terminal for details", f"{type(exc).__name__}: {exc}")]
    launcher = reg.app_launcher
    if not launcher.applications:
        out.append(Check("Opening apps", False, "my app list is empty.", "make sure config/apps.yaml exists next to the app", str(launcher.config_path)))
    else:
        missing = []
        for key in ("chrome", "notepad", "calculator"):
            info = launcher.resolve_app(key)
            if info and not _exe_found(info["command"]):
                missing.append(info["name"])
        if missing:
            out.append(Check("Opening apps", False, f"I can't find {', '.join(missing)} on this PC.", "put the full path to the app in config/apps.yaml", f"missing={missing}"))
        else:
            out.append(Check("Opening apps", True, "working.", detail=f"{len(launcher.applications)} apps, {len(launcher.web_apps)} web shortcuts"))
    from lora.tools.files.known import all_known_folders

    bad, lines = [], []
    for name, path in all_known_folders().items():
        verdict = reg.action_gate.policy_engine.check_path(str(path), origin="desktop", mode="read")
        lines.append(f"{name}={path} exists={path.exists()} allowed={verdict.allowed}")
        if not path.exists() or not verdict.allowed:
            bad.append(name)
    if bad:
        out.append(Check("Opening folders", False, f"I can't open: {', '.join(bad)}.", "tell me where they are or allow them in config/policy.yaml", "; ".join(lines)))
    else:
        out.append(Check("Opening folders", True, "working.", detail="; ".join(lines)))
    return out


def _check_sources(rt) -> Check:
    try:
        reg = rt._get_registry()
        path = Path(reg.media_db.embedding_engine.model_path)
    except Exception as exc:
        return Check("Sources", False, "the sources index failed to start.", "see the terminal for details", f"{type(exc).__name__}: {exc}")
    if not path.exists():
        return Check("Sources", True, "working with a basic index (the better search model isn't installed).", detail=f"embedding model path missing: {path}")
    return Check("Sources", True, "working.", detail=str(path))


async def run_diagnostics(rt) -> str:
    extras: dict[str, Any] = {}
    try:
        extras = rt.diag_extras() or {}
    except Exception:
        pass
    checks: list[Check] = [await _check_engine(rt), await _check_web()]
    checks += _check_apps(rt)
    checks.append(_check_sources(rt))

    hub = rt.mcp_hub
    if hub is not None and (hub.servers or hub.errors):
        if hub.errors:
            bad = ", ".join(hub.errors)
            checks.append(Check("Other tools", False, f"these tool servers did not start: {bad}.", "check ~/.lora/mcp.json and that the program (such as npx) is installed", str(hub.errors)))
        else:
            checks.append(Check("Other tools", True, f"{len(hub.servers)} tool server(s) connected, {len(hub.tool_names)} tools.", detail=str(hub.tool_names)))

    mic = extras.get("mic")
    if mic is not None:
        ok, text = mic
        checks.append(Check("Hearing", bool(ok), "your microphone is working." if ok else f"microphone problem: {text}", "check Windows sound settings and that no other app is using the mic", str(text)))
    tts = extras.get("tts")
    if tts is not None:
        checks.append(Check("Voice", tts["ok"], tts["summary"], tts.get("fix", ""), tts.get("detail", "")))

    for c in checks:
        logger.info("diagnostic [%s] ok=%s %s | %s", c.name, c.ok, c.summary, c.detail)
    bad = [c for c in checks if not c.ok]
    head = "Everything checks out." if not bad else f"{len(bad)} thing{'s' if len(bad) != 1 else ''} need attention:"
    return head + "\n\n" + "\n".join(c.line() for c in checks) + "\n\nFull details are in the terminal."
