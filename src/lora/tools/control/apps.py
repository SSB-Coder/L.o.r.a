"""Desktop Application and Web App Launching Tools for L.O.R.A."""

import ipaddress
import ntpath
import os
import shutil
import socket
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus, urlparse

import yaml


class AppLauncher:
    """Safely resolves and launches allowlisted applications and web URLs."""

    BLOCKED_ROUTER_HOSTS = {
        "192.168.1.1",
        "192.168.0.1",
        "192.168.50.1",
        "10.0.0.1",
        "routerlogin.net",
        "tplinkwifi.net",
        "localhost",
        "127.0.0.1",
        "0.0.0.0",
        "::1",
    }

    @staticmethod
    def _find_default_config() -> Path:
        from lora.core.paths import default_config_path

        for p in Path(__file__).resolve().parents:
            cand = p / "config" / "apps.yaml"
            if cand.exists():
                return cand
        local = Path("config/apps.yaml")
        return local if local.is_file() else default_config_path("apps.yaml")

    def __init__(self, config_path: Path | str | None = None) -> None:
        self.config_path = Path(config_path) if config_path else self._find_default_config()
        self.applications: dict[str, dict[str, Any]] = {}
        self.web_apps: dict[str, dict[str, Any]] = {}
        self._load_config()

    def _load_config(self) -> None:
        if self.config_path.exists():
            with open(self.config_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                self.applications = data.get("applications", {})
                self.web_apps = data.get("web_apps", {})

    def resolve_app(self, name: str) -> dict[str, Any] | None:
        """Resolve an application query against allowlisted names and aliases."""
        clean = name.strip().lower()

        # Check exact key match
        if clean in self.applications:
            return self.applications[clean]

        # Check aliases
        for app_info in self.applications.values():
            if clean == app_info.get("name", "").lower():
                return app_info
            aliases = [a.lower() for a in app_info.get("aliases", [])]
            if clean in aliases:
                return app_info

        return None

    def resolve_web_app(self, name: str, query: str = "") -> dict[str, Any] | None:
        """Resolve a web app shortcut and optional query parameter."""
        clean = name.strip().lower()

        target_app = None
        if clean in self.web_apps:
            target_app = self.web_apps[clean]
        else:
            for info in self.web_apps.values():
                if clean == info.get("name", "").lower():
                    target_app = info
                    break
                aliases = [a.lower() for a in info.get("aliases", [])]
                if clean in aliases:
                    target_app = info
                    break

        if target_app is None:
            return None

        # Build target URL
        if query and target_app.get("search_template"):
            url = target_app["search_template"].format(query=quote_plus(query.strip()))
        else:
            url = target_app["url"]

        return {
            "name": target_app["name"],
            "url": url,
        }

    @staticmethod
    def _command_candidates(app_info: dict[str, Any]) -> list[str]:
        """Platform alternatives are configuration, never user-supplied shell text."""
        commands = app_info.get("commands", {}).get(sys.platform)
        if commands is None:
            commands = app_info.get("linux_command") if sys.platform.startswith("linux") else None
        if commands is None:
            commands = app_info.get("command", [])
        if isinstance(commands, str):
            commands = [commands]
        return list(dict.fromkeys(c for c in commands if isinstance(c, str) and c.strip()))

    @staticmethod
    def _find_windows_app_candidate(cmd: str, app_info: dict[str, Any]) -> Path | None:
        if sys.platform != "win32":
            return None
        for configured in app_info.get("windows_paths", []):
            candidate = Path(ntpath.expandvars(configured))
            if candidate.is_file() and candidate.name.lower() == Path(cmd).name.lower():
                return candidate
        return None

    @staticmethod
    def _linux_desktop_entry(app_info: dict[str, Any]) -> Path | None:
        """Resolve only desktop IDs explicitly authorized in the app allowlist.

        gio handles Exec field codes and D-Bus activation; never parse/execute a
        .desktop Exec string ourselves or launch an arbitrary requested path.
        """
        if not sys.platform.startswith("linux"):
            return None
        roots = [Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local/share")))]
        roots.extend(Path(p) for p in os.environ.get("XDG_DATA_DIRS", "/usr/local/share:/usr/share").split(os.pathsep) if p)
        roots.extend([Path.home() / ".local/share/flatpak/exports/share", Path("/var/lib/flatpak/exports/share")])
        for desktop_id in app_info.get("desktop_ids", []):
            if not isinstance(desktop_id, str) or Path(desktop_id).name != desktop_id or not desktop_id.endswith(".desktop"):
                continue
            for root in roots:
                entry = root / "applications" / desktop_id
                if entry.is_file():
                    return entry
        return None

    def _start_app(self, app_info: dict[str, Any]) -> None:
        bundle_id = app_info.get("macos_bundle")
        if sys.platform == "darwin" and bundle_id and "darwin" not in app_info.get("commands", {}):
            try:
                subprocess.run(["/usr/bin/open", "-b", bundle_id], check=True, capture_output=True, text=True, timeout=15)
                return
            except (OSError, subprocess.SubprocessError):
                pass
        for command in self._command_candidates(app_info):
            try:
                subprocess.Popen([command], shell=False)
                return
            except OSError:
                resolved = shutil.which(command)
                if resolved:
                    try:
                        subprocess.Popen([resolved], shell=False)
                        return
                    except OSError:
                        pass
                if sys.platform == "win32":
                    candidate = self._find_windows_app_candidate(command, app_info)
                    if candidate:
                        subprocess.Popen([str(candidate)], shell=False)
                        return
                    try:
                        os.startfile(command)
                        return
                    except OSError:
                        pass
        entry = self._linux_desktop_entry(app_info)
        if entry:
            subprocess.run(["gio", "launch", str(entry)], check=True, capture_output=True, text=True, timeout=15)
            return
        raise FileNotFoundError("No configured executable or desktop launcher is installed. Check this app's entry in config/apps.yaml.")

    def open_app(self, name: str) -> dict[str, Any]:
        """Launch an allowlisted desktop application, using its configured alternatives."""
        app_info = self.resolve_app(name)
        if not app_info:
            return {"status": "error", "message": f"Application '{name}' is not on the allowed application list."}
        try:
            self._start_app(app_info)
            return {"status": "success", "message": f"Successfully launched {app_info['name']}.", "app": app_info["name"]}
        except Exception as exc:
            return {"status": "error", "message": f"Failed to launch application '{name}': {exc}"}

    def _check_blocked_host(self, host: str) -> tuple[bool, str]:
        # Reject numeric aliases locally even if DNS is unavailable. Windows DNS
        # does not consistently resolve the legacy hex/octal IPv4 forms.
        from lora.safety.egress import EgressFilter

        blocked, reason = EgressFilter()._is_ip_or_loopback_blocked(host)
        if blocked:
            return True, reason
        if not host or host.lower() in self.BLOCKED_ROUTER_HOSTS or host.lower().endswith((".local", ".lan", ".internal", ".localhost")) or host.lower() == "localhost":
            return True, "Local/private destination blocked"
        try:
            addresses = {info[4][0] for info in socket.getaddrinfo(host.strip("[]"), None, type=socket.SOCK_STREAM)}
            if not addresses or any(not ipaddress.ip_address(addr).is_global for addr in addresses):
                return True, "Local/private destination blocked"
        except (OSError, ValueError):
            return True, "Destination could not be validated"
        return False, ""

    def open_url(self, url: str) -> dict[str, Any]:
        """Validate and open a web URL in the user's default desktop browser."""
        url_clean = url.strip()

        try:
            parsed = urlparse(url_clean)
        except Exception as e:
            return {"status": "error", "message": f"Invalid URL structure: {e}"}

        # 1. Enforce HTTP/HTTPS protocol
        if parsed.username is not None or parsed.password is not None:
            return {"status": "error", "message": "Credentials in URLs are prohibited."}
        if (parsed.scheme or "").lower() not in ["http", "https"]:
            return {
                "status": "error",
                "message": f"Forbidden URL scheme '{parsed.scheme}'. Only HTTP and HTTPS are permitted.",
            }

        # 2. Block router and sensitive internal admin hosts, loopbacks, and SSRF
        host = (parsed.hostname or "").lower()
        is_blocked, reason = self._check_blocked_host(host)
        if is_blocked:
            return {
                "status": "error",
                "message": reason,
            }

        try:
            if not webbrowser.open(url_clean):
                return {"status": "error", "message": "No default browser accepted the URL."}
            return {
                "status": "success",
                "message": f"Opened {url_clean} in default browser.",
                "url": url_clean,
            }
        except Exception as e:
            return {"status": "error", "message": f"Failed to open URL '{url}': {e}"}
