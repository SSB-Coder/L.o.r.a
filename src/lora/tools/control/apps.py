"""Desktop Application and Web App Launching Tools for L.O.R.A."""

import ipaddress
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
        for p in Path(__file__).resolve().parents:
            cand = p / "config" / "apps.yaml"
            if cand.exists():
                return cand
        return Path("config/apps.yaml")

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

    def open_app(self, name: str) -> dict[str, Any]:
        """Launch an allowlisted desktop application."""
        app_info = self.resolve_app(name)
        if not app_info:
            return {
                "status": "error",
                "message": f"Application '{name}' is not on the allowed application list.",
            }

        command = app_info["command"]
        try:
            # shell=False to prevent shell injection
            try:
                subprocess.Popen([command], shell=False)
            except FileNotFoundError:
                # 1) shutil.which honours PATHEXT, so the allowlisted "code" finds code.cmd.
                # 2) Windows apps like chrome.exe live under "App Paths", which CreateProcess (Popen) ignores
                #    but ShellExecute (os.startfile) honours. The name is allowlisted either way.
                resolved = shutil.which(command)
                if resolved:
                    subprocess.Popen([resolved], shell=False)
                elif sys.platform == "win32":
                    os.startfile(command)  # noqa: S606
                else:
                    raise
            return {
                "status": "success",
                "message": f"Successfully launched {app_info['name']}.",
                "app": app_info["name"],
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to launch application '{name}': {e}",
            }

    def _check_blocked_host(self, host: str) -> tuple[bool, str]:
        """Check if a host is localhost, private/loopback IP, or router gateway."""
        if not host:
            return True, "URL host is missing or invalid."

        host_clean = host.strip("[]").lower()

        # 1. Localhost or internal domain checks
        if (
            host_clean == "localhost"
            or host_clean.endswith(".localhost")
            or host_clean.endswith(".local")
            or host_clean.endswith(".internal")
            or host_clean.endswith(".lan")
        ):
            return True, f"Access to local or private host '{host_clean}' is prohibited."

        # 2. Known router admin host checks
        if host_clean in self.BLOCKED_ROUTER_HOSTS:
            return True, f"Access to internal router administration host '{host_clean}' is blocked."

        # 3. Check for integer / hex IP formats (e.g. 2130706433 or 0x7f000001)
        try:
            val = None
            if host_clean.isdigit():
                val = int(host_clean)
            elif host_clean.startswith("0x") or host_clean.startswith("0X"):
                val = int(host_clean, 16)
            if val is not None:
                ip_obj = ipaddress.ip_address(val)
                if ip_obj.is_loopback or ip_obj.is_private or ip_obj.is_reserved or ip_obj.is_unspecified:
                    return True, f"Access to private/loopback IP '{ip_obj}' is blocked."
        except Exception:
            pass

        # 4. Standard dotted / IPv6 IP check
        try:
            ip = ipaddress.ip_address(host_clean)
            if ip.is_loopback or ip.is_unspecified:
                return True, f"Access to loopback IP '{host_clean}' is blocked."
            if ip.is_private and (
                str(ip).endswith(".1")
                or str(ip).endswith(".254")
                or str(ip).startswith("127.")
                or str(ip).startswith("10.")
                or str(ip).startswith("192.168.")
            ):
                return True, f"Access to internal gateway/private IP '{host_clean}' is blocked."
        except ValueError:
            pass

        # 5. Socket-level parsing for octal or other alternative notations (e.g. 0177.0.0.1)
        try:
            packed = socket.inet_aton(host_clean)
            ip_str = socket.inet_ntoa(packed)
            ip_from_socket = ipaddress.ip_address(ip_str)
            if ip_from_socket.is_loopback or ip_from_socket.is_unspecified:
                return True, f"Access to loopback IP '{ip_str}' is blocked."
            if ip_from_socket.is_private and (
                ip_str.endswith(".1") or ip_str.endswith(".254") or ip_str.startswith("127.")
            ):
                return True, f"Access to internal gateway/private IP '{ip_str}' is blocked."
        except (OSError, ValueError):
            pass

        return False, ""

    def open_url(self, url: str) -> dict[str, Any]:
        """Validate and open a web URL in the user's default desktop browser."""
        url_clean = url.strip()

        try:
            parsed = urlparse(url_clean)
        except Exception as e:
            return {"status": "error", "message": f"Invalid URL structure: {e}"}

        # 1. Enforce HTTP/HTTPS protocol
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
            webbrowser.open(url_clean)
            return {
                "status": "success",
                "message": f"Opened {url_clean} in default browser.",
                "url": url_clean,
            }
        except Exception as e:
            return {"status": "error", "message": f"Failed to open URL '{url}': {e}"}
