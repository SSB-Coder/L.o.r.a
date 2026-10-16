"""Hard deterministic security and capability policy engine for L.O.R.A.

Implements strict, non-bypassable policy evaluation:
1. Hard-deny rules evaluated FIRST and are non-overridable.
2. Canonical path traversal defense and root-containment validation.
3. Dangerous extension blocking (.exe, .bat, .cmd, .ps1, scripts).
4. Origin-based enforcement: mobile is strictly confined to Tier 0 research tools.
5. Domain allowlists for browser typing.
"""

import os
from enum import Enum, StrEnum
from fnmatch import fnmatch
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from lora.core.config import ConfigManager, PolicyConfig


class ActionRiskTier(int, Enum):
    TIER0 = 0  # Read-only search, inspection, web research
    TIER1 = 1  # Reversible UI, app launch, window focus
    TIER2 = 2  # Content producing, file creation in outputs
    TIER3 = 3  # Forbidden actions: delete, shell, sys config, format


class PolicyVerdict(StrEnum):
    ALLOW = "ALLOW"
    ASK = "ASK"
    HARD_DENY = "HARD_DENY"


class PolicyResult(BaseModel):
    allowed: bool
    verdict: PolicyVerdict
    reason: str
    rule_matched: str | None = None
    canonical_path: Path | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PolicyEngine:
    """Deterministic security policy engine for L.O.R.A."""

    DEFAULT_ALLOWED_ROOTS = [
        str(Path.home() / "Documents"),
        str(Path.home() / "Downloads"),
        str(Path.home() / "Desktop"),
        str(Path.cwd().resolve()),
    ]

    DEFAULT_BLOCKED_EXTENSIONS = [
        ".exe",
        ".bat",
        ".cmd",
        ".ps1",
        ".psm1",
        ".vbs",
        ".vbe",
        ".js",
        ".jse",
        ".wsf",
        ".wsh",
        ".reg",
        ".dll",
        ".sys",
        ".scr",
        ".cpl",
        ".msi",
        ".msp",
        ".com",
        ".hta",
        ".pif",
        ".docm",
        ".xlsm",
        ".pptm",
    ]

    DEFAULT_SAFE_EXTENSIONS = [
        ".txt",
        ".md",
        ".pdf",
        ".py",
        ".json",
        ".yaml",
        ".yml",
        ".csv",
        ".tsv",
        ".docx",
        ".xlsx",
        ".pptx",
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".svg",
        ".html",
        ".htm",
    ]

    PROTECTED_SYSTEM_DIRECTORIES = [
        "c:\\windows",
        "c:\\program files",
        "c:\\program files (x86)",
        "c:\\programdata",
        "c:\\recovery",
        "c:\\system volume information",
        "c:\\boot",
        "/bin",
        "/sbin",
        "/usr/bin",
        "/usr/sbin",
        "/lib64",
        "/proc",
        "/sys",
        "/etc",
        "/root",
        "/var/log",
    ]

    DESTRUCTIVE_TOOL_PATTERNS = [
        "*delete*",
        "*remove*",
        "*unlink*",
        "*rmdir*",
        "*wipe*",
        "*format*",
        "*erase*",
        "*destroy*",
        "*drop*",
        "*truncate*",
        "*del*",
        "*shred*",
        "*kill*",
        "*reformat*",
        "*mkfs*",
        "*zap*",
        "*rm_rf*",
        "*clear-content*",
        "*bootloader*",
    ]

    SHELL_TOOL_PATTERNS = [
        "*shell*",
        "*bash*",
        "*powershell*",
        "*cmd*",
        "*exec*",
    ]

    def __init__(
        self,
        config_path: Path | str | None = None,
        allowed_roots: list[str] | None = None,
        safe_extensions: list[str] | None = None,
        blocked_extensions: list[str] | None = None,
        typing_domain_allowlist: list[str] | None = None,
    ) -> None:
        self.config: PolicyConfig | None = None
        if config_path and Path(config_path).exists():
            self.config = ConfigManager().load_policy(config_path)

        # Resolve allowed roots to absolute canonical paths
        roots = allowed_roots if allowed_roots is not None else self.DEFAULT_ALLOWED_ROOTS
        self.allowed_roots: list[Path] = [Path(r).resolve() for r in roots]

        self.blocked_extensions: set[str] = {
            ext.lower()
            for ext in (blocked_extensions if blocked_extensions is not None else self.DEFAULT_BLOCKED_EXTENSIONS)
        }
        self.safe_extensions: set[str] = {
            ext.lower()
            for ext in (safe_extensions if safe_extensions is not None else self.DEFAULT_SAFE_EXTENSIONS)
        }

        # Typing domain allowlist
        if typing_domain_allowlist is not None:
            self.typing_domain_allowlist = typing_domain_allowlist
        elif self.config and self.config.typing_domain_allowlist:
            self.typing_domain_allowlist = self.config.typing_domain_allowlist
        else:
            self.typing_domain_allowlist = [
                "google.com",
                "duckduckgo.com",
                "chat.deepseek.com",
                "claude.ai",
                "chatgpt.com",
            ]

    def check_action(
        self,
        tool: str,
        origin: str,
        tier: int | ActionRiskTier,
        params: dict[str, Any] | None = None,
    ) -> PolicyResult:
        """Evaluate an action attempt against hard deterministic policy rules."""
        tool_lower = tool.lower()
        origin_lower = origin.lower()
        tier_int = int(tier)

        # -------------------------------------------------------------
        # STEP 1: Hard Deny Rules (Evaluated FIRST, Non-overridable)
        # -------------------------------------------------------------

        # 1.1 Deletion / destructive tool match
        for pat in self.DESTRUCTIVE_TOOL_PATTERNS:
            if fnmatch(tool_lower, pat):
                return PolicyResult(
                    allowed=False,
                    verdict=PolicyVerdict.HARD_DENY,
                    reason=f"Destructive operation '{tool}' is permanently forbidden by system policy",
                    rule_matched="destructive_tool_denylist",
                )

        # 1.2 Shell execution match
        for pat in self.SHELL_TOOL_PATTERNS:
            if fnmatch(tool_lower, pat):
                return PolicyResult(
                    allowed=False,
                    verdict=PolicyVerdict.HARD_DENY,
                    reason=f"Raw shell execution '{tool}' is strictly forbidden by policy",
                    rule_matched="shell_execution_denylist",
                )

        # 1.3 Tier 3 is unconditionally forbidden
        if tier_int >= ActionRiskTier.TIER3:
            return PolicyResult(
                allowed=False,
                verdict=PolicyVerdict.HARD_DENY,
                reason=f"Tier 3 action '{tool}' is forbidden by system policy with zero exceptions",
                rule_matched="tier3_forbidden",
            )

        # 1.4 Mobile origin restrictions
        if origin_lower in ("mobile", "user_remote"):
            # Mobile is strictly restricted to Tier 0 research
            if tier_int > ActionRiskTier.TIER0:
                return PolicyResult(
                    allowed=False,
                    verdict=PolicyVerdict.HARD_DENY,
                    reason=f"Mobile origin requests cannot execute Tier {tier_int} actions",
                    rule_matched="mobile_tier_restriction",
                )

            # Mobile cannot access local filesystem or launch applications
            forbidden_mobile_tools = [
                "list_dir",
                "read_file",
                "search_files",
                "open_app",
                "open_path",
                "focus_window",
                "browser_type",
                "mouse_click",
                "keyboard_press",
                "export_report",
                "create_note",
                "take_screenshot",
                "system_shutdown",
            ]
            if tool_lower in forbidden_mobile_tools:
                return PolicyResult(
                    allowed=False,
                    verdict=PolicyVerdict.HARD_DENY,
                    reason=f"Mobile origin is forbidden from invoking local desktop tool '{tool}'",
                    rule_matched="mobile_tool_restriction",
                )

        # -------------------------------------------------------------
        # STEP 2: Allow / Ask Rules
        # -------------------------------------------------------------

        # Mobile allowed research tools
        if origin_lower in ("mobile", "user_remote") and tier_int == ActionRiskTier.TIER0:
            allowed_mobile_tools = [
                "web_search",
                "fetch_page",
                "pdf_fetch",
                "knowledge_query",
                "notebook_query",
                "rewrite_depth",
            ]
            if tool_lower in allowed_mobile_tools:
                return PolicyResult(
                    allowed=True,
                    verdict=PolicyVerdict.ALLOW,
                    reason=f"Tool '{tool}' is an allowed mobile Tier 0 research capability",
                    rule_matched="mobile_research_allow",
                )

        # Desktop Tier 0 is auto-allowed
        if origin_lower == "desktop" and tier_int == ActionRiskTier.TIER0:
            return PolicyResult(
                allowed=True,
                verdict=PolicyVerdict.ALLOW,
                reason=f"Tool '{tool}' is an allowed desktop Tier 0 capability",
                rule_matched="desktop_tier0_allow",
            )

        # Desktop Tier 1 is allowed with supervisor / confirmation
        if origin_lower == "desktop" and tier_int == ActionRiskTier.TIER1:
            return PolicyResult(
                allowed=True,
                verdict=PolicyVerdict.ALLOW,
                reason=f"Tier 1 action '{tool}' permitted under desktop supervision",
                rule_matched="desktop_tier1_supervised",
            )

        # Desktop Tier 2 requires explicit user confirmation
        if origin_lower == "desktop" and tier_int == ActionRiskTier.TIER2:
            return PolicyResult(
                allowed=True,
                verdict=PolicyVerdict.ASK,
                reason=f"Tier 2 action '{tool}' requires user confirmation before proceeding",
                rule_matched="desktop_tier2_confirmation",
            )

        return PolicyResult(
            allowed=False,
            verdict=PolicyVerdict.HARD_DENY,
            reason=f"No matching permission rule for tool '{tool}', origin '{origin}', tier {tier_int}",
            rule_matched="default_deny",
        )

    def check_path(
        self,
        path: str | Path,
        origin: str,
        mode: str = "read",
    ) -> PolicyResult:
        """Validate filesystem path against allowed roots, system restrictions, and extension blocklists."""
        origin_lower = origin.lower()

        # Rule 1: Mobile origin cannot access local filesystem paths under any circumstances
        if origin_lower == "mobile":
            return PolicyResult(
                allowed=False,
                verdict=PolicyVerdict.HARD_DENY,
                reason="Mobile origin requests are strictly forbidden from accessing local filesystem",
                rule_matched="mobile_filesystem_block",
            )

        path_str = str(path).strip()
        path_lower = path_str.lower()

        # Rule 2: Block URI schemes, malformed paths, or null bytes
        if path_lower.startswith("file:") or path_lower.startswith("http:") or "\x00" in path_str:
            return PolicyResult(
                allowed=False,
                verdict=PolicyVerdict.HARD_DENY,
                reason="Malformed path or URI scheme blocked",
                rule_matched="malformed_path_block",
            )

        # Rule 2.1: Block NTFS Alternate Data Streams (ADS) e.g., file.txt:stream or ::$DATA
        path_without_drive = path_str[2:] if (len(path_str) >= 2 and path_str[1] == ":") else path_str
        if ":" in path_without_drive:
            return PolicyResult(
                allowed=False,
                verdict=PolicyVerdict.HARD_DENY,
                reason="Alternate Data Streams (ADS) syntax is strictly forbidden",
                rule_matched="ads_stream_block",
            )

        # Rule 3: Block UNC network shares and admin shares
        if path_str.startswith("\\\\") or path_str.startswith("//"):
            return PolicyResult(
                allowed=False,
                verdict=PolicyVerdict.HARD_DENY,
                reason="UNC network paths and administrative shares are strictly forbidden",
                rule_matched="unc_path_block",
            )

        # Rule 4: Check dangerous / script extensions
        clean_ext_target = path_without_drive.split(":")[0]
        suffix = Path(clean_ext_target).suffix.lower()
        if suffix in self.blocked_extensions:
            return PolicyResult(
                allowed=False,
                verdict=PolicyVerdict.HARD_DENY,
                reason=f"Forbidden: extension '{suffix}' is on the dangerous file execution blocklist",
                rule_matched="blocked_extension",
            )

        # Resolve path to canonical absolute path (expanding 8.3 short names and symlinks)
        try:
            expanded = os.path.expandvars(os.path.expanduser(path_str))
            real_path = os.path.realpath(expanded)
            resolved = Path(real_path).resolve()
        except Exception as e:
            return PolicyResult(
                allowed=False,
                verdict=PolicyVerdict.HARD_DENY,
                reason=f"Path resolution failed: {e}",
                rule_matched="path_resolution_error",
            )

        resolved_str = str(resolved).lower()

        # Rule 5: Check protected operating system directories
        for sys_dir in self.PROTECTED_SYSTEM_DIRECTORIES:
            sys_clean = sys_dir.lower()
            if resolved_str == sys_clean or resolved_str.startswith(sys_clean + "\\") or resolved_str.startswith(sys_clean + "/"):
                return PolicyResult(
                    allowed=False,
                    verdict=PolicyVerdict.HARD_DENY,
                    reason=f"Access to protected operating system directory '{sys_dir}' is strictly forbidden",
                    rule_matched="protected_os_directory",
                )

        # Rule 6: Check root directory escape
        # Path must be contained within at least one of the allowed roots
        is_contained = False
        for root in self.allowed_roots:
            try:
                # Path.is_relative_to introduced in Python 3.9
                if resolved.is_relative_to(root):
                    is_contained = True
                    break
            except (ValueError, AttributeError):
                continue

        if not is_contained:
            return PolicyResult(
                allowed=False,
                verdict=PolicyVerdict.HARD_DENY,
                reason=f"Path '{path}' resolves outside allowed roots (path traversal blocked)",
                rule_matched="path_traversal_blocked",
            )

        # Rule 7: Write mode restrictions (only safe files can be created)
        if mode == "write":
            if suffix and suffix not in self.safe_extensions:
                return PolicyResult(
                    allowed=False,
                    verdict=PolicyVerdict.HARD_DENY,
                    reason=f"Cannot write file with unapproved extension '{suffix}'",
                    rule_matched="unsafe_write_extension",
                )

        return PolicyResult(
            allowed=True,
            verdict=PolicyVerdict.ALLOW,
            reason="Path validated successfully within allowed roots",
            rule_matched="path_allowed",
            canonical_path=resolved,
        )

    def check_typing_domain(self, domain_or_url: str) -> PolicyResult:
        """Validate domain against the external AI/browser typing allowlist."""
        target = domain_or_url.strip()
        if "://" in target:
            try:
                parsed = urlparse(target)
                target = parsed.netloc
            except Exception:
                pass

        # Strip port if present
        if ":" in target:
            target = target.split(":")[0]

        target_lower = target.lower()

        for allowed_domain in self.typing_domain_allowlist:
            allowed_clean = allowed_domain.lower()
            if target_lower == allowed_clean or target_lower.endswith("." + allowed_clean):
                return PolicyResult(
                    allowed=True,
                    verdict=PolicyVerdict.ALLOW,
                    reason=f"Domain '{target}' is on the typing allowlist",
                    rule_matched="typing_domain_allow",
                )

        return PolicyResult(
            allowed=False,
            verdict=PolicyVerdict.HARD_DENY,
            reason=f"Typing into external domain '{domain_or_url}' is not on the approved allowlist",
            rule_matched="typing_domain_denied",
        )
