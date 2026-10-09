import datetime
import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


class MemoryStore:
    """Manages reading and appending curated long-term memories in user_profile.md."""

    def __init__(self, profile_path: Path) -> None:
        self._profile_path = profile_path
        self._lock = threading.Lock()

    @property
    def profile_path(self) -> Path:
        return self._profile_path

    def get_profile_markdown(self) -> str:
        """Reads and returns the current user profile markdown text."""
        with self._lock:
            if not self._profile_path.is_file():
                return ""
            try:
                return self._profile_path.read_text(encoding="utf-8")
            except Exception as e:
                logger.warning("Failed to read user profile at %s: %s", self._profile_path, e)
                return ""

    def append_note(self, category: str, note: str) -> None:
        """Appends a categorized memory note with a timestamp to user_profile.md."""
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        block = f"\n### {category} ({now_str})\n- {note}\n"

        with self._lock:
            try:
                if not self._profile_path.is_file():
                    self._profile_path.parent.mkdir(parents=True, exist_ok=True)
                    self._profile_path.write_text("# User Profile & Preferences\n", encoding="utf-8")

                with open(self._profile_path, "a", encoding="utf-8") as f:
                    f.write(block)
                logger.info("Appended memory note to %s [%s]", self._profile_path, category)
            except Exception as e:
                logger.error("Failed to append memory note to %s: %s", self._profile_path, e)
