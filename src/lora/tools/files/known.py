"""Real locations of the user's standard folders.

``Path.home() / "Documents"`` is wrong on many Windows machines: with OneDrive "Known Folder Move"
Documents, Desktop and Pictures live under ``C:\\Users\\<you>\\OneDrive``, and Downloads can be
redirected too. Windows knows the truth (SHGetKnownFolderPath); everything else falls back to the
home directory.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

_FOLDER_IDS = {
    "downloads": "374DE290-123F-4565-9164-39C4925E467B",
    "documents": "FDD39AD0-238F-46AF-ADB4-6C85480369C7",
    "desktop": "B4BFCC3A-DB2C-424C-B029-7FE99A87C641",
    "pictures": "33E28130-4E1E-4676-835A-98395C3BC3BB",
    "music": "4BD8D571-6D19-48D3-BE97-422220080E43",
    "videos": "18989B1D-99B5-455B-841C-AB7C74E4DDFC",
}
FOLDER_NAMES = tuple(_FOLDER_IDS)


def _windows_known_folder(name: str) -> Path | None:
    if sys.platform != "win32":
        return None
    try:
        import ctypes

        class GUID(ctypes.Structure):
            _fields_ = [
                ("Data1", ctypes.c_ulong),
                ("Data2", ctypes.c_ushort),
                ("Data3", ctypes.c_ushort),
                ("Data4", ctypes.c_ubyte * 8),
            ]

        u = uuid.UUID(_FOLDER_IDS[name])
        guid = GUID(u.time_low, u.time_mid, u.time_hi_version, (ctypes.c_ubyte * 8)(*u.bytes[8:]))
        out = ctypes.c_wchar_p()
        shell32 = ctypes.windll.shell32  # type: ignore[attr-defined]
        if shell32.SHGetKnownFolderPath(ctypes.byref(guid), 0, None, ctypes.byref(out)) != 0:
            return None
        try:
            return Path(out.value) if out.value else None
        finally:
            ctypes.windll.ole32.CoTaskMemFree(out)  # type: ignore[attr-defined]
    except Exception:
        return None


def known_folder(name: str) -> Path:
    """Location of 'downloads', 'documents', 'desktop', 'pictures', 'music' or 'videos'."""
    key = name.strip().lower()
    if key not in _FOLDER_IDS:
        raise KeyError(name)
    return _windows_known_folder(key) or (Path.home() / key.capitalize())


def all_known_folders() -> dict[str, Path]:
    return {n: known_folder(n) for n in FOLDER_NAMES}


def known_folder_roots() -> list[str]:
    """Existing standard folders, for the path policy's allowed roots (de-duplicated)."""
    seen: list[str] = []
    for p in all_known_folders().values():
        s = str(p)
        if p.exists() and s not in seen:
            seen.append(s)
    return seen
