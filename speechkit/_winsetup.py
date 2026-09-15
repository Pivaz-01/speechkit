"""
Native library search paths on Windows.

`phonemes_from_audio.py` used to open with three calls like::

    os.add_dll_directory(r"C:\\Users\\<username>\\AppData\\Local\\anaconda3\\envs\\speech_analysis")

which crash on any other machine and raise AttributeError on macOS and Linux,
where `os.add_dll_directory` does not exist at all. The import therefore failed
before reaching line 5 for everyone except the author.

The directories those three lines pointed at are all fixed offsets from the
Python installation that is running, so they can be derived instead of typed:

    <sys.prefix>                            the env root
    <sys.prefix>/Library/bin                where conda puts native DLLs
    <sys.prefix>/Library/mingw-w64/bin      where some conda builds put them
    <sys.prefix>/DLLs                       stock CPython
    <torch package>/lib                     torch's own CUDA/MKL DLLs

That covers conda, venv and stock installs with no configuration. For an
unusual layout, SPEECHKIT_DLL_DIRS accepts extra directories separated by the
platform path separator, and is prepended to the list above.

Call `configure()` once, before torch or parselmouth is imported.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

IS_WINDOWS = sys.platform.startswith("win")

_report: list[str] = []
_configured = False


def _torch_lib_dir() -> Path | None:
    """Locate torch/lib without importing torch (importing it is what we are
    trying to make work)."""
    try:
        spec = importlib.util.find_spec("torch")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    candidate = Path(list(spec.submodule_search_locations)[0]) / "lib"
    return candidate if candidate.is_dir() else None


def candidate_dirs() -> list[Path]:
    dirs: list[Path] = []
    for raw in os.environ.get("SPEECHKIT_DLL_DIRS", "").split(os.pathsep):
        raw = raw.strip().strip('"')
        if raw:
            dirs.append(Path(raw))
    prefix = Path(sys.prefix)
    dirs += [
        prefix,
        prefix / "Library" / "bin",
        prefix / "Library" / "mingw-w64" / "bin",
        prefix / "Library" / "usr" / "bin",
        prefix / "DLLs",
    ]
    torch_lib = _torch_lib_dir()
    if torch_lib is not None:
        dirs.append(torch_lib)

    seen: set[str] = set()
    unique: list[Path] = []
    for d in dirs:
        key = str(d).lower()
        if key not in seen:
            seen.add(key)
            unique.append(d)
    return unique


def configure(allow_duplicate_openmp: bool = True) -> list[str]:
    """
    Register the native library directories and return a human-readable report
    of what happened, which the interface shows on its environment panel.

    Safe to call more than once and on every platform: off Windows it only sets
    the OpenMP variable and returns.
    """
    global _configured
    if _configured:
        return _report

    if allow_duplicate_openmp:
        # torch and MKL can each ship their own OpenMP runtime; loading both
        # aborts the process with an "OMP: Error #15" unless this is set, and
        # it has to be set before either library loads. This is the one part of
        # the original preamble that was not a path problem.
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        _report.append("KMP_DUPLICATE_LIB_OK=TRUE (duplicate OpenMP tolerated)")

    if not IS_WINDOWS:
        _report.append(f"{sys.platform}: no DLL directories needed")
        _configured = True
        return _report

    add = getattr(os, "add_dll_directory", None)
    if add is None:  # Windows on Python < 3.8
        _report.append("os.add_dll_directory unavailable; relying on PATH")
        _configured = True
        return _report

    added = 0
    for d in candidate_dirs():
        if not d.is_dir():
            continue
        try:
            add(str(d))
        except OSError as exc:
            _report.append(f"could not add {d}: {exc}")
            continue
        added += 1
        _report.append(f"added {d}")
    if added == 0:
        _report.append(
            "no native library directories found; if torch or soundfile fails "
            "to import, set SPEECHKIT_DLL_DIRS to the folder holding the DLLs")
    _configured = True
    return _report


def report() -> list[str]:
    return list(_report)
