"""
speechkit — acoustic analysis, phoneme recognition and phoneme-to-text
alignment for speech recordings, behind one local interface.

Three stages, each usable on its own:

    from speechkit.acoustics import analyze_files, run_acoustics
    from speechkit.phonemes import decode_file, run_phonemes
    from speechkit.alignment import run_alignment

The heavy dependencies are imported lazily by the modules that need them, so
importing this package costs nothing and the interface can open and report a
missing dependency rather than failing at import.
"""

from __future__ import annotations

import importlib.util
import sys

from . import _winsetup

__version__ = "1.0.0"

# Must run before torch, parselmouth or soundfile pull in native libraries.
_winsetup.configure()

_REQUIREMENTS = {
    "acoustics": [
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("librosa", "librosa"),
        ("soundfile", "soundfile"),
        ("matplotlib", "matplotlib"),
        ("parselmouth", "praat-parselmouth"),
    ],
    "phonemes": [
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("soundfile", "soundfile"),
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("huggingface_hub", "huggingface_hub"),
    ],
    "alignment": [],          # standard library only
}

_OPTIONAL = {
    "alignment": [("phonemizer", "phonemizer")],
}


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


# The supported range is set by librosa, which needs numba, which needs
# llvmlite. Those three are the slowest packages here to support a new CPython,
# and numba 0.61 dropped Python 3.9. The alignment stage is pure standard
# library and runs on anything, which is why this reports rather than refuses.
PYTHON_TESTED = ((3, 10), (3, 13))
PYTHON_PREFERRED = "3.11 or 3.12"


def python_check() -> dict:
    major, minor = sys.version_info[:2]
    low, high = PYTHON_TESTED
    if (major, minor) < low:
        return {
            "ok": False,
            "message": f"Python {major}.{minor} is too old. librosa cannot install "
                       f"because numba requires {low[0]}.{low[1]} or later, so the "
                       f"acoustic analysis stage will not run. Use "
                       f"{PYTHON_PREFERRED}.",
        }
    if (major, minor) > high:
        return {
            "ok": None,
            "message": f"Python {major}.{minor} is newer than this was tested on "
                       f"({low[0]}.{low[1]} to {high[0]}.{high[1]}). librosa, numba "
                       f"and torch may not have wheels for it yet. If installing "
                       f"fails, use {PYTHON_PREFERRED}.",
        }
    return {"ok": True, "message": f"Python {major}.{minor}"}


def environment() -> dict:
    """
    What is installed, per stage, so the interface can grey out a stage it
    cannot run and say which package is missing instead of showing a traceback.
    """
    stages = {}
    for stage, required in _REQUIREMENTS.items():
        missing = [pip for module, pip in required if not _installed(module)]
        optional = [
            {"package": pip, "installed": _installed(module)}
            for module, pip in _OPTIONAL.get(stage, [])
        ]
        stages[stage] = {
            "ready": not missing,
            "missing": missing,
            "optional": optional,
        }

    cuda = None
    if _installed("torch"):
        try:
            import torch

            cuda = {
                "available": bool(torch.cuda.is_available()),
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "torch": torch.__version__,
            }
        except Exception as exc:
            cuda = {"available": False, "device": None, "error": str(exc)}

    return {
        "version": __version__,
        "python": sys.version.split()[0],
        "python_check": python_check(),
        "platform": sys.platform,
        "prefix": sys.prefix,
        "stages": stages,
        "cuda": cuda,
        "dll_report": _winsetup.report(),
    }
