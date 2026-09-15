"""
Turning what the user typed into a list of files.

The three original scripts each had their own version of this, one of them
recursive and one of them not. One implementation, one meaning of "recursive".
"""

from __future__ import annotations

from pathlib import Path

AUDIO_SUFFIXES = (".wav",)


def collect_audio(inputs, recursive: bool = False, suffixes=AUDIO_SUFFIXES) -> list[str]:
    """
    Expand a mixed list of files and folders into a sorted, de-duplicated list
    of audio paths. Missing entries are skipped rather than raising, so one bad
    line in the input list does not lose the rest of the batch.
    """
    found: list[Path] = []
    for item in inputs or []:
        text = str(item).strip().strip('"')
        if not text:
            continue
        path = Path(text).expanduser()
        if path.is_dir():
            pattern = "**/*" if recursive else "*"
            found += sorted(
                p for p in path.glob(pattern)
                if p.is_file() and p.suffix.lower() in suffixes
            )
        elif path.is_file() and path.suffix.lower() in suffixes:
            found.append(path)

    seen: set[str] = set()
    unique: list[str] = []
    for p in found:
        key = str(p.resolve()).lower()
        if key not in seen:
            seen.add(key)
            unique.append(str(p))
    return unique


def missing_inputs(inputs) -> list[str]:
    """Entries that do not exist, so the interface can say which line is wrong
    instead of reporting an empty batch."""
    out = []
    for item in inputs or []:
        text = str(item).strip().strip('"')
        if text and not Path(text).expanduser().exists():
            out.append(text)
    return out


def resolve_output_dir(configured: str, fallback: Path) -> Path:
    """An explicit output folder if one was set, otherwise beside the input."""
    text = str(configured or "").strip()
    if text:
        target = Path(text).expanduser()
        target.mkdir(parents=True, exist_ok=True)
        return target
    return fallback
