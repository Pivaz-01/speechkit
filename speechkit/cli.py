"""
Command line, for scripting and for running a stage without the interface.

    python -m speechkit                      open the interface
    python -m speechkit acoustics FOLDER -o OUT
    python -m speechkit phonemes FOLDER
    python -m speechkit align FOLDER
    python -m speechkit env                  what is installed
    python -m speechkit settings             print the current settings as JSON

Anything not given on the command line comes from the saved settings, so the
interface and the command line always agree.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__, environment, settings


def _apply(values, pairs):
    """--set NAME=VALUE, for the settings that have no dedicated flag."""
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--set needs NAME=VALUE, got {pair!r}")
        name, _, raw = pair.partition("=")
        name = name.strip().upper()
        if name not in settings.BY_NAME:
            raise SystemExit(f"unknown setting {name!r}. Try: python -m speechkit settings")
        try:
            values[name] = settings.coerce(name, raw)
        except settings.SettingError as exc:
            raise SystemExit(str(exc)) from None
    return values


def _require(stage: str) -> None:
    """Say which package is missing, instead of letting an ImportError from
    deep inside the module reach the user."""
    info = environment()["stages"][stage]
    if not info["ready"]:
        raise SystemExit(
            f"{stage} needs {', '.join(info['missing'])}.\n"
            f"  pip install {' '.join(info['missing'])}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="speechkit",
        description="Acoustic analysis, phoneme recognition and alignment for "
                    "speech recordings.")
    parser.add_argument("--version", action="version", version=f"speechkit {__version__}")
    sub = parser.add_subparsers(dest="command")

    serve_p = sub.add_parser("serve", help="open the interface (the default)")
    serve_p.add_argument("--host")
    serve_p.add_argument("--port", type=int)
    serve_p.add_argument("--no-browser", action="store_true")

    for name, help_text in (("acoustics", "run the Praat measures"),
                            ("phonemes", "recognise phonemes and write TSV files"),
                            ("align", "align phonemes to the passage text")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("inputs", nargs="*", help="folders or files")
        p.add_argument("-o", "--output", help="output folder")
        p.add_argument("-r", "--recursive", action="store_true")
        if name == "acoustics":
            p.add_argument("-t", "--task",
                           choices=["reading", "sustained_vowel", "auto"],
                           help="what is in the recordings; required unless already "
                                "set in the interface")
        p.add_argument("--set", action="append", metavar="NAME=VALUE",
                       help="override any setting; repeatable")

    sub.add_parser("env", help="report what is installed")
    sub.add_parser("settings", help="print the current settings as JSON")

    args = parser.parse_args(argv)
    command = args.command or "serve"

    if command == "env":
        json.dump(environment(), sys.stdout, indent=2)
        print()
        return 0

    if command == "settings":
        json.dump(settings.load(), sys.stdout, indent=2, ensure_ascii=False)
        print()
        return 0

    if command == "serve":
        from .web.app import serve

        serve(host=args.host, port=args.port,
              open_browser=False if args.no_browser else None)
        return 0

    values = settings.load()
    _apply(values, getattr(args, "set", None))

    if command == "acoustics":
        if args.inputs:
            values["WAV_INPUTS"] = list(args.inputs)
        if args.output:
            values["OUTPUT_FOLDER"] = args.output
        if args.recursive:
            values["RECURSIVE"] = True
        if args.task:
            values["TASK"] = args.task
        if not str(values.get("TASK", "")).strip():
            raise SystemExit(
                "no speech task set. Reading passages and sustained vowels are "
                "measured differently, so pick one:\n"
                "  --task reading\n"
                "  --task sustained_vowel\n"
                "  --task auto            (a folder holding both)")
        _require("acoustics")
        from .acoustics import run_acoustics

        try:
            run_acoustics(values)
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
        return 0

    if command == "phonemes":
        if args.inputs:
            values["PHONEME_INPUTS"] = list(args.inputs)
        if args.output:
            values["PHONEME_OUTPUT_DIR"] = args.output
        if args.recursive:
            values["PHONEME_RECURSIVE"] = True
        _require("phonemes")
        from .phonemes import run_phonemes

        run_phonemes(values, write_files=True, keep_audio=False)
        return 0

    if command == "align":
        if args.inputs:
            values["INPUT_DIRS"] = list(args.inputs)
        from .alignment import run_alignment

        try:
            run_alignment(values)
        except (ValueError, FileNotFoundError) as exc:
            raise SystemExit(str(exc)) from None
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
