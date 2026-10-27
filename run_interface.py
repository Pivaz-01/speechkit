#!/usr/bin/env python3
"""
Open the interface. Equivalent to `python -m speechkit`, and here so the
project can be started from a file manager or a double-clickable shortcut
without anyone needing to know the module syntax.
"""

from speechkit.cli import main

# The guard matters: the DDK stage runs its trials in worker processes, and on
# Windows and macOS each worker re-imports this file. Without it, every worker
# would start another copy of the interface.
if __name__ == "__main__":
    raise SystemExit(main(["serve"]))
