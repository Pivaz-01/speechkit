#!/usr/bin/env python3
"""
Open the interface. Equivalent to `python -m speechkit`, and here so the
project can be started from a file manager or a double-clickable shortcut
without anyone needing to know the module syntax.
"""

from speechkit.cli import main

raise SystemExit(main(["serve"]))
