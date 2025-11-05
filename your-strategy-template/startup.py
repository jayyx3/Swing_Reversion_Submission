#!/usr/bin/env python3
"""Bootstrap script for running the Swing Grid strategy template locally."""

from __future__ import annotations

import os
import sys

BASE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "base-bot-template")
if not os.path.exists(BASE_PATH):
    BASE_PATH = "/app/base"

sys.path.insert(0, BASE_PATH)

from cli import launch_from_environ  # type: ignore  # noqa: E402


def main() -> None:
    os.environ.setdefault("BOT_STRATEGY", "swing_grid")
    launch_from_environ()


if __name__ == "__main__":
    main()
