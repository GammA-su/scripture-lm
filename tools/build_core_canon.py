"""Offline corpus builder: uv run python tools/build_core_canon.py from the project root."""

from scripture_lm.corpus.build_core_canon import main

if __name__ == "__main__":
    raise SystemExit(main())
