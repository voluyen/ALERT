"""Command-line entry point for answergain.ssft.attribution."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from answergain.ssft.attribution import main  # noqa: E402

if __name__ == "__main__":
    main()
