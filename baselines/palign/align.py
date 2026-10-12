"""Command-line entry point for answergain.palign.align."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from answergain.palign.align import main  # noqa: E402

if __name__ == "__main__":
    main()
