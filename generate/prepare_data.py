"""Command-line entry point for answergain.data.prepare."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from answergain.data.prepare import main  # noqa: E402

if __name__ == "__main__":
    main()
