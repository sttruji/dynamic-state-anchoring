#!/usr/bin/env python3
"""Train the prepared single-device DSA pilot and persist resumable state."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dsa.training.pilot import main


if __name__ == "__main__":
    main()
