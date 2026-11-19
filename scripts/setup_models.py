#!/usr/bin/env python3
"""Clone the pinned SadTalker source and download its weights (same as `localavatar setup`)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from localavatar import models  # noqa: E402

models.fetch()
