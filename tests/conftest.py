"""makes the agent's modules (assistant, planner, domain, ...) importable from the tests"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
