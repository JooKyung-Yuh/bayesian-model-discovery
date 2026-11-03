import os
import sys

sys.dont_write_bytecode = True
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def pytest_configure(config):
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("run the recovery-judge tests with PYTHONHASHSEED=0 (sympy's set order must be fixed)")
