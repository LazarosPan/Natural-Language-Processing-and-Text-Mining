# src/logs.py
# ---------------------------------------------------------------------
"""
Tiny helper for writing consistent comma-separated logging rows.

Usage
-----
from src.logs import log_event, LogKind

# default writes to metric_logs/benchmarks.csv
log_event(LogKind.TEST, model="MyModel", phase="eval", seconds=1.23)
# to write to a folder-specific CSV, pass folder="lr0.05_ep10"
log_event(LogKind.TEST, folder="lr0.05_ep10", model="MyModel", phase="eval", seconds=1.23)

All logs live under  ./metric_logs/<kind>(_<folder>).csv
Each line is one line, no header needed – easy to `pandas.read_csv(...)`.
"""

from __future__ import annotations
import os
import time
from enum import Enum
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent     # project root
LOG_DIR = ROOT / "metric_logs"
LOG_DIR.mkdir(exist_ok=True)

class LogKind(str, Enum):
    SPLIT           = "splits"          # 00_split
    EDA             = "eda"             # 01_eda
    PREPROCESSING   = "preprocessing"   # 02_preprocessing
    FEATURES        = "features"        # 03_feature_engineering
    MODEL           = "models"          # 04_models  (single expert or gate)
    GATE            = "gates"           # subset gate tuning
    TEST            = "benchmarks"      # 05_benchmarks


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def _caller() -> str:
    # name of the top-level script / notebook (best-effort)
    import inspect, pathlib
    frame = inspect.stack()[-1]
    file  = pathlib.Path(frame.filename).name
    return file


def log_event(kind: LogKind, **kv):
    """
    Append one comma-separated line to metric_logs/<kind>(_folder).csv.

    Pass an optional 'folder' kwarg to write to e.g. benchmarks_<folder>.csv
    """
    # extract optional folder suffix
    folder = kv.pop('folder', None)
    # build filename
    fname = kind.value + (f"_{folder}" if folder else "") + ".csv"
    fp = LOG_DIR / fname

    # Ensure parent directory exists
    fp.parent.mkdir(parents=True, exist_ok=True)

    with fp.open("a", encoding="utf8") as f:
        prefix = f"{_ts()},{_caller()},pid={os.getpid()}"
        body   = ",".join(f"{k}={v}" for k, v in kv.items())
        line = prefix + ("," + body if body else "") + "\n"

        f.write(line)