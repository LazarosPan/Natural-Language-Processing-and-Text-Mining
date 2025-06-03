# src/logs.py
# ---------------------------------------------------------------------
"""
Tiny helper for writing consistent comma-separated logging rows.

Usage
-----
from src.logs import log_event, LogKind

log_event(LogKind.FEATURES, model="PCA-600", phase="fit",
          seconds=12.8, extra="explained_var=0.952")

log_event(LogKind.MODEL, model="LightGBM", phase="train",
          seconds=3.42, src_dims=768, valid_LL=0.0741)

All logs live under  ./metric_logs/<kind>.csv
Each row is one line, no header needed – easy to `pandas.read_csv(...)`.
"""

from __future__ import annotations
import os
import time
from enum import Enum
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent     # project root
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
    Append one comma-separated line to metric_logs/<kind>.csv.

    Every line starts with: timestamp, caller, pid … then user-supplied key=value pairs.
    """
    # Use kind.value (e.g. "splits", "eda", etc.) rather than str(kind)
    fname = f"{kind.value}.csv"
    fp = LOG_DIR / fname
    
    # Ensure parent directory exists (although we created LOG_DIR above)
    fp.parent.mkdir(parents=True, exist_ok=True)
    
    with fp.open("a", encoding="utf8") as f:
        prefix = f"{_ts()},{_caller()},pid={os.getpid()}"
        body   = ",".join(f"{k}={v}" for k, v in kv.items())
        line = prefix + ("," + body if body else "") + "\n"
        f.write(line)