# notebooks/setup.py

import sys
import pathlib
import os
import warnings
import logging

# -----------------------------------------------------------------------------
# Disable TensorFlow / Flax for HuggingFace (we only use PyTorch)
# -----------------------------------------------------------------------------
os.environ["USE_TF"]               = "0"   # transformers: disable TensorFlow
os.environ["TRANSFORMERS_NO_TF"]   = "1"   # same effect (legacy flag)
os.environ["TRANSFORMERS_NO_FLAX"] = "1"   # skip JAX / Flax
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # hide any residual TF C++ logs

# Optionally mute protobuf warning spam
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# -----------------------------------------------------------------------------
# Add project root and src/ to sys.path
# -----------------------------------------------------------------------------
project_root = pathlib.Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

SRC = project_root / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# -----------------------------------------------------------------------------
# Expose our logging helper so every notebook/script can call it
# -----------------------------------------------------------------------------
from src.logs import log_event, LogKind   # noqa: F401