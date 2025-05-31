# notebooks/setup.py
import sys, pathlib

# --- PyTorch-only, no TF/JAX -------------------------------------------------
import os, warnings, logging
os.environ["USE_TF"]              = "0"   # transformers: disable TensorFlow
os.environ["TRANSFORMERS_NO_TF"]  = "1"   # same effect (legacy flag)
os.environ["TRANSFORMERS_NO_FLAX"] = "1"  # skip JAX / Flax
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # hide any residual TF C++ logs

# Optionally mute protobuf warning spam
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# Project root file path
project_root = pathlib.Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))