# ────────────────────────────────────────────────────────────────
# src/preprocessing.py
# ----------------------------------------------------------------
"""
• Porter‐stem + stop‐word cleaning
• Sentence‐Transformer bi‐encoder caching
  (MiniLM‐L6‐v2 by default, GPU‐aware)

PUBLIC API
----------
clean_text(text)                → str
clean_and_stats(text)           → (str, char_len, word_cnt)
clean_series(pd.Series)         → list[str]
build_st_embeddings(corpus, …)  → np.ndarray   (mmap‐cached)
"""

from __future__ import annotations
import re
import unicodedata
import math
import hashlib
from functools import lru_cache
from typing import List
from pathlib import Path

import numpy as np
import torch
import nltk

# ────────────────────────────────
# 1. TEXT‐CLEANING UTILITIES
# ────────────────────────────────

# Attempt to load English stopwords; if missing, download.
try:
    _stop = nltk.corpus.stopwords.words("english")
except LookupError:
    nltk.download("stopwords")
    nltk.download("punkt")
    _stop = nltk.corpus.stopwords.words("english")

STOPWORDS = set(_stop)
_STEMMER  = nltk.PorterStemmer()

# Precompile regex patterns:
_MATH_RE   = re.compile(r"\[math].*?\[/math\]", re.IGNORECASE | re.DOTALL)
_HTML_RE   = re.compile(r"<[^>]+>")
_BAD_CHR   = re.compile(r"[^a-z0-9'? ]+")
_TOKEN_RE  = re.compile(r"[a-z0-9']+|\?")

@lru_cache(maxsize=200_000)
def _stem(tok: str) -> str:
    """
    Apply Porter stemming to a single token, with LRU cache to speed up repeated tokens.
    """
    return _STEMMER.stem(tok)

def _ascii_lower(s: str) -> str:
    """
    Normalize Unicode → ASCII (NFKD), then lowercase.
    """
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii").lower()

def clean_text(text: str | None) -> str:
    """
    Clean a single question string:
      1. Handle None / NaN → return empty string
      2. Unicode normalize & lowercase
      3. Remove [math]...[/math], strip out HTML tags
      4. Replace all non‐[a-z0-9'? ] characters with spaces
      5. Tokenize via regex ([a-z0-9']+ or '?')
      6. Drop NLTK English stopwords
      7. Porter‐stem each token (cached)
      8. Join with single spaces

    Returns:
        A cleaned, stemmed string (tokens joined by spaces).
    """
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""

    # 1) normalize → lowercase
    text = _ascii_lower(text)

    # 2) strip math blocks & HTML
    text = _MATH_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)

    # 3) remove bad characters (punctuation except apostrophes, etc.)
    text = _BAD_CHR.sub(" ", text)

    # 4) find tokens and drop stopwords → stem
    tokens = _TOKEN_RE.findall(text)
    cleaned = [_stem(tok) for tok in tokens if tok not in STOPWORDS]

    return " ".join(cleaned)

def clean_and_stats(text: str | None) -> tuple[str, int, int]:
    """
    Clean a single question and return:
      ( cleaned_string, char_length_of_cleaned, word_count_of_cleaned )
    """
    c = clean_text(text)
    return c, len(c), len(c.split())

def clean_series(series) -> List[str]:
    """
    Vectorised wrapper: given a pandas Series of raw question strings,
    return a Python list of cleaned strings (using clean_text).
    """
    return [clean_text(x) for x in series.tolist()]


# ────────────────────────────────
# 2. BI‐ENCODER EMBEDDINGS CACHE
# ────────────────────────────────

def _cache_name(model_name: str) -> str:
    """
    Deterministic cache filename (8 hex chars) from the model_name.
    """
    h = hashlib.sha1(model_name.encode()).hexdigest()[:8]
    return f"st_{h}.npy"

def build_st_embeddings(
    corpus      : List[str],
    model_name  : str = "sentence-transformers/all-MiniLM-L6-v2",
    cache_dir   : str = "models",
    batch_size  : int = 512
) -> np.ndarray:
    """
    Encode each cleaned sentence in `corpus` into a dense Sentence-Transformer embedding.

    On first run:
      • Downloads `model_name` from Hugging Face
      • Encodes in batches on GPU (if available), normalizes to unit length
      • Saves resulting float32 array to disk: <cache_dir>/st_<sha8>.npy

    On subsequent runs:
      • Memory‐map loads (<cache_dir>/st_<sha8>.npy) to return a read‐only view.

    Returns:
      NumPy float32 array of shape (n_sentences, embedding_dim).
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)

    fp = cache_path / _cache_name(model_name)
    if fp.exists():
        # mmap load for zero‐copy
        return np.load(fp, mmap_mode="r")

    # Otherwise, instantiate a SentenceTransformer and encode
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = SentenceTransformer(model_name, device=device)

    emb = model.encode(
        corpus,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    np.save(fp, emb)
    return emb