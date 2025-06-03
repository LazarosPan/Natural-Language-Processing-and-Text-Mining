# ────────────────────────────────────────────────────────────────
# src/preprocessing.py
# ----------------------------------------------------------------
"""
• Porter‐stem + stop‐word cleaning
• Sentence‐Transformer bi‐encoder caching
  (supports multiple models; Quora‐fine‑tuned DistilBERT by default)
  Recommended models include:
    384‑dim  → sentence-transformers/all-MiniLM-L6-v2,
                sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2,
                sentence-transformers/all-MiniLM-L12-v2,
                BAAI/bge-small-en-v1.5
    768‑dim  → sentence-transformers/all-mpnet-base-v2,
                BAAI/bge-base-en-v1.5,
                cross-encoder/quora-roberta-base,
                cross-encoder/quora-distilroberta-base,
                sentence-transformers/distilbert-base-nli-stsb-quora-ranking

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
from typing import List, Sequence
from pathlib import Path

import numpy as np
import torch
import nltk

# ────────────────────────────────
# 1. TEXT‐CLEANING UTILITIES
# ────────────────────────────────

# Attempt to load NLTK English stopwords; if missing, download them.
try:
    _stop = nltk.corpus.stopwords.words("english")
except LookupError:
    nltk.download("stopwords")
    nltk.download("punkt")
    _stop = nltk.corpus.stopwords.words("english")

STOPWORDS = set(_stop)
_STEMMER  = nltk.PorterStemmer()

# Precompile regex patterns for cleaning
_MATH_RE   = re.compile(r"\[math].*?\[/math\]", re.IGNORECASE | re.DOTALL)
_HTML_RE   = re.compile(r"<[^>]+>")
_BAD_CHR   = re.compile(r"[^a-z0-9'? ]+")        # anything not a-z, 0-9, apostrophe, question mark, or space
_TOKEN_RE  = re.compile(r"[a-z0-9']+|\?")        # tokens: alphanumeric+apostrophes or standalone question marks

@lru_cache(maxsize=200_000)
def _stem(tok: str) -> str:
    """
    Apply Porter stemming to a single token, with LRU cache to speed up repeated tokens.
    """
    return _STEMMER.stem(tok)

def _ascii_lower(s: str) -> str:
    """
    Normalize Unicode → ASCII (NFKD), then lowercase.
    This removes accents/diacritics and ensures purely ASCII lowercase text.
    """
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii").lower()

def clean_text(text: str | None) -> str:
    """
    Clean a single question string:
      1. Handle None / NaN → return empty string
      2. Unicode normalize & lowercase
      3. Remove [math]...[/math] blocks & strip out HTML tags
      4. Replace all non-[a-z0-9'? ] characters with spaces
      5. Tokenize via regex (tokens are [a-z0-9']+ or '?')
      6. Drop NLTK English stopwords
      7. Porter‐stem each token (cached via lru_cache)
      8. Join tokens with single spaces into a cleaned string

    Returns:
      A cleaned, stemmed string where tokens are separated by single spaces.
    """
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""

    # 1) Normalize to ASCII lowercase
    text = _ascii_lower(text)

    # 2) Remove math blocks and HTML tags
    text = _MATH_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)

    # 3) Replace all “bad” characters with spaces
    text = _BAD_CHR.sub(" ", text)

    # 4) Tokenize and drop stopwords → stem
    tokens = _TOKEN_RE.findall(text)
    cleaned = [_stem(tok) for tok in tokens if tok not in STOPWORDS]

    return " ".join(cleaned)

def clean_and_stats(text: str | None) -> tuple[str, int, int]:
    """
    Clean a single question and return:
      ( cleaned_string, char_length_of_cleaned, word_count_of_cleaned )

    - cleaned_string: result of clean_text(text)
    - char_length_of_cleaned: len(cleaned_string)
    - word_count_of_cleaned: number of tokens in cleaned_string
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
    Deterministic cache filename (first 8 hex of SHA1) based on `model_name`.
    Example: "sentence-transformers/distilbert-base-nli-stsb-quora-ranking"
             → "st_ab12cd34.npy"
    """
    h = hashlib.sha1(model_name.encode()).hexdigest()[:8]
    return f"st_{h}.npy"

def build_st_embeddings(
    corpus      : List[str],
    model_name  : str | Sequence[str] = "sentence-transformers/distilbert-base-nli-stsb-quora-ranking",
    cache_dir   : str = "models",
    batch_size  : int = 512,
    out_fp      : str | None = None
) -> np.ndarray:
    """
    Encode each cleaned sentence in `corpus` into a dense Sentence-Transformer embedding.

    On first run:
      • Downloads `model_name` from Hugging Face
      • Encodes in batches (batch_size) on GPU if available
      • Normalizes embeddings to unit length (L2)
      • Saves resulting float32 array to disk: <cache_dir>/st_<sha8>.npy

    On subsequent runs:
      • Memory‐map loads (<cache_dir>/st_<sha8>.npy) to return a read‐only view.

    Optionally, with `out_fp` provided:
      • Also write the same embedding matrix (float32) to `out_fp`.
      This is useful for “canonical” downstream file locations
      (e.g., "data/processed/question_embeddings.npy").

    Arguments
    ---------
    corpus      : List[str]
                  List of cleaned question strings of length N_q.
    model_name  : str | Sequence[str]
                  Single model name or list of names. Each model is cached individually.
                  Default is Quora‐fine‐tuned DistilBERT.
    cache_dir   : str
                  Directory to cache the hashed embedding file.
    batch_size  : int
                  Batch size for SentenceTransformer.encode(…).
    out_fp      : str or None
                  If provided, also write embeddings to this exact filepath.

    Returns
    -------
    emb : np.ndarray, dtype=float32, shape = (N_q, embedding_dim)
          `embedding_dim` depends on the model(s): 768 for base models,
          384 for MiniLM variants, or the sum when multiple models are used.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # If user passed multiple models, build each then concatenate
    if isinstance(model_name, Sequence) and not isinstance(model_name, (str, bytes)):
        embeddings = [
            build_st_embeddings(corpus, m, cache_dir=cache_dir, batch_size=batch_size)
            for m in model_name
        ]
        emb = np.hstack(embeddings).astype("float32")
        if out_fp is not None:
            out_path = Path(out_fp)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, emb)
        return emb

    # Compute the hashed filename under cache_dir
    fp = cache_path / _cache_name(model_name)

    # If that file already exists, mmap-load and return it
    if fp.exists():
        return np.load(fp, mmap_mode="r")

    # Otherwise, we need to instantiate the SentenceTransformer and encode
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = SentenceTransformer(model_name, device=device)

    # Encode entire corpus in batches, normalize to unit length, return float32
    emb = model.encode(
        corpus,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    # Save under the hashed filename for future runs
    np.save(fp, emb)

    # If the user requested a canonical output path, write there as well
    if out_fp is not None:
        out_path = Path(out_fp)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, emb)

    return emb
