from __future__ import annotations

"""src/preprocessing.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Utility functions for **text cleaning** and **Sentence‑Transformer embeddings**
used throughout the NLP similarity pipeline.

Improvements in this revision
-----------------------------
*  Robust *clean_text* pipeline (HTML / math tag removal, stop‑words, stemming).
*  **SUPPORTED_EMB** registry -> choose models by desired output dimension.
*  On‑disk cache keyed by SHA‑1 hash of the model name – avoids repeat encodes.
*  Centralised runtime logging via :pyfunc:`src.logs.log_event` (PREPROCESSING).
*  Returns char/word stats to align with most predictive signals found in
       01_eda (e.g. negative correlation of length features with duplicates).

Public API
~~~~~~~~~~
clean_text(text)                -> str
clean_and_stats(text)           -> (str, char_len, word_cnt)
clean_series(pd.Series)         -> list[str]
build_st_embeddings(texts, …)   -> np.ndarray  `(n, dim)`
"""

###############################################################################
# Standard library
###############################################################################
import hashlib
import math
import re
import time
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Sequence

###############################################################################
# Third‑party
###############################################################################
import numpy as np
import torch
import nltk
from sentence_transformers import SentenceTransformer  # type: ignore

###############################################################################
# Project
###############################################################################
from src.logs import log_event, LogKind

###############################################################################
# 1. TEXT‑CLEANING HELPERS
###############################################################################

# Ensure stop‑words are available (first run may need to download)
try:
    _STOP = set(nltk.corpus.stopwords.words("english"))
except LookupError:  # pragma: no cover – network I/O in CI skipped
    nltk.download("stopwords")
    _STOP = set(nltk.corpus.stopwords.words("english"))

_STEMMER = nltk.PorterStemmer()

_MATH_RE = re.compile(r"\[math].*?\[/math]", re.I | re.S)
_HTML_RE = re.compile(r"<[^>]+>")
_BAD_CHR = re.compile(r"[^a-z0-9'? ]+")
_TOKEN_RE = re.compile(r"[a-z0-9']+|\?")


def _ascii_lower(s: str) -> str:
    """Unicode -> ASCII (NFKD) then lowercase."""
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii").lower()


@lru_cache(maxsize=200_000)
def _stem(tok: str) -> str:  # pylint: disable=invalid-name
    return _STEMMER.stem(tok)


def clean_text(text: str | None) -> str:
    """Return a cleaned + stemmed version of *text* suitable for TF‑IDF / SBERT.

    Steps
    -----
    1. Handle *None* / *NaN*.
    2. Unicode normalisation + lowercase.
    3. Strip `[math]…[/math]` blocks and HTML tags.
    4. Replace all non‑alnum punctuation with spaces.
    5. Tokenise, drop stop‑words, apply Porter stemmer.
    """
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""

    # 1‑3) normalise + remove noise
    text = _ascii_lower(text)
    text = _MATH_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    text = _BAD_CHR.sub(" ", text)

    # 4‑5) tokenise + filter
    tokens = _TOKEN_RE.findall(text)
    tokens = [_stem(t) for t in tokens if t not in _STOP]
    return " ".join(tokens)


def clean_and_stats(text: str | None) -> tuple[str, int, int]:
    """Clean *text* and return `(cleaned, char_len, word_cnt)` as used in EDA."""
    cleaned = clean_text(text)
    return cleaned, len(cleaned), len(cleaned.split())


def clean_series(series) -> List[str]:  # type: ignore[valid-type]
    """Vectorised convenience wrapper for pandas Series -> list[str]."""
    return [clean_text(x) for x in series.tolist()]

###############################################################################
# 2. SENTENCE‑TRANSFORMER EMBEDDINGS
###############################################################################

# Recommended back‑bones keyed by output dimensionality
SUPPORTED_EMB: dict[int, str] = {
    384: "sentence-transformers/all-MiniLM-L6-v2",
    768: "sentence-transformers/all-mpnet-base-v2",  # quora‑friendly, strong quality
}


def _resolve_model(target_dim: int | None, model_name: str | None) -> str:
    if model_name is not None:
        return model_name
    if target_dim is None:
        raise ValueError("Either target_dim or model_name must be supplied")
    try:
        return SUPPORTED_EMB[target_dim]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported target_dim={target_dim}. Choose one of {list(SUPPORTED_EMB)}"
        ) from exc


def _cache_name(model_name: str) -> Path:
    """Return `<cache_dir>/st_<sha8>.npy` based on *model_name*."""
    h = hashlib.sha1(model_name.encode()).hexdigest()[:8]
    return Path(f"st_{h}.npy")


def build_st_embeddings(
    texts: Iterable[str],
    *,
    target_dim: int | None = 768,
    model_name: str | None = None,
    cache_dir: str | Path = "models",  # relative to repo root
    batch_size: int = 128,
    normalize: bool = True,
    save_path: str | Path | None = None,
) -> np.ndarray:
    """Encode *texts* into SBERT embeddings with transparent on‑disk caching.

    This wrapper **adds logging** and a dim->model registry on top of
    :pyclass:`sentence_transformers.SentenceTransformer`.
    """

    texts = list(texts)  # may be generator; we need len() twice
    model_name = _resolve_model(target_dim, model_name)

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_fp = cache_dir / _cache_name(model_name)

    if cache_fp.exists():
        emb = np.load(cache_fp, mmap_mode="r")
        # optional save_path copy for canonical location
        if save_path is not None and not Path(save_path).exists():
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            np.save(save_path, emb)
        return emb

    # -------------------------------------------------- encode (cache miss)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)

    start = time.perf_counter()
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
    ).astype(np.float32)
    elapsed = round(time.perf_counter() - start, 3)

    if target_dim is not None and emb.shape[1] != target_dim:
        raise ValueError(
            f"Model {model_name!r} produced {emb.shape[1]}‑d vectors; expected {target_dim}."
        )

    # -------------------------------------------------- save + log
    np.save(cache_fp, emb)
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, emb)

    log_event(
        LogKind.PREPROCESSING,
        model=model_name,
        dim=emb.shape[1],
        n=len(texts),
        seconds=elapsed,
        saved=str(save_path) if save_path else None,
    )

    return emb

###############################################################################
# 3. EXPORTS
###############################################################################
__all__ = [
    "SUPPORTED_EMB",
    "clean_text",
    "clean_and_stats",
    "clean_series",
    "build_st_embeddings",
]