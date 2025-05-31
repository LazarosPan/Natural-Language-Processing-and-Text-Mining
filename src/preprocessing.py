from __future__ import annotations

"""
src/preprocessing.py  (2025-05-31)
──────────────────────────────────
✔ Text cleaning  (stop-word removal + Porter stemming)
✔ Sentence-Transformers embedding cache (PyTorch / CUDA)

Public helpers
--------------
clean_text(text)               → cleaned str
clean_and_stats(text)          → (clean, char_len, word_cnt)
clean_series(pd.Series)        → list[str]
build_st_embeddings(corpus, model_name, cache_dir) → np.ndarray
sentence_vector(sentence, st_matrix, id_map)       → np.ndarray
"""

import re, unicodedata, math, os, hashlib, numpy as np
from functools import lru_cache
from typing import List
from pathlib import Path
import torch

# ── 1. LIGHT CLEANING ───────────────────────────────────────────────
import nltk
try:
    _stop = nltk.corpus.stopwords.words("english")
except LookupError:
    nltk.download("stopwords"); nltk.download("punkt")
    _stop = nltk.corpus.stopwords.words("english")
STOPWORDS = set(_stop)
_STEMMER  = nltk.PorterStemmer()

_MATH_RE  = re.compile(r"\[math].*?\[/math]", re.I | re.S)
_HTML_RE  = re.compile(r"<[^>]+>")
_BAD_CHR  = re.compile(r"[^a-z0-9'? ]+")
_TOKEN_RE = re.compile(r"[a-z0-9']+|\?")

@lru_cache(maxsize=200_000)
def _stem(tok: str) -> str: return _STEMMER.stem(tok)

def _ascii_lower(s: str) -> str:
    return (unicodedata.normalize("NFKD", s)
            .encode("ascii", "ignore").decode("ascii").lower())

def clean_text(text: str | None) -> str:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    text = _ascii_lower(text)
    text = _MATH_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    text = _BAD_CHR.sub(" ", text)
    toks = _TOKEN_RE.findall(text)
    return " ".join(_stem(t) for t in toks if t not in STOPWORDS)

def clean_and_stats(text: str | None) -> tuple[str, int, int]:
    c = clean_text(text)
    return c, len(c), len(c.split())

def clean_series(series) -> List[str]:
    return [clean_text(x) for x in series.tolist()]

# ── 2. SENTENCE-TRANSFORMERS EMBEDDINGS ─────────────────────────────
from sentence_transformers import SentenceTransformer

def _model_cache_name(model_name: str) -> str:
    h = hashlib.sha1(model_name.encode()).hexdigest()[:8]
    return f"st_{h}.npy"

def build_st_embeddings(corpus: List[str],
                        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                        cache_dir: str = "models",
                        batch_size: int = 512) -> np.ndarray:
    """
    Encode each sentence in `corpus` into a 384-D (MiniLM) or 768-D (MPNet) vector.
    1st call downloads weights and writes <cache_dir>/st_<hash>.npy
    Later calls just mmap-load the file (read-only, zero-copy).
    Returns float32 ndarray shape (n_sent, dim).
    """
    cache = Path(cache_dir); cache.mkdir(exist_ok=True)
    fp = cache / _model_cache_name(model_name)
    if fp.exists():
        return np.load(fp, mmap_mode="r")
    model = SentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
    emb = model.encode(corpus,
                       batch_size=batch_size,
                       normalize_embeddings=True,
                       convert_to_numpy=True).astype("float32")
    np.save(fp, emb)
    return emb

def sentence_vector(sentence: str,
                    model: SentenceTransformer) -> np.ndarray:
    """
    Convenience one-off encoder (no caching).  
    Prefer `build_st_embeddings` for bulk processing.
    """
    return model.encode([sentence], normalize_embeddings=True,
                        convert_to_numpy=True)[0].astype("float32")

# ── SMOKE TEST ──────────────────────────────────────────────────────
if __name__ == "__main__":
    demo = "What is the step-by-step guide to invest in share market in India?"
    c, ln, wc = clean_and_stats(demo)
    print("CLEAN:", c, "| len", ln, "| words", wc)

    corpus = [c, clean_text("How to invest in US stock market?")]
    vecs   = build_st_embeddings(corpus, model_name="sentence-transformers/all-MiniLM-L6-v2",
                                 cache_dir="models_demo", batch_size=2)
    print("Embeddings:", vecs.shape)  # (2, 384)
