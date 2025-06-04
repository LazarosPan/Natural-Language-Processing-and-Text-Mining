# ────────────────────────────────────────────────────────────────
# src/features.py
# ----------------------------------------------------------------
"""
Builds per‐pair features for Quora duplicate‐question detection, then applies
optional dimensionality reduction (IncrementalPCA / UMAP). All “fit” and “load”
events are logged via src.logs.log_event(LogKind.FEATURES, …).

Key blocks:
  A) TF-IDF cosine similarities (word‐ and char‐level)         -> 2 cols
  B) SVD projections on TF-IDF (150 & 100 dims per question)  -> 500 cols
  C) SBERT‐dense pair vectors (u, v, |u−v|, u⋅v)               -> 4*dim cols
  D) RapidFuzz fuzzy & length‐difference features             -> 4 cols
  E) Jaccard token overlap                                    -> 1 col
  F) Numeric “magic” features (lengths, len‐diff bins, freq)   -> 18 cols
  G) Cross‐Encoder duplicate‐probability                       -> 1 col
  ------------------------------------------------------------------
  Raw dim = 2 + 500 + (4*dim) + 4 + 1 + 18 + 1 = 526 + 4*dim

After building X_raw, reduction options:
  • ipca: chunked two‐pass IncrementalPCA -> either k95 (95% var) or given n_components  
  • umap: first chunked IPCA->k95, then UMAP->n_components  

All intermediate pickles live under cache_dir. Logging goes to metric_logs/features.csv.
"""
from __future__ import annotations
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"                   # suppress TF/Abseil INFO
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import time
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD, IncrementalPCA

from src.logs import log_event, LogKind

# Optional UMAP
try:
    import umap.umap_ as umap  # type: ignore
except ImportError:
    umap = None

# ─────────────────────────────────────────────────────────────────────────────
# A. JACCARD HELPER (token‐level)
# ─────────────────────────────────────────────────────────────────────────────
import re

_RE_TOK    = re.compile(r"[A-Za-z0-9']+")
_STOPWORDS = set(ENGLISH_STOP_WORDS)

def _jaccard(a: str, b: str) -> float:
    """
    Compute Jaccard overlap of token sets (excluding stopwords).
    """
    s1 = {t for t in _RE_TOK.findall(a.lower()) if t not in _STOPWORDS}
    s2 = {t for t in _RE_TOK.findall(b.lower()) if t not in _STOPWORDS}
    return len(s1 & s2) / (len(s1 | s2) or 1)

# ─────────────────────────────────────────────────────────────────────────────
# B. TF-IDF COSINES (word‐level & char‐level)
# ─────────────────────────────────────────────────────────────────────────────
def _fit_vecs(
    corpus: List[str],
    cache: Path,
    *,
    allow_fit: bool = True,
) -> Tuple[TfidfVectorizer, TfidfVectorizer]:
    """
    Fit or load two TfidfVectorizer objects:
      - vec_w: word n-grams (1–2), min_df=3, sublinear_tf=True
      - vec_c: char n-grams (3–5), min_df=10

    Cache files: cache/tfidf_w.pkl and cache/tfidf_c.pkl.
    """
    cache.mkdir(parents=True, exist_ok=True)
    fp_w = cache / "tfidf_w.pkl"
    fp_c = cache / "tfidf_c.pkl"

    if fp_w.exists() and fp_c.exists():
        vec_w = joblib.load(fp_w)
        vec_c = joblib.load(fp_c)
        log_event(LogKind.FEATURES, model="TFIDF", phase="load")
        return vec_w, vec_c

    if not allow_fit:
        raise RuntimeError("TF-IDF pickles not found; run TRAIN split first.")

    start = time.time()
    vec_w = TfidfVectorizer(ngram_range=(1,2), min_df=3, sublinear_tf=True)
    vec_c = TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=10)
    vec_w.fit(corpus)
    vec_c.fit(corpus)
    joblib.dump(vec_w, fp_w)
    joblib.dump(vec_c, fp_c)
    elapsed = round(time.time() - start, 2)
    log_event(LogKind.FEATURES, model="TFIDF", phase="fit", seconds=elapsed)
    return vec_w, vec_c

def _cosine_rows(X: sp.csr_matrix, idx1: np.ndarray, idx2: np.ndarray) -> np.ndarray:
    """
    Given CSR matrix X, compute cosine similarity between rows idx1[i] and idx2[i].
    Returns float32 array of length n_pairs.
    """
    Xn   = normalize(X, axis=1)
    sims = Xn[idx1].multiply(Xn[idx2]).sum(axis=1)
    return np.asarray(sims).ravel().astype("float32")

# ─────────────────────────────────────────────────────────────────────────────
# C. SVD DIMENSIONALITY REDUCTION
# ─────────────────────────────────────────────────────────────────────────────
def _svd_projection(
    X: sp.csr_matrix,
    n_comp: int,
    cache: Path
) -> np.ndarray:
    """
    Load or fit TruncatedSVD(n_components=n_comp) on X. Cache ->
    cache/svd_w_150.pkl or cache/svd_c_100.pkl. Returns float32 matrix.
    """
    cache.mkdir(parents=True, exist_ok=True)
    if n_comp == 150:
        fp_model = cache / "svd_w_150.pkl"
        model_label = "SVD-150"
    elif n_comp == 100:
        fp_model = cache / "svd_c_100.pkl"
        model_label = "SVD-100"
    else:
        raise ValueError(f"Unsupported n_comp={n_comp} for SVD caching")

    if fp_model.exists():
        svd = joblib.load(fp_model)
        Z   = svd.transform(X).astype("float32")
        log_event(LogKind.FEATURES, model=model_label, phase="load")
        return Z

    start = time.time()
    svd = TruncatedSVD(n_components=n_comp, random_state=None).fit(X)
    joblib.dump(svd, fp_model)
    Z = svd.transform(X).astype("float32")
    elapsed = round(time.time() - start, 2)
    log_event(LogKind.FEATURES, model=model_label, phase="fit", seconds=elapsed)
    return Z

# ─────────────────────────────────────────────────────────────────────────────
# D. SBERT‐DENSE PAIR VECTOR
# ─────────────────────────────────────────────────────────────────────────────
def _dense_pair(emb: np.ndarray, idx1: np.ndarray, idx2: np.ndarray) -> np.ndarray:
    """
    Given SBERT embeddings (n_q × dim), return [u, v, |u−v|, u*v] (n_pairs × 4*dim).
    """
    u = emb[idx1]
    v = emb[idx2]
    return np.hstack([u, v, np.abs(u - v), u * v]).astype("float32")

# ─────────────────────────────────────────────────────────────────────────────
# E. FUZZY / LENGTH‐DIFF BLOCK
# ─────────────────────────────────────────────────────────────────────────────
from rapidfuzz.fuzz import ratio, token_set_ratio

def _fuzzy_block(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    For each pair of raw strings, compute:
      - ratio(a,b)/100
      - token_set_ratio(a,b)/100
      - |char_len_diff|
      - |word_count_diff|
    Returns (n_pairs × 4) float32.
    """
    n   = len(q1)
    out = np.zeros((n, 4), dtype="float32")
    for i, (a, b) in enumerate(zip(q1, q2)):
        out[i, 0] = ratio(a, b) / 100.0
        out[i, 1] = token_set_ratio(a, b) / 100.0
        out[i, 2] = abs(len(a) - len(b))
        out[i, 3] = abs(len(a.split()) - len(b.split()))
    return out

# ─────────────────────────────────────────────────────────────────────────────
# F. NUMERIC “MAGIC” FEATURES
# ─────────────────────────────────────────────────────────────────────────────
def _numeric(
    meta: pd.DataFrame,
    idx1: np.ndarray,
    idx2: np.ndarray,
    freq_arr: np.ndarray
) -> np.ndarray:
    """
    Build numeric features per pair:
      - char-length q1, char-length q2
      - |char_len_diff| -> one-hot (10 bins)
      - min frequency -> log1p + quartile one-hot (4)
    Returns (n_pairs × 18) float32.
    """
    # Raw char lengths
    len_q1 = meta["len"].values[idx1].astype("float32")[:, None]
    len_q2 = meta["len"].values[idx2].astype("float32")[:, None]

    # Absolute char-length diff
    ld = np.abs(len_q1.squeeze() - len_q2.squeeze()).astype("float32")
    len_diff = ld[:, None]

    # 10-bin one-hot for len_diff
    bins = pd.cut(ld, [0,2,4,6,8,11,16,22,31,50,1e6],
                  labels=False, include_lowest=True)
    bins = np.asarray(bins, dtype="float32")
    bins_filled = np.where(np.isnan(bins), 0, bins).astype(int)
    len_oh = np.eye(10, dtype="float32")[bins_filled]

    # Min-freq log1p + quartile one-hot
    mf     = np.minimum(freq_arr[idx1], freq_arr[idx2]).astype("float32")
    mf_log = np.log1p(mf)[:, None]

    quart   = pd.qcut(mf, 4, labels=False, duplicates="drop")
    quart   = np.asarray(quart, dtype="float32")
    quart_f = np.where(np.isnan(quart), 0, quart).astype(int)
    mf_oh   = np.eye(4, dtype="float32")[quart_f]

    return np.hstack([len_q1, len_q2, len_diff, len_oh, mf_log, mf_oh])

# ─────────────────────────────────────────────────────────────────────────────
# G. CROSS‐ENCODER PROBABILITY BLOCK
# ─────────────────────────────────────────────────────────────────────────────
import time
from sentence_transformers import CrossEncoder  # type: ignore

_CROSS_CACHE: dict[str, CrossEncoder] = {}

def _cross_scores(
    pair_df     : pd.DataFrame,
    model_name  : str = "cross-encoder/quora-roberta-large",
    batch_size  : int = 64,
    cache_file  : str | None = None
) -> np.ndarray:
    """
    Compute (or load) CrossEncoder duplicate-prob scores for each (q1, q2).
    If cache_file exists, load & return. Otherwise:
      • Instantiate CrossEncoder(model_name) with Sigmoid
      • Retry up to 10 times on ConnectionError/OSError
      • Predict scores in batches
      • Save to cache_file if provided
    Returns float32 array (n_pairs,).
    """
    # 1) Load from cache if available
    if cache_file and Path(cache_file).exists():
        arr = np.load(cache_file, mmap_mode="r").astype("float32")
        log_event(LogKind.FEATURES, model=f"CROSS:{model_name}", phase="load")
        return arr

    # 2) Instantiate with retries
    if model_name not in _CROSS_CACHE:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        last_exc: Exception | None = None
        for attempt in range(10):
            try:
                _CROSS_CACHE[model_name] = CrossEncoder(
                    model_name,
                    device=dev,
                    activation_fn=torch.nn.Sigmoid()
                )
                last_exc = None
                break
            except Exception as e:
                if isinstance(e, (ConnectionError, OSError)) and attempt < 9:
                    last_exc = e
                    time.sleep(2)
                    continue
                raise
        if last_exc is not None:
            raise last_exc
        log_event(LogKind.FEATURES, model=f"CROSS:{model_name}", phase="fit")

    # 3) Predict with retries
    model = _CROSS_CACHE[model_name]
    tuples = list(zip(pair_df.question1.values, pair_df.question2.values))
    last_exc = None
    for attempt in range(10):
        try:
            scores = model.predict(tuples, batch_size=batch_size).astype("float32")
            last_exc = None
            break
        except Exception as e:
            if isinstance(e, (ConnectionError, OSError)) and attempt < 9:
                last_exc = e
                time.sleep(2)
                continue
            raise
    if last_exc is not None:
        raise last_exc

    # 4) Save if requested
    if cache_file:
        np.save(cache_file, scores)
    return scores

# ─────────────────────────────────────────────────────────────────────────────
# Helper: Two‐pass chunked IPCA for 95% variance
# ─────────────────────────────────────────────────────────────────────────────
def _incremental_pca_95(
    raw_fp: Path,
    cache_dir: Path,
    chunk_size: int = 5_000
) -> Tuple[IncrementalPCA, int]:
    """
    Perform a two‐pass IncrementalPCA on X_raw in chunks to discover k95 and fit.
    raw_fp:     Path to .npy of X_raw (shape=(n_pairs, D))
    cache_dir:  where to dump pca_95.pkl afterward
    chunk_size: number of rows to load at once (adjust to fit memory)
    """
    X_all = np.load(raw_fp, mmap_mode="r")  # shape (n_rows, D)
    n_rows, _ = X_all.shape

    # First pass: partial_fit to compute explained_variance_ratio_
    ipca_full = IncrementalPCA(n_components=None)
    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)
        ipca_full.partial_fit(X_all[start:end])
    cumvar = np.cumsum(ipca_full.explained_variance_ratio_)
    k95 = int(np.searchsorted(cumvar, 0.95)) + 1

    # Second pass: fit exactly k95 components
    ipca = IncrementalPCA(n_components=k95)
    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)
        ipca.partial_fit(X_all[start:end])

    joblib.dump(ipca, cache_dir / "pca_95.pkl")
    return ipca, k95

# ─────────────────────────────────────────────────────────────────────────────
# H. MASTER BUILDER + DIMENSIONALITY REDUCTION
# ─────────────────────────────────────────────────────────────────────────────
def build_features(
    pair_df         : pd.DataFrame,
    clean_questions : List[str],
    meta_df         : pd.DataFrame,
    embedding_path  : str = "../data/processed/question_embeddings_768.npy",
    cache_dir       : str = "../models/features",
    cross_cache     : str | None = "../data/processed/train_cross_scores.npy",
    fit_pca         : bool = False,
    features_cache  : str | None = None,
    reduction       : str = "ipca",
    n_components    : int | None = None,
) -> np.ndarray:
    """
    Build raw feature matrix (n_pairs × (526 + 4*dim)), then reduce to (n_pairs × n_components).

    Args
    ----
    pair_df: DataFrame with columns ["question1","question2"].
    clean_questions: List[str] of all unique questions (same order as meta_df["question"]).
    meta_df: DataFrame with ["question","len","words"], aligned with clean_questions.
    embedding_path: Path to SBERT embeddings (n_questions × dim). dim ∈ {384,768}.
    cache_dir: Directory to cache TF-IDF/SVD/IPCA/UMAP pickles.
    cross_cache: Optional .npy path for cross‐encoder scores (train split).
    fit_pca: True -> fit new IPCA/UMAP; False -> load existing pickles.
    features_cache: Optional .npy path to save/load raw X_raw.
    reduction: "ipca" | "umap".
    n_components: Target dims after reduction. None -> 95% var for IPCA; must be set for UMAP.

    Returns
    -------
    X_red: np.ndarray of shape (n_pairs, n_components) if reduction="umap", or (n_pairs, k95) if reduction="ipca".
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    # 1) Map question text -> index
    question_to_pos: dict[str, int] = {
        q: i for i, q in enumerate(meta_df["question"].astype(str).tolist())
    }
    try:
        idx1 = np.array([question_to_pos[q] for q in pair_df.question1.values], dtype="int32")
        idx2 = np.array([question_to_pos[q] for q in pair_df.question2.values], dtype="int32")
    except KeyError as ke:
        raise KeyError(f"Question not found in meta_df: {ke}")

    # 2) Load or build raw features
    if features_cache and (not fit_pca) and Path(features_cache).exists():
        X_raw = np.load(features_cache, mmap_mode="r")
        log_event(LogKind.FEATURES, model="RAW_FEATURES", phase="load")
    else:
        # Build raw feature blocks from scratch
        # A) TF-IDF Cosines
        vec_w, vec_c = _fit_vecs(clean_questions, cache, allow_fit=fit_pca)
        Xw = vec_w.transform(clean_questions)  # (n_q, vocab_w)
        Xc = vec_c.transform(clean_questions)  # (n_q, vocab_c)

        cos_w = _cosine_rows(Xw, idx1, idx2)  # (n_pairs,)
        cos_c = _cosine_rows(Xc, idx1, idx2)  # (n_pairs,)
        cos_block = np.vstack([cos_w, cos_c]).T.astype("float32")  # (n_pairs, 2)

        # B) SVD Projections
        Zw = _svd_projection(Xw, n_comp=150, cache=cache)  # (n_q, 150)
        Zc = _svd_projection(Xc, n_comp=100, cache=cache)  # (n_q, 100)

        svd_left   = Zw[idx1]
        svd_right  = Zw[idx2]
        svd_word   = np.hstack([svd_left, svd_right]).astype("float32")  # (n_pairs, 300)

        svd_left_c  = Zc[idx1]
        svd_right_c = Zc[idx2]
        svd_char    = np.hstack([svd_left_c, svd_right_c]).astype("float32")  # (n_pairs, 200)

        svd_block = np.hstack([svd_word, svd_char]).astype("float32")  # (n_pairs, 500)

        # C) SBERT Dense Pair
        emb = np.load(embedding_path, mmap_mode="r")  # (n_q, dim)
        dense_block = _dense_pair(emb, idx1, idx2)     # (n_pairs, 4*dim)

        # D) Fuzzy / Length-Diff
        fuzzy_block = _fuzzy_block(
            pair_df.question1.values,
            pair_df.question2.values
        )  # (n_pairs, 4)

        # E) Jaccard
        jacc_block = np.fromiter(
            (_jaccard(a, b) for a, b in zip(pair_df.question1, pair_df.question2)),
            dtype="float32"
        ).reshape(-1, 1)  # (n_pairs, 1)

        # F) Numeric “Magic” Features
        all_qs = np.concatenate([pair_df.question1.values, pair_df.question2.values])
        counts = pd.Series(all_qs).value_counts().to_dict()
        freq_arr = np.array([counts.get(q, 0) for q in meta_df["question"].values], dtype="float32")

        numeric_block = _numeric(meta_df, idx1, idx2, freq_arr)  # (n_pairs, 18)

        # G) Cross-Encoder Scores
        cross_block = _cross_scores(
            pair_df,
            model_name="cross-encoder/quora-roberta-large",
            batch_size=64,
            cache_file=cross_cache
        ).reshape(-1, 1)  # (n_pairs, 1)

        # H) Concatenate all raw blocks
        X_raw = np.hstack([
            cos_block,      # 2
            dense_block,    # 4*dim
            fuzzy_block,    # 4
            jacc_block,     # 1
            numeric_block,  # 18
            cross_block,    # 1
            svd_block       # 500
        ]).astype("float32")  # -> (n_pairs, 526 + 4*dim)

        if features_cache:
            np.save(features_cache, X_raw)
            log_event(LogKind.FEATURES, model="RAW_FEATURES", phase="fit", seconds=0.0)

    # 3) Dimensionality reduction
    red = reduction.lower()

    # ─────────────────────────────────────────────────────────────────────
    # IPCA path (chunked two‐pass for 95% variance)
    # ─────────────────────────────────────────────────────────────────────
    if red == "ipca":
        # If n_components=None, we auto‐detect k95; otherwise we fix to given n_components.
        if n_components is None:
            pca_fp = cache / "pca_95.pkl"
            model_label = "IPCA-95"
        else:
            pca_fp = cache / f"ipca_{n_components}.pkl"
            model_label = f"IPCA-{n_components}"

        # If we need to (re)fit:
        if fit_pca or not pca_fp.exists():
            start = time.time()
            if n_components is None:
                # Ensure raw features are on disk:
                if not features_cache or not Path(features_cache).exists():
                    raise RuntimeError("Raw features must be saved before running chunked IPCA.")
                ipca, k95 = _incremental_pca_95(
                    raw_fp    = Path(features_cache),
                    cache_dir = cache,
                    chunk_size=5_000
                )
            else:
                # Fit exactly n_components via partial_fit in chunks
                raw_arr = np.load(features_cache, mmap_mode="r")
                n_rows, _ = raw_arr.shape
                ipca = IncrementalPCA(n_components=n_components)
                for start_row in range(0, n_rows, 5_000):
                    end_row = min(start_row + 5_000, n_rows)
                    ipca.partial_fit(raw_arr[start_row:end_row])

            # Transform raw in chunks
            raw_arr = np.load(features_cache, mmap_mode="r")
            n_rows, _ = raw_arr.shape
            X_red = np.zeros((n_rows, ipca.n_components_), dtype="float32")
            for start_row in range(0, n_rows, 5_000):
                end_row = min(start_row + 5_000, n_rows)
                X_red[start_row:end_row] = ipca.transform(raw_arr[start_row:end_row]).astype("float32")

            joblib.dump(ipca, pca_fp)
            elapsed = round(time.time() - start, 2)
            log_event(
                LogKind.FEATURES,
                model=model_label,
                phase="fit",
                seconds=elapsed,
                n_components=ipca.n_components_,
            )
            return X_red

        # Else load existing IPCA and transform X_raw directly:
        ipca = joblib.load(pca_fp)
        X_red = ipca.transform(X_raw).astype("float32")
        log_event(
            LogKind.FEATURES,
            model=model_label,
            phase="load",
            n_components=ipca.n_components_,
        )
        return X_red

    # ─────────────────────────────────────────────────────────────────────
    # UMAP path (chunked IPCA->k95 first, then UMAP->n_components)
    # ─────────────────────────────────────────────────────────────────────
    if red == "umap":
        if umap is None:
            raise RuntimeError("umap-learn is required for reduction='umap'")
        if n_components is None:
            raise ValueError("n_components must be specified for UMAP")

        umap_fp = cache / f"umap_{n_components}.pkl"
        pca_fp  = cache / "pca_95.pkl"
        model_label = f"UMAP-{n_components}"

        # 1) If fitting from scratch (TRAIN or no saved pickles yet):
        if fit_pca or (not pca_fp.exists()) or (not umap_fp.exists()):
            start = time.time()

            # (a) Chunked IPCA->k95 (saving pca_95.pkl)
            if not pca_fp.exists():
                if not features_cache or not Path(features_cache).exists():
                    raise RuntimeError("Raw features must be saved before running chunked IPCA.")
                ipca, k95 = _incremental_pca_95(
                    raw_fp    = Path(features_cache),
                    cache_dir = cache,
                    chunk_size=5_000
                )
            else:
                ipca = joblib.load(pca_fp)
                k95 = ipca.n_components_

            # (b) Build full X_ipca (n_pairs × k95) in memory
            raw_arr = np.load(features_cache, mmap_mode="r")
            n_rows, _ = raw_arr.shape
            X_ipca = np.zeros((n_rows, k95), dtype="float32")
            for start_row in range(0, n_rows, 5_000):
                end_row = min(start_row + 5_000, n_rows)
                X_ipca[start_row:end_row] = ipca.transform(raw_arr[start_row:end_row]).astype("float32")

            # (c) Fit UMAP on X_ipca
            reducer = umap.UMAP(
                n_components   = n_components,
                n_neighbors    = 50,
                random_state   = None,
                n_jobs         = -1
            )
            X_red = reducer.fit_transform(X_ipca).astype("float32")
            joblib.dump(reducer, umap_fp)

            elapsed = round(time.time() - start, 2)
            log_event(
                LogKind.FEATURES,
                model=model_label,
                phase="fit",
                seconds=elapsed,
                n_components=n_components,
            )
            return X_red

        # 2) Otherwise (load existing pickles):
        #   (a) Load saved IPCA (pca_95.pkl) and chunk-transform raw -> X_ipca
        ipca = joblib.load(pca_fp)
        k95 = ipca.n_components_
        raw_arr = np.load(features_cache, mmap_mode="r")
        n_rows, _ = raw_arr.shape
        X_ipca = np.zeros((n_rows, k95), dtype="float32")
        for start_row in range(0, n_rows, 5_000):
            end_row = min(start_row + 5_000, n_rows)
            X_ipca[start_row:end_row] = ipca.transform(raw_arr[start_row:end_row]).astype("float32")

        #   (b) Load UMAP pickled model, then transform
        reducer = joblib.load(umap_fp)
        X_red = reducer.transform(X_ipca).astype("float32")
        log_event(
            LogKind.FEATURES,
            model=model_label,
            phase="load",
            n_components=n_components,
        )
        return X_red

    raise ValueError(f"Unknown reduction='{reduction}'")