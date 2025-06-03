# ────────────────────────────────────────────────────────────────
# src/features.py
# ----------------------------------------------------------------
"""
Builds the full feature matrix for Quora Duplicate Question pairs,
then applies an IncrementalPCA step to reduce dimensionality while retaining
95% of the variance.

BLOCKS
  A  2 × TF-IDF cosine (word & char)                               -> 2   cols
  B  Sentence-embedding pair-vector (u, v, |u−v|, u·v).
     Embedding dim may be 768 (e.g. DistilBERT/MPNet) or 384 for MiniLM models.
     The block size is 4×dim.
  C  RapidFuzz lexical + length diffs                              -> 4   cols
  D  Jaccard token overlap                                         -> 1   col
  E  Numeric “magic” features (len_q1, len_q2, len_diff buckets, freq) -> 18 cols
  F  Quora Cross-Encoder duplicate probability                      -> 1   col
  G  TF-IDF SVD reduction (150 & 100 dims per side)                 -> 500 cols

  -> raw concatenated feature‐vector has:
       2 + (4×dim) + 4 + 1 + 18 + 1 + 500 = 526 + 4×dim dimensions.

After assembling this high-dimensional matrix, we apply **IncrementalPCA** in two passes:
  1. In TRAIN mode (fit_pca=True): 
       -  first pass with IncrementalPCA(n_components=None) to compute explained_variance_ratio_
       -  derive `k95` = #components needed for 95% variance
       -  second pass with IncrementalPCA(n_components=k95) to fit_transform X_raw -> X_red
       -  save the fitted IPCA model to cache_dir/pca_95.pkl
  2. In VALID/TEST mode (fit_pca=False):
       -  load `cache_dir/pca_95.pkl` and just transform any new X_raw 

Total return dimension: (n_pairs, k95)  with k95 ≪ 526 + 4×dim.
"""

from __future__ import annotations
import re, joblib, numpy as np, pandas as pd, scipy.sparse as sp
from pathlib import Path
from typing import List, Tuple
from datetime import datetime
import torch

try:
    import umap.umap_ as umap
except Exception:  # pragma: no cover - optional dependency
    umap = None

_METRIC_DIR = Path("../metric_logs")
_METRIC_DIR.mkdir(parents=True, exist_ok=True)

def _log_metric(file: str, msg: str) -> None:
    """Append a timestamped line to metric_logs/<file>."""
    fp = _METRIC_DIR / file
    with open(fp, "a", encoding="utf-8") as f:
        f.write(f"{datetime.utcnow().isoformat()}\t{msg}\n")


# ─────────────────────────────────────────────────────────────────────────────
# A. JACCARD HELPER (token-level)
# ─────────────────────────────────────────────────────────────────────────────
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
_STOPWORDS = set(ENGLISH_STOP_WORDS)
_RE_TOK    = re.compile(r"[A-Za-z0-9']+")

def _jaccard(a: str, b: str) -> float:
    """
    Compute Jaccard overlap of token sets (excluding stopwords).
    """
    s1 = {t for t in _RE_TOK.findall(a.lower()) if t not in _STOPWORDS}
    s2 = {t for t in _RE_TOK.findall(b.lower()) if t not in _STOPWORDS}
    return len(s1 & s2) / (len(s1 | s2) or 1)


# ─────────────────────────────────────────────────────────────────────────────
# B. TF-IDF COSINES (word-level & char-level)
# ─────────────────────────────────────────────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

def _fit_vecs(
    corpus: List[str],
    cache: Path,
    *,
    allow_fit: bool = True,
) -> Tuple[TfidfVectorizer, TfidfVectorizer]:
    """
    Fit (or load from cache) two TfidfVectorizer objects:
      - vec_w: word n-grams (1–2), min_df=3
      - vec_c: char n-grams (3–5), min_df=10

    If cache/tfidf_w.pkl & cache/tfidf_c.pkl both exist, simply load & return them.
    If not found and allow_fit == False, raise a RuntimeError.
    Otherwise, fit on the provided `corpus`, save to disk, and return.
    """
    cache.mkdir(parents=True, exist_ok=True)
    fp_w = cache / "tfidf_w.pkl"
    fp_c = cache / "tfidf_c.pkl"

    if fp_w.exists() and fp_c.exists():
        return joblib.load(fp_w), joblib.load(fp_c)

    if not allow_fit:
        raise RuntimeError(
            "TF-IDF pickles not found in cache. Run feature-engineering on TRAIN first to create them."
        )

    vec_w = TfidfVectorizer(ngram_range=(1,2), min_df=3, sublinear_tf=True)
    vec_c = TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=10)
    vec_w.fit(corpus)
    vec_c.fit(corpus)
    joblib.dump(vec_w, fp_w)
    joblib.dump(vec_c, fp_c)
    return vec_w, vec_c

def _cosine_rows(X: sp.csr_matrix, idx1: np.ndarray, idx2: np.ndarray) -> np.ndarray:
    """
    Given a CSR matrix X (each row is a TF-IDF vector), compute
    cosine similarity between rows idx1[i] and idx2[i] for i=0...n-1.
    Returns a float32 array of length n.
    """
    Xn   = normalize(X, axis=1)
    sims = Xn[idx1].multiply(Xn[idx2]).sum(axis=1)
    return np.asarray(sims).ravel().astype("float32")


# ─────────────────────────────────────────────────────────────────────────────
# C. SVD DIMENSIONALITY REDUCTION
# ─────────────────────────────────────────────────────────────────────────────
from sklearn.decomposition import TruncatedSVD

def _svd_projection(
    X: sp.csr_matrix,
    n_comp: int,
    cache: Path
) -> np.ndarray:
    """
    Either load a pre-fitted TruncatedSVD from disk, or fit it now on X,
    then return X_transformed (float32) of shape (n_rows, n_comp).

    We expect:
      - cache/"svd_w_150.pkl"   if n_comp == 150
      - cache/"svd_c_100.pkl"   if n_comp == 100
    """
    cache.mkdir(parents=True, exist_ok=True)

    if n_comp == 150:
        pickle_name = "svd_w_150.pkl"
    elif n_comp == 100:
        pickle_name = "svd_c_100.pkl"
    else:
        raise ValueError(f"Unsupported n_comp={n_comp} for SVD caching")

    fp_model = cache / pickle_name

    if fp_model.exists():
        svd = joblib.load(fp_model)
        Z   = svd.transform(X).astype("float32")
        return Z

    svd = TruncatedSVD(n_components=n_comp, random_state=42).fit(X)
    joblib.dump(svd, fp_model)
    Z   = svd.transform(X).astype("float32")
    return Z


# ─────────────────────────────────────────────────────────────────────────────
# D. QUORA-DISTILBERT PAIR-VECTOR BLOCK
# ─────────────────────────────────────────────────────────────────────────────
def _dense_pair(emb: np.ndarray, idx1: np.ndarray, idx2: np.ndarray) -> np.ndarray:
    """
    Given precomputed sentence embeddings of shape (n_questions, dim),
    produce pairwise vectors: [u, v, |u−v|, u*v] for each pair (idx1[i], idx2[i]).
    Returns shape (n_pairs, 4*dim) as float32.
    """
    u = emb[idx1]
    v = emb[idx2]
    return np.hstack([u, v, np.abs(u - v), u * v]).astype("float32")


# ─────────────────────────────────────────────────────────────────────────────
# E. FUZZY / LENGTH-DIFF BLOCK
# ─────────────────────────────────────────────────────────────────────────────
from rapidfuzz.fuzz import ratio, token_set_ratio

def _fuzzy_block(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    For each pair of raw question strings, compute:
      - ratio(q1,q2) /100
      - token_set_ratio(q1,q2) /100
      - |char_len_diff|
      - |word_count_diff|
    Returns a float32 array of shape (n_pairs, 4).
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
      - raw char-length of q1  (len_q1_raw)
      - raw char-length of q2  (len_q2_raw)
      - absolute char-length diff -> one-hot into 10 bins
      - min frequency of the two question IDs -> log1p + quartile one-hot (4)

    Returns float32 array of shape (n_pairs, 18).
    """
    # raw character lengths for each question
    len_q1 = meta["len"].values[idx1].astype("float32")[:, None]  # (n_pairs, 1)
    len_q2 = meta["len"].values[idx2].astype("float32")[:, None]  # (n_pairs, 1)

    # absolute length diff
    ld = np.abs(len_q1.squeeze() - len_q2.squeeze())              # (n_pairs,)
    len_diff = ld[:, None]                                        # (n_pairs, 1)

    # ── bucket into 10 bins ───────────────────────────────────────
    # pd.cut(..., labels=False) returns a numpy array of ints or NaN, not a Series.
    bins = pd.cut(ld, [0,2,4,6,8,11,16,22,31,50,1e5],
                  labels=False, include_lowest=True)
    # Convert to float array, replace any NaN with 0, then cast to int
    bins = np.asarray(bins, dtype="float32")
    bins_filled = np.where(np.isnan(bins), 0, bins).astype(int)  # (n_pairs,)
    len_oh = np.eye(10, dtype="float32")[bins_filled]            # (n_pairs, 10)

    # ── min_freq log1p + quartile one-hot ─────────────────────────
    mf     = np.minimum(freq_arr[idx1], freq_arr[idx2]).astype("float32")  # (n_pairs,)
    mf_log = np.log1p(mf)[:, None]  # (n_pairs, 1)

    # pd.qcut(..., labels=False) also returns a numpy array with possible NaNs
    quart = pd.qcut(mf, 4, labels=False, duplicates="drop")
    quart = np.asarray(quart, dtype="float32")  # (n_pairs,) of ints or NaN
    quart_filled = np.where(np.isnan(quart), 0, quart).astype(int)  # fill NaN -> 0
    mf_oh = np.eye(4, dtype="float32")[quart_filled]  # (n_pairs, 4)

    return np.hstack([len_q1, len_q2, len_diff, len_oh, mf_log, mf_oh])


# ─────────────────────────────────────────────────────────────────────────────
# G. CROSS-ENCODER PROBABILITY BLOCK (with retry logic)
# ─────────────────────────────────────────────────────────────────────────────
import time
from sentence_transformers import CrossEncoder

_CROSS_CACHE: dict[str, CrossEncoder] = {}

def _cross_scores(
    pair_df     : pd.DataFrame,
    model_name  : str = "cross-encoder/quora-roberta-large",
    batch_size  : int = 64,
    cache_file  : str | None = None
) -> np.ndarray:
    """
    Compute (or load) CrossEncoder duplicate-prob scores for each (q1, q2).
    If `cache_file` exists on disk, mmap-loads and returns that array.
    Otherwise, attempts to instantiate + run CrossEncoder, with up to 10 retries
    on network/ConnectionError.  If all retries fail, re-raises the last error.

    Returns:
        float32 array of shape (n_pairs,).
    """
    # 1) If we already have a cached .npy, just load it and return:
    if cache_file and Path(cache_file).exists():
        return np.load(cache_file, mmap_mode="r")

    # 2) If we haven't cached this model in memory yet, try to instantiate it:
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
                # Only retry on network‐related errors
                if isinstance(e, ConnectionError) or isinstance(e, OSError):
                    last_exc = e
                    if attempt < 9:
                        # wait a bit, then retry
                        print(
                            f"[features.py] Warning: failed to load CrossEncoder '{model_name}' "
                            f"(attempt {attempt+1}/10). Retrying in 2s..."
                        )
                        time.sleep(2)
                        continue
                # If it's not a ConnectionError or we've exhausted retries, re-raise
                raise

        if last_exc is not None:
            # If we fell out of the loop still with an exception, raise it now
            raise last_exc

    # 3) Having (perhaps) just loaded the model into _CROSS_CACHE, run inference:
    model = _CROSS_CACHE[model_name]
    tuples = list(zip(pair_df.question1.values, pair_df.question2.values))

    # We don't expect predict(...) to hit the network again,
    # but wrap in the same retry logic just in case.
    last_exc = None
    for attempt in range(10):
        try:
            scores = model.predict(tuples, batch_size=batch_size).astype("float32")
            last_exc = None
            break
        except Exception as e:
            if isinstance(e, ConnectionError) or isinstance(e, OSError):
                last_exc = e
                if attempt < 9:
                    print(
                        f"[features.py] Warning: CrossEncoder.predict failed "
                        f"(attempt {attempt+1}/10). Retrying in 2s..."
                    )
                    time.sleep(2)
                    continue
            raise

    if last_exc is not None:
        raise last_exc

    # 4) Save to disk if requested, then return
    if cache_file:
        np.save(cache_file, scores)

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# H. MASTER BUILDER FUNCTION w/ INCREMENTAL PCA REDUCTION
# ─────────────────────────────────────────────────────────────────────────────
from sklearn.decomposition import IncrementalPCA

def build_features(
    pair_df         : pd.DataFrame,
    clean_questions : List[str],
    meta_df         : pd.DataFrame,
    embedding_path  : str = "../data/processed/question_embeddings.npy",
    cache_dir       : str = "../models/features",
    cross_cache     : str | None = "../data/processed/train_cross_scores.npy",
    fit_pca         : bool = False,
    features_cache  : str | None = None,
    reduction       : str = "ipca",
    n_components    : int | None = None,
) -> np.ndarray:
    """
    Build the full feature matrix of size (n_pairs × (526 + 4×dim)), then
    optionally apply dimensionality reduction via IncrementalPCA and/or UMAP.
    Results can be cached to disk for reuse.

    Arguments
    ---------
    pair_df: DataFrame with columns ['qid1','qid2','question1','question2'].
    clean_questions: List[str] of length = n_unique_questions.
    meta_df: DataFrame from question_meta.csv (contains at least 'len').
    embedding_path: path -> (n_questions×dim) .npy file of sentence embeddings,
        where `dim` is typically 768 (DistilBERT/MPNet) or 384 for MiniLM models.
    cache_dir: directory where TF-IDF/SVD/PCA models live.
    cross_cache: optional path -> .npy of cached cross-encoder scores; if None, recompute.
    fit_pca: if True, (re)fit the dimensionality-reduction model(s).
             If False, load from cache_dir.
    features_cache: optional path to save/load the raw feature matrix
                    before dimensionality reduction.
    reduction: 'ipca' | 'umap' | 'pca_umap'.
    n_components: target dimension after reduction. If None and reduction='ipca'
                  the PCA 95%%-variance heuristic is used.

    Returns
    -------
    X_red: np.ndarray of shape (n_pairs, n_components). When n_components is
           None with reduction='ipca', the output dimension equals the number of
           components needed for 95% explained variance.
    """
    # 1) Extract pair indices
    idx1 = pair_df.qid1.values.astype("int32")
    idx2 = pair_df.qid2.values.astype("int32")

    if np.isnan(idx1).any() or np.isnan(idx2).any():
        raise ValueError("Found NaN in qid1 / qid2. Drop null questions first.")

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    # ───────────────────────────────────────────────────────────────────
    # A. TF-IDF COSINES
    # ───────────────────────────────────────────────────────────────────
    vec_w, vec_c = _fit_vecs(clean_questions, cache, allow_fit=fit_pca)
    Xw           = vec_w.transform(clean_questions)  # (n_q, vocab_w)
    Xc           = vec_c.transform(clean_questions)  # (n_q, vocab_c)

    cos_w = _cosine_rows(Xw, idx1, idx2)  # (n_pairs,)
    cos_c = _cosine_rows(Xc, idx1, idx2)  # (n_pairs,)
    cos_block = np.vstack([cos_w, cos_c]).T  # (n_pairs, 2)

    # ───────────────────────────────────────────────────────────────────
    # B. SVD PROJECTIONS
    # ───────────────────────────────────────────────────────────────────
    Zw = _svd_projection(Xw, n_comp=150, cache=cache)  # (n_q, 150)
    Zc = _svd_projection(Xc, n_comp=100, cache=cache)  # (n_q, 100)

    svd_left   = Zw[idx1]   # (n_pairs, 150)
    svd_right  = Zw[idx2]
    svd_word   = np.hstack([svd_left, svd_right])     # (n_pairs, 300)

    svd_left_c  = Zc[idx1]   # (n_pairs, 100)
    svd_right_c = Zc[idx2]
    svd_char    = np.hstack([svd_left_c, svd_right_c]) # (n_pairs, 200)

    svd_block = np.hstack([svd_word, svd_char])  # (n_pairs, 500)

    # ───────────────────────────────────────────────────────────────────
    # C. QUORA-DISTILBERT PAIR-VECTOR
    # ───────────────────────────────────────────────────────────────────
    emb         = np.load(embedding_path, mmap_mode="r")  # (n_q, dim)
    dense_block = _dense_pair(emb, idx1, idx2)            # (n_pairs, 4*dim)

    # ───────────────────────────────────────────────────────────────────
    # D. FUZZY / LENGTH DIFFS
    # ───────────────────────────────────────────────────────────────────
    fuzzy_block = _fuzzy_block(
        pair_df.question1.values,
        pair_df.question2.values
    )  # (n_pairs, 4)

    # ───────────────────────────────────────────────────────────────────
    # E. JACCARD
    # ───────────────────────────────────────────────────────────────────
    jacc_block = np.fromiter(
        (_jaccard(a, b) for a, b in zip(pair_df.question1, pair_df.question2)),
        dtype="float32"
    ).reshape(-1, 1)  # (n_pairs, 1)

    # ───────────────────────────────────────────────────────────────────
    # F. NUMERIC “MAGIC” FEATURES
    # ───────────────────────────────────────────────────────────────────
    freq_arr = np.bincount(
        np.concatenate([pair_df.qid1.values, pair_df.qid2.values]),
        minlength=len(meta_df)
    ).astype("float32")

    numeric_block = _numeric(meta_df, idx1, idx2, freq_arr)  # (n_pairs, 18)

    # ───────────────────────────────────────────────────────────────────
    # G. CROSS-ENCODER PROBABILITIES
    # ───────────────────────────────────────────────────────────────────
    cross_block = _cross_scores(
        pair_df,
        model_name="cross-encoder/quora-roberta-large",
        batch_size=64,
        cache_file=cross_cache
    ).reshape(-1, 1)  # (n_pairs, 1)

    # ───────────────────────────────────────────────────────────────────
    # CONCAT raw feature blocks
    # ───────────────────────────────────────────────────────────────────
    X_raw = np.hstack([
        cos_block,      # 2
        dense_block,    # 3072
        fuzzy_block,    # 4
        jacc_block,     # 1
        numeric_block,  # 18
        cross_block,    # 1
        svd_block       # 500
    ]).astype("float32")  # -> (n_pairs, 526 + 4*dim)

    if features_cache:
        fp = Path(features_cache)
        if fit_pca and not fp.exists():
            np.save(fp, X_raw)
            _log_metric("feature_engineering.txt", f"saved raw features to {fp}")
        elif fp.exists():
            X_raw = np.load(fp, mmap_mode="r")
            _log_metric("feature_engineering.txt", f"loaded raw features from {fp}")

    # ───────────────────────────────────────────────────────────────────
    # Dimensionality reduction
    # ───────────────────────────────────────────────────────────────────
    red = reduction.lower()

    if red == "ipca":
        if n_components is None:
            pca_fp = cache / "pca_95.pkl"
        else:
            pca_fp = cache / f"ipca_{n_components}.pkl"

        if fit_pca or not pca_fp.exists():
            start = time.time()
            if n_components is None:
                ipca_full = IncrementalPCA(n_components=None)
                ipca_full.fit(X_raw)
                cumvar = np.cumsum(ipca_full.explained_variance_ratio_)
                k95 = int(np.searchsorted(cumvar, 0.95)) + 1
                ipca = IncrementalPCA(n_components=k95)
            else:
                ipca = IncrementalPCA(n_components=n_components)
            X_red = ipca.fit_transform(X_raw).astype("float32")
            joblib.dump(ipca, pca_fp)
            _log_metric("feature_engineering.txt", f"fit IPCA -> {X_red.shape[1]} dims in {time.time()-start:.1f}s")
        else:
            ipca = joblib.load(pca_fp)
            X_red = ipca.transform(X_raw).astype("float32")
            _log_metric("feature_engineering.txt", f"loaded IPCA model {pca_fp}")
        return X_red

    if red == "umap":
        if umap is None:
            raise RuntimeError("umap-learn package is required for reduction='umap'")
        if n_components is None:
            raise ValueError("n_components must be specified for UMAP reduction")
        umap_fp = cache / f"umap_{n_components}.pkl"
        if fit_pca or not umap_fp.exists():
            start = time.time()
            reducer = umap.UMAP(n_components=n_components, random_state=42)
            X_red = reducer.fit_transform(X_raw).astype("float32")
            joblib.dump(reducer, umap_fp)
            _log_metric("feature_engineering.txt", f"fit UMAP -> {n_components} dims in {time.time()-start:.1f}s")
        else:
            reducer = joblib.load(umap_fp)
            X_red = reducer.transform(X_raw).astype("float32")
            _log_metric("feature_engineering.txt", f"loaded UMAP model {umap_fp}")
        return X_red

    if red == "pca_umap":
        if umap is None:
            raise RuntimeError("umap-learn package is required for reduction='pca_umap'")
        if n_components is None:
            raise ValueError("n_components must be specified for PCA+UMAP reduction")
        pca_fp = cache / "pca_95.pkl"
        umap_fp = cache / f"pca_umap_{n_components}.pkl"
        if fit_pca or not (pca_fp.exists() and umap_fp.exists()):
            start = time.time()
            ipca_full = IncrementalPCA(n_components=None)
            ipca_full.fit(X_raw)
            cumvar = np.cumsum(ipca_full.explained_variance_ratio_)
            k95 = int(np.searchsorted(cumvar, 0.95)) + 1
            ipca = IncrementalPCA(n_components=k95)
            X_ipca = ipca.fit_transform(X_raw)
            joblib.dump(ipca, pca_fp)

            reducer = umap.UMAP(n_components=n_components, random_state=42)
            X_red = reducer.fit_transform(X_ipca).astype("float32")
            joblib.dump(reducer, umap_fp)
            _log_metric(
                "feature_engineering.txt",
                f"fit PCA({k95})+UMAP({n_components}) in {time.time()-start:.1f}s",
            )
        else:
            ipca = joblib.load(pca_fp)
            reducer = joblib.load(umap_fp)
            X_ipca = ipca.transform(X_raw)
            X_red = reducer.transform(X_ipca).astype("float32")
            _log_metric("feature_engineering.txt", f"loaded PCA+UMAP models")
        return X_red

    raise ValueError(f"Unknown reduction='{reduction}'")
