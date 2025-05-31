# ────────────────────────────────────────────────────────────────
# src/features.py
# ----------------------------------------------------------------
"""
Builds the full feature matrix for Quora Duplicate Question pairs.

BLOCKS
 A 2 × TF-IDF cosine (word & char)                                → 2   cols
 B MiniLM pair-vector (u, v, |u-v|, u*v) (384-dim each)           → 1 536 cols
 C RapidFuzz lexical + length diffs                                → 4   cols
 D Jaccard token overlap                                           → 1   col
 E Numeric “magic” features (len_diff + bucket one-hots + freq)    → 16  cols
 F Quora Cross-Encoder duplicate probability                       → 1   col
 G TF-IDF SVD reduction (150 & 100 dims per side)                   → 500 cols

Total output dimension (float32): 2 060 columns per question-pair.
"""

from __future__ import annotations
import re, joblib, numpy as np, pandas as pd, scipy.sparse as sp
from pathlib import Path
from typing import List, Tuple
import torch

# --------------------------------------
# A. JACCARD HELPER (token-level)
# --------------------------------------
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


# --------------------------------------
# B. TF-IDF COSINES (word-level & char-level)
# --------------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# ───────────────────────────────────────────────────────────────────
def _fit_vecs(
    corpus: List[str],
    cache: Path,
    *,
    allow_fit: bool = True,           # ⟵ new argument
) -> Tuple[TfidfVectorizer, TfidfVectorizer]:
    """
    Fit (or load from cache) two TfidfVectorizer objects:
      • vec_w: word n-grams (1–2), min_df=3
      • vec_c: char n-grams (3–5), min_df=10

    If cache/tfidf_w.pkl & cache/tfidf_c.pkl both exist, simply load & return them.
    If they do not exist and allow_fit == False, raise a RuntimeError.
    Otherwise, .fit(...) on `corpus`, save to disk, and return.
    """
    cache.mkdir(exist_ok=True)
    fp_w = cache / "tfidf_w.pkl"
    fp_c = cache / "tfidf_c.pkl"

    # if pickles exist, load them (transform-only for valid/test)
    if fp_w.exists() and fp_c.exists():
        return joblib.load(fp_w), joblib.load(fp_c)

    # if we’re not allowed to fit (i.e. we’re in valid/test), error out
    if not allow_fit:
        raise RuntimeError(
            "TF-IDF pickles not found in 'models/'.\n"
            "You must run feature-engineering on TRAIN first to create them."
        )

    # otherwise, fit on the provided `corpus` (which must be train_corpus)
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
    Returns float32 array of length n.
    """
    Xn = normalize(X, axis=1)  # row-normalize to unit length
    # elementwise multiply and sum per‐row to get dot product
    sims = Xn[idx1].multiply(Xn[idx2]).sum(axis=1)
    return np.asarray(sims).ravel().astype("float32")


# --------------------------------------
# C. SVD DIMENSIONALITY REDUCTION
# --------------------------------------
from sklearn.decomposition import TruncatedSVD

# ───────────────────────────────────────────────────────────────────
def _svd_projection(
    X: sp.csr_matrix,
    n_comp: int,
    cache: Path
) -> np.ndarray:
    """
    Either load a pre‐fitted TruncatedSVD from disk, or fit it now on X,
    then return X_transformed (dense float32) with shape (n_rows, n_comp).

    We expect the following filenames for cached SVD pickles:
      • cache/"svd_w_150.pkl"   if n_comp == 150
      • cache/"svd_c_100.pkl"   if n_comp == 100

    If that Pickle exists, we do:
        svd = joblib.load(pickle_path)
        Z   = svd.transform(X).astype("float32")
        return Z

    Otherwise (first time, on TRAIN), we do:
        svd = TruncatedSVD(n_components=n_comp, random_state=42).fit(X)
        joblib.dump(svd, pickle_path)
        Z = svd.transform(X).astype("float32")
        return Z
    """
    cache.mkdir(exist_ok=True)

    # Determine which pickle to look for
    if n_comp == 150:
        pickle_name = "svd_w_150.pkl"
    elif n_comp == 100:
        pickle_name = "svd_c_100.pkl"
    else:
        raise ValueError(f"Unsupported n_comp={n_comp} for SVD caching")

    fp_model = cache / pickle_name

    # If the Pickle exists, load & transform
    if fp_model.exists():
        import joblib
        svd = joblib.load(fp_model)
        Z   = svd.transform(X).astype("float32")
        return Z

    # Otherwise: fit on X, then save the Pickle, then transform & return
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=n_comp, random_state=42).fit(X)
    joblib.dump(svd, fp_model)
    Z = svd.transform(X).astype("float32")
    return Z


# --------------------------------------
# D. MiniLM PAIR-VECTOR BLOCK
# --------------------------------------
def _dense_pair(emb: np.ndarray, idx1: np.ndarray, idx2: np.ndarray) -> np.ndarray:
    """
    Given precomputed sentence embeddings (emb) of shape (n_questions, dim),
    produce pairwise vectors: [u, v, |u−v|, u*v] for each pair (idx1[i], idx2[i]).
    Returns shape (n_pairs, 4*dim) dtype float32.
    """
    u = emb[idx1]  # shape (n_pairs, dim)
    v = emb[idx2]
    return np.hstack([u, v, np.abs(u - v), u * v]).astype("float32")


# --------------------------------------
# E. FUZZY / LENGTH-DIFF BLOCK
# --------------------------------------
from rapidfuzz.fuzz import ratio, token_set_ratio

def _fuzzy_block(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    For each pair of raw question strings, compute:
      • ratio(q1,q2) / 100
      • token_set_ratio(q1,q2) / 100
      • abs(char_len_diff)
      • abs(word_count_diff)
    Returns float32 array of shape (n_pairs, 4).
    """
    n = len(q1)
    out = np.zeros((n, 4), dtype="float32")
    for i, (a, b) in enumerate(zip(q1, q2)):
        out[i, 0] = ratio(a, b) / 100.0
        out[i, 1] = token_set_ratio(a, b) / 100.0
        out[i, 2] = abs(len(a) - len(b))
        out[i, 3] = abs(len(a.split()) - len(b.split()))
    return out


# --------------------------------------
# F. NUMERIC “MAGIC” FEATURES
# --------------------------------------
def _numeric(
    meta: pd.DataFrame,
    idx1: np.ndarray,
    idx2: np.ndarray,
    freq_arr: np.ndarray
) -> np.ndarray:
    """
    Build numeric features per pair:
      • raw char-length diff (ld)
      • one-hot bucket (10 bins) of ld
      • min frequency of the two question IDs → log1p + quartile one-hot (4)
    Returns float32 array of shape (n_pairs, 1 + 10 + 1 + 4 = 16).
    """
    # raw length diff
    ld = np.abs(meta["len"].values[idx1] - meta["len"].values[idx2]).astype("float32")
    len_raw = ld[:, None]  # shape (n_pairs, 1)

    # bucket into 10 bins
    bins = pd.cut(ld, [0,2,4,6,8,11,16,22,31,50,1e5],
                  labels=False, include_lowest=True)
    len_oh = np.eye(10, dtype="float32")[bins]  # (n_pairs, 10)

    # min_freq log1p + quartile one-hot
    mf      = np.minimum(freq_arr[idx1], freq_arr[idx2]).astype("float32")
    mf_log  = np.log1p(mf)[:, None]             # (n_pairs, 1)
    quart   = pd.qcut(mf, 4, labels=False, duplicates="drop")
    mf_oh   = np.eye(4, dtype="float32")[quart]  # (n_pairs, 4)

    return np.hstack([len_raw, len_oh, mf_log, mf_oh])


# --------------------------------------
# G. CROSS-ENCODER PROBABILITY BLOCK
# --------------------------------------
from sentence_transformers import CrossEncoder
_CROSS_CACHE: dict[str, CrossEncoder] = {}

def _cross_scores(
    pair_df     : pd.DataFrame,
    model_name  : str = "cross-encoder/quora-roberta-large",
    batch_size  : int = 64,
    cache_file  : str | None = None
) -> np.ndarray:
    """
    Compute (or load) CrossEncoder duplicate-prob scores for each (q1, q2) pair.
    If `cache_file` exists on disk, mmap-loads and returns that array.
    Otherwise, loads CrossEncoder once, predicts on all pairs, saves to cache_file.
    Returns float32 array shape (n_pairs,).
    """
    if cache_file and Path(cache_file).exists():
        return np.load(cache_file, mmap_mode="r")

    if model_name not in _CROSS_CACHE:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _CROSS_CACHE[model_name] = CrossEncoder(
            model_name,
            device=dev,
            activation_fn=torch.nn.Sigmoid()
        )

    model = _CROSS_CACHE[model_name]
    tuples = list(zip(pair_df.question1.values, pair_df.question2.values))
    scores = model.predict(tuples, batch_size=batch_size).astype("float32")

    if cache_file:
        np.save(cache_file, scores)
    return scores


# --------------------------------------
# MASTER BUILDER FUNCTION
# --------------------------------------
def build_features(
    pair_df         : pd.DataFrame,
    clean_questions : List[str],
    meta_df         : pd.DataFrame,
    embedding_path  : str = "../data/processed/question_embeddings.npy",
    cache_dir       : str = "models",
    cross_cache     : str | None = "../data/processed/train_cross_scores.npy"
) -> np.ndarray:
    """
    Build the full feature matrix for a DataFrame of pairs.

    Arguments
    ---------
    pair_df: DataFrame containing columns:
             ['qid1', 'qid2', 'question1', 'question2']  
             where qid1/qid2 are integer indices into clean_questions & meta_df.

    clean_questions: List[str] of length = n_unique_questions, where
                     clean_questions[i] is the cleaned text of question ID i.

    meta_df: DataFrame loaded from question_meta.csv, containing at least
             a 'len' column (cleaned char-length) and matching index=question ID.

    embedding_path: path → npy file of shape (n_unique_questions, 384),
                    for MiniLM embeddings cached by preprocess step.

    cache_dir: directory to store / load TF-IDF vectorizers and SVD projections.

    cross_cache: optional path to .npy file for cached CrossEncoder scores
                 for training pairs. If None, Cross‐Encoder is always recomputed.

    Returns
    -------
    X: np.ndarray, shape (n_pairs, 2060), dtype float32.
    """
    # 1) extract pair indices
    idx1 = pair_df.qid1.values.astype("int32")
    idx2 = pair_df.qid2.values.astype("int32")

    # Safety check: no NaNs in qid mapping
    if np.isnan(idx1).any() or np.isnan(idx2).any():
        raise ValueError("NaN found in qid1 / qid2. Did you drop null questions?")

    cache = Path(cache_dir)

    # ───────────────────────────────────────────────────────────────────
    # A. TF-IDF COSINES
    # ───────────────────────────────────────────────────────────────────
    vec_w, vec_c = _fit_vecs(clean_questions, cache)
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

    svd_left  = Zw[idx1]  # (n_pairs, 150)
    svd_right = Zw[idx2]
    svd_word  = np.hstack([svd_left, svd_right])  # (n_pairs, 300)

    svd_left_c  = Zc[idx1]  # (n_pairs, 100)
    svd_right_c = Zc[idx2]
    svd_char    = np.hstack([svd_left_c, svd_right_c])  # (n_pairs, 200)

    svd_block = np.hstack([svd_word, svd_char])  # (n_pairs, 500)

    # ───────────────────────────────────────────────────────────────────
    # C. MiniLM PAIR-VECTOR
    # ───────────────────────────────────────────────────────────────────
    emb         = np.load(embedding_path, mmap_mode="r")  # (n_q, 384)
    dense_block = _dense_pair(emb, idx1, idx2)            # (n_pairs, 1536)

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
    # Build freq array (size = n_q) from all qid occurrences in this pair_df
    freq_arr = np.bincount(
        np.concatenate([pair_df.qid1.values, pair_df.qid2.values]),
        minlength=len(meta_df)
    ).astype("float32")

    numeric_block = _numeric(meta_df, idx1, idx2, freq_arr)  # (n_pairs, 16)

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
    # FINAL CONCATENATION
    # ───────────────────────────────────────────────────────────────────
    X = np.hstack([
        cos_block,       # 2
        dense_block,     # 1536
        fuzzy_block,     # 4
        jacc_block,      # 1
        numeric_block,   # 16
        cross_block,     # 1
        svd_block        # 500
    ]).astype("float32")  # → total 2060 columns

    return X