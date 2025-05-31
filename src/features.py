"""
src/features.py
────────────────────────────────────────────────────────────────────────────
Builds the full feature matrix for a (qid1, qid2) pairs dataframe.

Blocks
 A. TF-IDF cosine  (word & char)                              → 2   cols
 B. MiniLM sentence-pair vector  (u, v, |u-v|, u*v)           → 1536 cols
 C. Fuzzy / length diffs                                     → 4   cols
 D. Token-level Jaccard                                      → 1   col
 E. “Magic” numeric features (len_diff + buckets + min_freq) → 16  cols
Total = 1 559 columns  (float32)
"""

from __future__ import annotations
import os, re, joblib, numpy as np, pandas as pd, scipy.sparse as sp
from pathlib import Path
from typing import List

# ──────────────────────────────────────────────────────────────
# basic token / stopword helpers
# ──────────────────────────────────────────────────────────────
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
_STOP = set(ENGLISH_STOP_WORDS)
_TOK  = re.compile(r"[A-Za-z0-9']+")


def _jaccard(a: str, b: str) -> float:
    s1 = {t for t in _TOK.findall(a.lower()) if t not in _STOP}
    s2 = {t for t in _TOK.findall(b.lower()) if t not in _STOP}
    return len(s1 & s2) / (len(s1 | s2) or 1)


# ──────────────────────────────────────────────────────────────
# A. TF-IDF cosine (word & char)
# ──────────────────────────────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def _fit_vecs(corpus: List[str], cache: Path):
    """Load or fit two TfidfVectorizers (word + char)."""
    cache.mkdir(exist_ok=True)
    fp_w, fp_c = cache / "tfidf_w.pkl", cache / "tfidf_c.pkl"

    if fp_w.exists():
        return joblib.load(fp_w), joblib.load(fp_c)

    vec_w = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    vec_c = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=10)
    vec_w.fit(corpus)
    vec_c.fit(corpus)
    joblib.dump(vec_w, fp_w)
    joblib.dump(vec_c, fp_c)
    return vec_w, vec_c


def _cosine_rows(X: sp.csr_matrix, i1, i2) -> np.ndarray:
    X = normalize(X, axis=1)
    return np.asarray(X[i1].multiply(X[i2]).sum(1)).ravel().astype("float32")


# ──────────────────────────────────────────────────────────────
# B. SBERT pair-vector block (embedding file already on disk)
# ──────────────────────────────────────────────────────────────
def _dense_pair(emb: np.ndarray, i1, i2) -> np.ndarray:
    u, v = emb[i1], emb[i2]
    return np.hstack([u, v, np.abs(u - v), u * v]).astype("float32")


# ──────────────────────────────────────────────────────────────
# C. RapidFuzz lexical / length diff block
# ──────────────────────────────────────────────────────────────
from rapidfuzz.fuzz import ratio, token_set_ratio


def _fuzzy_block(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    out = np.zeros((len(q1), 4), dtype="float32")
    for k, (a, b) in enumerate(zip(q1, q2)):
        out[k, 0] = ratio(a, b) / 100.0
        out[k, 1] = token_set_ratio(a, b) / 100.0
        out[k, 2] = abs(len(a) - len(b))
        out[k, 3] = abs(len(a.split()) - len(b.split()))
    return out


# ──────────────────────────────────────────────────────────────
# D/E. Numeric “magic” features
# ──────────────────────────────────────────────────────────────
def _numeric(meta: pd.DataFrame,
             i1: np.ndarray,
             i2: np.ndarray,
             freq_arr: np.ndarray) -> np.ndarray:
    # raw char-length diff
    ld = np.abs(meta["len"].to_numpy()[i1] -
                meta["len"].to_numpy()[i2]).astype("float32")
    len_raw = ld[:, None]

    # 10-bin one-hot bucket
    bins = pd.cut(
        ld, [0, 2, 4, 6, 8, 11, 16, 22, 31, 50, 1e5],
        labels=False, include_lowest=True
    )
    len_oh = np.eye(10, dtype="float32")[bins]

    # min question frequency + log1p
    mf = np.minimum(freq_arr[i1], freq_arr[i2]).astype("float32")
    mf_log = np.log1p(mf)[:, None]
    quart = pd.qcut(mf, 4, labels=False, duplicates="drop")
    mf_oh = np.eye(4, dtype="float32")[quart]

    return np.hstack([len_raw, len_oh, mf_log, mf_oh])


# ──────────────────────────────────────────────────────────────
# MASTER BUILDER
# ──────────────────────────────────────────────────────────────
def build_features(pair_df: pd.DataFrame,
                   clean_questions: List[str],
                   meta_df: pd.DataFrame,
                   embedding_path: str = "../data/processed/question_embeddings.npy",
                   cache_dir: str = "models") -> np.ndarray:
    """
    pair_df  : dataframe with columns [qid1, qid2, question1, question2]
    clean_questions : list[str] ordered so index == qid
    meta_df : question_meta.csv loaded (must contain 'len' col)
    """

    cache = Path(cache_dir)

    # indices mapping pair rows -> question-ID
    idx1 = pair_df.qid1.to_numpy()
    idx2 = pair_df.qid2.to_numpy()

    # A. TF-IDF cosine
    vec_w, vec_c = _fit_vecs(clean_questions, cache)
    cos_block = np.vstack([
        _cosine_rows(vec_w.transform(clean_questions), idx1, idx2),
        _cosine_rows(vec_c.transform(clean_questions), idx1, idx2)
    ]).T  # shape (n_pairs, 2)

    # B. SBERT pair vector   (1536 columns)
    emb = np.load(embedding_path, mmap_mode="r")  # (n_q, 384)
    dense_block = _dense_pair(emb, idx1, idx2)

    # C. Fuzzy & length diffs
    fuzzy_block = _fuzzy_block(pair_df.question1.to_numpy(),
                               pair_df.question2.to_numpy())

    # D. Jaccard
    jacc_block = np.fromiter(
        (_jaccard(a, b) for a, b in zip(pair_df.question1, pair_df.question2)),
        dtype="float32"
    ).reshape(-1, 1)

    # E. Magic numeric (len_diff + min_freq)
    #   Build frequency array once from the whole dataset
    q_freq = np.bincount(
        np.concatenate([pair_df.qid1.values, pair_df.qid2.values]),
        minlength=len(meta_df)
    )
    numeric_block = _numeric(meta_df, idx1, idx2, q_freq.astype("float32"))

    # final concat
    X = np.hstack([
        cos_block,
        dense_block,
        fuzzy_block,
        jacc_block,
        numeric_block
    ]).astype("float32")

    return X


# ──────────────────────────────────────────────────────────────
# Smoke-test   (≈ 1000 rows)
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pandas as pd

    pair = pd.read_csv("data/splits/train.csv")
    meta = pd.read_csv("data/processed/question_meta.csv")
    clean = np.load("data/processed/clean_questions.npy", allow_pickle=True).tolist()

    X = build_features(pair, clean, meta,
                       embedding_path="data/processed/question_embeddings.npy")
    print("Feature matrix:", X.shape)