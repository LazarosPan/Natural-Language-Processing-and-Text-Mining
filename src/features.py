from __future__ import annotations
import os, sys

"""
Feature extraction for Quora Duplicate Question task
----------------------------------------------------
Blocks
 A  word- and char-level TF-IDF cosine               (2 cols)
 B  SBERT pair vector (u, v, |u-v|, u*v)             (3072 cols)
 C  Fuzzy lexical + length/word diffs                (4 cols)
 D  Jaccard token overlap                            (1 col)
 E  Numeric / “magic” features
       • len_diff raw + 10 bins one-hot             (11 cols)
       • min_freq log1p + quartile one-hot          (5  cols)
Total dims = 3095
"""
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"     # skip TensorFlow / Keras
os.environ["TRANSFORMERS_NO_FLAX"] = "1"   # skip JAX / Flax
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # silence leftover TF logs

# optional: verify nothing from TF is on sys.modules
print("boot ok, TF loaded:", any(m.startswith("tensorflow") for m in sys.modules))

import re, joblib, numpy as np, pandas as pd, scipy.sparse as sp
from pathlib import Path
from typing import List

# ------------------------------------------------------------------ #
# tiny helpers
# ------------------------------------------------------------------ #
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
_Stop = set(ENGLISH_STOP_WORDS)
_TOK = re.compile(r"[A-Za-z0-9']+")

def _jaccard(a: str, b: str) -> float:
    s1 = {t for t in _TOK.findall(a) if t not in _Stop}
    s2 = {t for t in _TOK.findall(b) if t not in _Stop}
    return len(s1 & s2) / (len(s1 | s2) or 1)

# ------------------------------------------------------------------ #
# TF-IDF
# ------------------------------------------------------------------ #
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

def _fit_vecs(corpus: List[str], cache: Path):
    cache.mkdir(exist_ok=True)
    fw, fc = cache/"tfidf_w.pkl", cache/"tfidf_c.pkl"
    if fw.exists():
        vec_w, vec_c = joblib.load(fw), joblib.load(fc)
    else:
        vec_w = TfidfVectorizer(ngram_range=(1,2), min_df=3, sublinear_tf=True)
        vec_c = TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=10)
        vec_w.fit(corpus); vec_c.fit(corpus)
        joblib.dump(vec_w, fw); joblib.dump(vec_c, fc)
    return vec_w, vec_c

def _cosine_rows(X: sp.csr_matrix, i1, i2):
    X = normalize(X, axis=1)
    return np.asarray(X[i1].multiply(X[i2]).sum(1)).ravel().astype("float32")

# ------------------------------------------------------------------ #
# SBERT
# ------------------------------------------------------------------ #
from sentence_transformers import SentenceTransformer
_SBERT = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda")

def _embed(texts: List[str], cache: Path):
    fp = cache/"sbert.npy"
    if fp.exists():
        return np.load(fp, mmap_mode="r")
    emb = _SBERT.encode(texts, batch_size=512,
                        normalize_embeddings=True,
                        convert_to_numpy=True).astype("float32")
    np.save(fp, emb); return emb

def _dense_pair(emb: np.ndarray, i1, i2):
    u, v = emb[i1], emb[i2]
    return np.hstack([u, v, np.abs(u-v), u*v])

# ------------------------------------------------------------------ #
# Fuzzy
# ------------------------------------------------------------------ #
from rapidfuzz.fuzz import ratio, token_set_ratio
def _fuzzy_block(q1, q2):
    out = np.zeros((len(q1), 4), dtype="float32")
    for k,(a,b) in enumerate(zip(q1, q2)):
        out[k,0] = ratio(a,b)/100
        out[k,1] = token_set_ratio(a,b)/100
        out[k,2] = abs(len(a)-len(b))
        out[k,3] = abs(len(a.split())-len(b.split()))
    return out

# ------------------------------------------------------------------ #
# Numeric / buckets
# ------------------------------------------------------------------ #
def _numeric(meta, i1, i2, freq_arr):
    ld = np.abs(meta["len"].to_numpy()[i1] - meta["len"].to_numpy()[i2]).astype("float32")
    len_raw = ld.reshape(-1,1)
    bins = pd.cut(ld, [0,2,4,6,8,11,16,22,31,50,1e5], labels=False, include_lowest=True)
    len_oh = np.eye(10, dtype="float32")[bins]

    mf = np.minimum(freq_arr[i1], freq_arr[i2]).astype("float32")
    mf_log = np.log1p(mf).reshape(-1,1)
    quart = pd.qcut(mf, 4, labels=False, duplicates="drop")
    mf_oh = np.eye(4, dtype="float32")[quart]

    return np.hstack([len_raw, len_oh, mf_log, mf_oh])

# ------------------------------------------------------------------ #
def build_features(pair_df: pd.DataFrame,
                   clean_questions: List[str],
                   meta_df: pd.DataFrame,
                   cache_dir="models") -> np.ndarray:
    """
    pair_df must contain columns:
        qid1, qid2, question1, question2
    meta_df must have 'len' col; index = question-numeric-ID
    """
    cache = Path(cache_dir)
    vec_w, vec_c = _fit_vecs(clean_questions, cache)
    Xw = vec_w.transform(clean_questions)
    Xc = vec_c.transform(clean_questions)

    idx1 = pair_df.qid1.to_numpy()
    idx2 = pair_df.qid2.to_numpy()

    # blocks
    cos = np.vstack([_cosine_rows(Xw, idx1, idx2),
                     _cosine_rows(Xc, idx1, idx2)]).T

    emb  = _embed(clean_questions, cache)
    dense= _dense_pair(emb, idx1, idx2)

    fuzzy= _fuzzy_block(pair_df.question1.to_numpy(), pair_df.question2.to_numpy())

    jacc = np.fromiter((_jaccard(a,b) for a,b
                        in zip(pair_df.question1, pair_df.question2)),
                       dtype="float32").reshape(-1,1)

    freq_map = pd.Series(range(len(meta_df)), index=meta_df.index)  # identity map
    freq_arr = np.array(
        pd.concat([pair_df.qid1, pair_df.qid2]).value_counts()
        .reindex(range(len(meta_df)), fill_value=1)
    )
    num   = _numeric(meta_df, idx1, idx2, freq_arr)

    X = np.hstack([cos, dense, fuzzy, jacc, num]).astype("float32")
    return X