# ────────────────────────────────────────────────────────────────
# src/custom_models.py
# ----------------------------------------------------------------
"""
“Feature-based” experts built on IPCA- or UMAP-reduced feature matrices
for different SBERT-dim tracks (e.g., 384 or 768). Each expert wraps
an sklearn-style estimator and exposes a .predict_prob(pairs) API.

Supported experts:
  * LRFeatureExpert        – LogisticRegression
  * XGBFeatureExpert       – XGBoost
  * LGBMFeatureExpert      – LightGBM
  * KNNFeatureExpert       – K-Nearest Neighbors
  * RFFeatureExpert        – Random Forest
  * SVMFeatureExpert       – SVM with calibration

Usage examples:
    # LogisticRegression on 384-dim IPCA features
    lr_ipca = LRFeatureExpert(dim=384, reduction="ipca")
    lr_ipca.fit(train_df, y_train)
    probs = lr_ipca.predict_prob(valid_pairs)

    # RandomForest on 768-dim UMAP(10) features
    rf_umap = RFFeatureExpert(dim=768, reduction="umap", n_components=10)
    rf_umap.fit(train_df, y_train)
    probs = rf_umap.predict_prob(valid_pairs)
"""

from __future__ import annotations
import pickle
from pathlib import Path
from typing import List, Tuple
import time

import numpy as np
import pandas as pd

from src.features import build_features
from src.pretrained_models import BaseExpert  # assumes abstract interface
from src.logs import log_event, LogKind

Pair = Tuple[str, str]


# ─────────────────────────────────────────────────────────────────────────────
# 1) CONSTANTS & PRE-LOADED ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────

# Directory where per-question artifacts live
_PROCESSED = Path("../data/processed")
if not _PROCESSED.exists():
    raise FileNotFoundError(
        f"Expected directory '{_PROCESSED}' does not exist. "
        "Run preprocessing notebooks to generate per-question artifacts."
    )

# question_meta.csv must exist
_META_FP = _PROCESSED / "question_meta.csv"
if not _META_FP.exists():
    raise FileNotFoundError(
        f"Missing '{_META_FP}'. Run preprocessing to generate question_meta.csv."
    )
_meta = pd.read_csv(_META_FP)

# Map: raw question text → integer ID (row index in question_meta.csv)
_qid_of = {q: i for i, q in enumerate(_meta["question"].astype(str).tolist())}
if not _qid_of:
    raise RuntimeError("Loaded question_meta.csv but found no entries.")

# Cleaned texts aligned by index
_CLEAN_FP = _PROCESSED / "clean_questions.npy"
if not _CLEAN_FP.exists():
    raise FileNotFoundError(
        f"Missing '{_CLEAN_FP}'. Run preprocessing to generate clean_questions.npy."
    )
_clean_qs = np.load(_CLEAN_FP, allow_pickle=True).tolist()
if len(_clean_qs) != len(_meta):
    raise RuntimeError(
        f"Length mismatch: clean_questions.npy has {len(_clean_qs)}, "
        f"but question_meta.csv has {len(_meta)} rows."
    )

# ─────────────────────────────────────────────────────────────────────────────
# 2) HELPER: CONVERT RAW PAIRS → DataFrame WITH qid1/qid2
# ─────────────────────────────────────────────────────────────────────────────

def _pairs_to_df(pairs: List[Pair]) -> pd.DataFrame:
    """
    Given a list of (q1, q2) raw-text pairs, return a DataFrame with:
      ['question1','question2','qid1','qid2'].
    Raises KeyError if any question not found in question_meta.csv.
    """
    if not pairs:
        return pd.DataFrame(columns=["question1", "question2", "qid1", "qid2"])

    q1_texts, q2_texts = zip(*pairs)
    df = pd.DataFrame({"question1": q1_texts, "question2": q2_texts})
    try:
        df = df.assign(
            qid1=lambda d: d.question1.map(_qid_of).astype(int),
            qid2=lambda d: d.question2.map(_qid_of).astype(int),
        )
    except Exception as e:
        raise KeyError("One or more questions not found in question_meta.csv.") from e
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3) ABSTRACT BASE CLASS: FeatureExpert
# ─────────────────────────────────────────────────────────────────────────────

class FeatureExpert(BaseExpert):
    """
    Base class for any sklearn-style model using IPCA- or UMAP-reduced features
    (for a specified SBERT-dim track). Subclasses must override:
        _new_model() → returns an unfitted sklearn estimator.

    Initialization:
      FeatureExpert(dim=384, reduction="ipca", n_components=None) sets:
        self.dim = 384
        self.reduction = "ipca"         # either "ipca" or "umap"
        self.n_components = None        # only used if reduction="umap"
        self.model_path = Path("../models/custom/<expert_name>_384_ipca.pkl")
        Expects precomputed files under data/processed/:
          • IPCA tracks:  X_train_384_ipca.npy, X_valid_384_ipca.npy, X_test_384_ipca.npy
          • UMAP tracks:  X_train_384_umap{n}.npy, X_valid_384_umap{n}.npy, X_test_384_umap{n}.npy
    """

    def __init__(
        self,
        dim: int,
        *,
        reduction: str = "ipca",
        n_components: int | None = None
    ):
        self.dim = dim
        self.reduction = reduction.lower()
        if self.reduction not in {"ipca", "umap"}:
            raise ValueError("`reduction` must be either 'ipca' or 'umap'")

        # If UMAP, n_components must be provided
        if self.reduction == "umap" and (n_components is None or n_components <= 0):
            raise ValueError("For 'umap' reduction you must specify a positive n_components")
        self.n_components = n_components

        # Ensure SBERT embeddings for this dim exist (needed by build_features)
        self._emb_fp = _PROCESSED / f"question_embeddings_{dim}.npy"
        if not self._emb_fp.exists():
            raise FileNotFoundError(
                f"Missing '{self._emb_fp}'. Run preprocessing to generate SBERT embeddings."
            )

        # Load or create underlying classifier
        base = Path("../models/custom")
        base.mkdir(parents=True, exist_ok=True)
        name = self.__class__.__name__

        # Append reducer to model filename, e.g. "_384_ipca" or "_768_umap10"
        if self.reduction == "ipca":
            suffix = f"{dim}_ipca"
        else:
            suffix = f"{dim}_umap{self.n_components}"
        self.model_path = base / f"{name}_{suffix}.pkl"

        self._load()

    def _load(self):
        """
        Attempt to load a pickled estimator. If not found, instantiate a fresh one.
        """
        if hasattr(self, "clf") and self.clf is not None:
            return  # already loaded
        if self.model_path.exists():
            with open(self.model_path, "rb") as f:
                self.clf = pickle.load(f)
            log_event(
                LogKind.MODEL,
                model=f"{self.__class__.__name__}_{self.reduction}_{self.dim}",
                phase="load",
                src_dims=self.dim,
            )
        else:
            self.clf = self._new_model()

    def _predict_impl(self, pairs: List[Pair]) -> np.ndarray:
        """
        Given list of (q1, q2) pairs, build reduced features on-the-fly (with caching),
        then return classifier.predict_proba(X)[:,1].
        """
        pair_df = _pairs_to_df(pairs)

        # Dynamically compute features for any arbitrary pair list
        X = build_features(
            pair_df=pair_df,
            clean_questions=_clean_qs,
            meta_df=_meta,
            embedding_path=str(self._emb_fp),
            cache_dir=str(Path(f"../models/features_{self.dim}")),
            cross_cache=None,
            fit_pca=False,  # Do not refit PCA on inference
            features_cache=None,
            reduction=self.reduction,
            n_components=self.n_components,
        )

        probs = self.clf.predict_proba(X)[:, 1].astype("float32")
        return probs


    def fit(self, train_df: pd.DataFrame, y: np.ndarray):
        """
        Fit the underlying classifier on training data for this dim+reduction track.

        1) If X_train_{dim}_{reduction}.npy exists, load it (reduced).
        2) Else, generate via build_features(..., fit_pca=True) on TRAIN split.
        3) Fit classifier and pickle.
        """
        # Decide train feature‐filename
        if self.reduction == "ipca":
            x_train_fp = _PROCESSED / f"X_train_{self.dim}_ipca.npy"
        else:
            x_train_fp = _PROCESSED / f"X_train_{self.dim}_umap{self.n_components}.npy"

        if x_train_fp.exists():
            X_train = np.load(x_train_fp)
            log_event(
                LogKind.MODEL,
                model=f"{self.__class__.__name__}_{self.reduction}_{self.dim}",
                phase="load_features",
                src_dims=X_train.shape[1],
            )
        else:
            # Build reduced features for TRAIN split
            X_train = build_features(
                pair_df         = train_df,
                clean_questions = _clean_qs,
                meta_df         = _meta,
                embedding_path  = str(self._emb_fp),
                cache_dir       = str(Path(f"../models/features_{self.dim}")),
                cross_cache     = None,
                fit_pca         = True,
                features_cache  = str(_PROCESSED / f"train_raw_{self.dim}.npy"),
                reduction       = self.reduction,
                n_components    = self.n_components,
            )
            np.save(x_train_fp, X_train)
            log_event(
                LogKind.MODEL,
                model=f"{self.__class__.__name__}_{self.reduction}_{self.dim}",
                phase="fit_features",
                src_dims=X_train.shape[1],
            )

        # Fit the classifier on those reduced features
        t0 = time.time()
        self.clf = self._new_model()
        self.clf.fit(X_train, y)
        t1 = time.time()
        elapsed = round(t1 - t0, 2)
        log_event(
            LogKind.MODEL,
            model=f"{self.__class__.__name__}_{self.reduction}_{self.dim}",
            phase="fit",
            seconds=elapsed,
            src_dims=X_train.shape[1],
        )

        # Persist the model weights to disk
        with open(self.model_path, "wb") as f:
            pickle.dump(self.clf, f)

    def predict_prob(self, pairs: List[Pair]) -> np.ndarray:
        """
        Public API: returns a numpy array of duplicate probabilities for each pair.
        """
        return self._predict_impl(pairs)

    def _new_model(self):
        """
        Subclasses must override to return an unfitted sklearn-compatible estimator.
        """
        raise NotImplementedError("Subclasses must override _new_model().")


# ─────────────────────────────────────────────────────────────────────────────
# 4) CONCRETE FEATURE EXPERT IMPLEMENTATIONS
# ─────────────────────────────────────────────────────────────────────────────

class LRFeatureExpert(FeatureExpert):
    """
    Logistic Regression on reduced features.
    """
    def __init__(self, dim: int = 384, *, reduction: str = "ipca", n_components: int | None = None):
        super().__init__(dim, reduction=reduction, n_components=n_components)

    def _new_model(self):
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            max_iter=1000,
        )


class XGBFeatureExpert(FeatureExpert):
    """
    XGBoost on reduced features.
    """
    def __init__(self, dim: int = 384, *, reduction: str = "ipca", n_components: int | None = None):
        super().__init__(dim, reduction=reduction, n_components=n_components)

    def _new_model(self):
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=7,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            n_jobs=-1,
            use_label_encoder=False,
            random_state=13,
        )


class LGBMFeatureExpert(FeatureExpert):
    """
    LightGBM on reduced features.
    """
    def __init__(self, dim: int = 384, *, reduction: str = "ipca", n_components: int | None = None):
        super().__init__(dim, reduction=reduction, n_components=n_components)

    def _new_model(self):
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=64,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary",
            random_state=13,
        )


class KNNFeatureExpert(FeatureExpert):
    """
    K-Nearest Neighbors on reduced features.
    """
    def __init__(self, dim: int = 384, *, reduction: str = "ipca", n_components: int | None = None):
        super().__init__(dim, reduction=reduction, n_components=n_components)

    def _new_model(self):
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(
            n_neighbors=10,
            weights="distance",
            n_jobs=-1,
        )


class RFFeatureExpert(FeatureExpert):
    """
    Random Forest on reduced features.
    """
    def __init__(self, dim: int = 384, *, reduction: str = "ipca", n_components: int | None = None):
        super().__init__(dim, reduction=reduction, n_components=n_components)

    def _new_model(self):
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=13
        )


class SVMFeatureExpert(FeatureExpert):
    """
    Linear SVM with probability calibration on reduced features.
    """
    def __init__(self, dim: int = 384, *, reduction: str = "ipca", n_components: int | None = None):
        super().__init__(dim, reduction=reduction, n_components=n_components)

    def _new_model(self):
        from sklearn.svm import LinearSVC
        from sklearn.calibration import CalibratedClassifierCV

        base_svc = LinearSVC(
            C=1.0,
            max_iter=5_000,
            random_state=13,
        )
        return CalibratedClassifierCV(
            estimator=base_svc,
            cv=3,
            method="sigmoid",
        )