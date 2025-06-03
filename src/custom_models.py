# ────────────────────────────────────────────────────────────────
# src/custom_models.py
# ----------------------------------------------------------------
"""
“Feature-based” experts built on our PCA-reduced feature matrix
(see notebooks/03_feature_engineering.ipynb + PCA step).

Each Expert implements the same .predict_prob(pairs) API as our Hugging-Face
models, so they can be blended seamlessly in the Mixture-of-Experts gate.

Ready-made experts:
  * LRFeatureExpert        – LogisticRegression
  * XGBFeatureExpert       – XGBoost (requires `pip install xgboost`)
  * LGBMFeatureExpert      – LightGBM (requires `pip install lightgbm`)
  * KNNFeatureExpert       – K-Nearest Neighbors (scikit-learn)
  * RFFeatureExpert        – Random Forest classifier (scikit-learn)
  * SVMFeatureExpert       – SVM with probability estimates (scikit-learn)

Usage example:
    from src.custom_models import LRFeatureExpert

    lr = LRFeatureExpert()
    lr.fit(train_df, y_train)          # trains on precomputed X_train.npy if available
    p  = lr.predict_prob(valid_pairs)   # returns P(duplicate) for each pair
"""

from __future__ import annotations
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.features import build_features
from src.pretrained_models import BaseExpert  # reuse the abstract BaseExpert interface

Pair = Tuple[str, str]  # alias for readability: a (question1, question2) tuple


# ─────────────────────────────────────────────────────────────────────────────
# 1) CONSTANTS & PRE-LOADED ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────

# We assume the working directory is the project root so that
# “data/processed” lives at project_root/data/processed.
_PROCESSED = Path("../data/processed")
if not _PROCESSED.exists():
    raise FileNotFoundError(
        f"Expected directory ‘{_PROCESSED}’ does not exist. "
        "Run 02_preprocessing.ipynb first to create per-question artifacts."
    )

# Load question_meta.csv → must contain columns: ["question", "clean", "len", "words"].
_meta_fp = _PROCESSED / "question_meta.csv"
if not _meta_fp.exists():
    raise FileNotFoundError(
        f"Missing {_meta_fp}. Run 02_preprocessing.ipynb to generate question_meta.csv."
    )
_meta = pd.read_csv(_meta_fp)

# Build a mapping: raw question → integer ID (row index in question_meta.csv)
_qid_of = {q: i for i, q in enumerate(_meta["question"].tolist())}
if not _qid_of:
    raise RuntimeError("Loaded question_meta.csv but found no questions!")

# Load the cleaned text array (index-aligned with question_meta.csv)
_clean_fp = _PROCESSED / "clean_questions.npy"
if not _clean_fp.exists():
    raise FileNotFoundError(
        f"Missing {_clean_fp}. Run 02_preprocessing.ipynb to generate clean_questions.npy."
    )
_clean_qs = np.load(_clean_fp, allow_pickle=True).tolist()
if len(_clean_qs) != len(_meta):
    raise RuntimeError(
        f"Length mismatch: clean_questions.npy has {len(_clean_qs)} entries, "
        f"but question_meta.csv has {len(_meta)} rows."
    )

# Path to the canonical question_embeddings.npy (should be 768-dim per row)
_emb_fp = _PROCESSED / "question_embeddings.npy"
if not _emb_fp.exists():
    raise FileNotFoundError(
        f"Missing {_emb_fp}. Run 02_preprocessing.ipynb (with out_fp) to create question_embeddings.npy."
    )

# Path to precomputed PCA-reduced feature files:
_X_TRAIN_FP = _PROCESSED / "X_train.npy"   # shape = (n_train_pairs, n_pca_dims)
_X_VALID_FP = _PROCESSED / "X_valid.npy"
_X_TEST_FP  = _PROCESSED / "X_test.npy"


# ─────────────────────────────────────────────────────────────────────────────
# 2) HELPER: CONVERT RAW PAIRS → DataFrame WITH qid1/qid2
# ─────────────────────────────────────────────────────────────────────────────

def _pairs_to_df(pairs: List[Pair]) -> pd.DataFrame:
    """
    Given a list of (q1, q2) raw-text pairs, return a DataFrame with columns:
      ["question1", "question2", "qid1", "qid2"]
    where qid1/qid2 are integer indices into `question_meta.csv`.
    Raises KeyError if any question is not found in the mapping.
    """
    if not pairs:
        # Return an empty DataFrame with the expected columns
        return pd.DataFrame(columns=["question1", "question2", "qid1", "qid2"])

    q1_texts, q2_texts = zip(*pairs)
    df = pd.DataFrame({"question1": q1_texts, "question2": q2_texts})

    try:
        df = df.assign(
            qid1=lambda d: d.question1.map(_qid_of).astype(int),
            qid2=lambda d: d.question2.map(_qid_of).astype(int),
        )
    except Exception as e:
        raise KeyError(
            "One or more questions in `pairs` were not found in question_meta.csv."
        ) from e

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3) ABSTRACT BASE CLASS: FeatureExpert
# ─────────────────────────────────────────────────────────────────────────────

class FeatureExpert(BaseExpert):
    """
    Base class for any scikit-learn–style model that uses the precomputed,
    PCA-reduced feature matrix (instead of raw text). Subclasses must override:
        _new_model() → an unfitted sklearn-compatible estimator.

    Implementation details:
      * __init__ (inherited from BaseExpert) calls self._load().
      * _load(): tries to unpickle self.model_path; if not found, calls self._new_model().
      * _predict_impl(pairs): 
          1) converts raw-text pairs → DataFrame with qid1, qid2
          2) calls build_features(...) → (n_pairs, 3 598) numpy array
             followed by PCA internally
          3) runs self.clf.predict_proba(X_pca)[:, 1] → duplicate probabilities
      * fit(train_df, y): 
          1) tries to load X_train.npy (already PCA-reduced) from disk; 
             if missing, calls build_features(...) → PCA internally
          2) trains self.clf = self._new_model().fit(X_train, y)
          3) pickles the fitted model under self.model_path
    """

    model_path: Path  # where the fitted estimator is pickled

    def _load(self):
        """
        Attempt to load a pre-fitted model from disk. If not found,
        instantiate a new, unfitted estimator via self._new_model().
        """
        if getattr(self, "model_path", None) is None:
            raise ValueError(
                f"{self.__class__.__name__} must define a class-level `model_path: Path`."
            )

        if self.model_path.exists():
            with open(self.model_path, "rb") as f:
                self.clf = pickle.load(f)
        else:
            self.clf = self._new_model()

    def _predict_impl(self, pairs: List[Pair]) -> np.ndarray:
        """
        Given a list of (q1, q2) pairs, build the engineered feature matrix
        (3 598 dims) and then apply PCA (internally, via build_features),
        returning a 1D numpy array of duplicate-probabilities via self.clf.
        """
        # 1) Convert raw pairs → DataFrame with qid1 & qid2
        pair_df = _pairs_to_df(pairs)

        # 2) Build the full feature matrix (shape: n_pairs × 3 598),
        #    then PCA-reduce internally (95% variance).
        X = build_features(
            pair_df         = pair_df,
            clean_questions = _clean_qs,
            meta_df         = _meta,
            embedding_path  = _emb_fp,
            cache_dir       = Path("../models/features"),  # TF-IDF & SVD pickles live here
            cross_cache     = None                        # compute cross-encoder on the fly
        )

        # 3) Use underlying classifier to produce P(duplicate)
        prob = self.clf.predict_proba(X)[:, 1].astype("float32")
        return prob

    def fit(self, train_df: pd.DataFrame, y: np.ndarray):
        """
        Fit the underlying classifier on *training* pairs.

        Strategy:
          1) If `data/processed/X_train.npy` exists, load it directly (already PCA-reduced).
          2) Otherwise, call build_features(...) → PCA internally to produce X_train.
          3) Train a fresh estimator from self._new_model() on X_train, y.
          4) Pickle the fitted model under self.model_path.
        """
        # 1) Try to load precomputed PCA-reduced training set:
        if _X_TRAIN_FP.exists():
            X_train = np.load(_X_TRAIN_FP)
        else:
            # If not found, build from scratch (3 598 dims → PCA inside build_features)
            X_train = build_features(
                pair_df         = train_df,
                clean_questions = _clean_qs,
                meta_df         = _meta,
                embedding_path  = _emb_fp,
                cache_dir       = Path("../models/features"),
                cross_cache     = None
            )

        # 2) Instantiate & fit
        self.clf = self._new_model()
        self.clf.fit(X_train, y)

        # 3) Ensure “models/custom/” directory exists before pickling
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump(self.clf, f)

    def _new_model(self):
        """
        Must be overridden by subclasses to return an
        unfitted sklearn–compatible estimator (with predict_proba).
        """
        raise NotImplementedError("Subclasses must override _new_model().")


# ─────────────────────────────────────────────────────────────────────────────
# 4) CONCRETE FEATURE EXPERT IMPLEMENTATIONS
# ─────────────────────────────────────────────────────────────────────────────

class LRFeatureExpert(FeatureExpert):
    """
    3 598-dim → LogisticRegression (solver=liblinear, balanced class weights).
    """
    model_name = "LRFeatureExpert"
    model_path = Path("../models/custom/feature_lr.pkl")

    def _new_model(self):
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            max_iter=1000,
        )


class XGBFeatureExpert(FeatureExpert):
    """
    3 598-dim → XGBClassifier (requires `pip install xgboost`).
    """
    model_name = "XGBFeatureExpert"
    model_path = Path("../models/custom/feature_xgb.pkl")

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
        )


class LGBMFeatureExpert(FeatureExpert):
    """
    3 598-dim → LightGBM classifier (requires `pip install lightgbm`).
    """
    model_name = "LGBMFeatureExpert"
    model_path = Path("../models/custom/feature_lgbm.pkl")

    def _new_model(self):
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=64,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary"
        )


class KNNFeatureExpert(FeatureExpert):
    """
    3 598-dim → K-Nearest Neighbors classifier (scikit-learn).
    """
    model_name = "KNNFeatureExpert"
    model_path = Path("../models/custom/feature_knn.pkl")

    def _new_model(self):
        from sklearn.neighbors import KNeighborsClassifier
        # You can tune n_neighbors or other parameters as needed
        return KNeighborsClassifier(
            n_neighbors=10,
            weights="distance",
            n_jobs=-1,
        )


class RFFeatureExpert(FeatureExpert):
    """
    3 598-dim → Random Forest classifier (scikit-learn).
    """
    model_name = "RFFeatureExpert"
    model_path = Path("../models/custom/feature_rf.pkl")

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
    3 598-dim → linear SVM with probability calibration (fast on high-dim data).
    """
    model_name = "SVMFeatureExpert"
    model_path = Path("../models/custom/feature_svm.pkl")

    def _new_model(self):
        from sklearn.svm import LinearSVC
        from sklearn.calibration import CalibratedClassifierCV

        # 1) Use a linear SVM (LinearSVC) for speed on high-dimensional inputs.
        base_svc = LinearSVC(
            C=1.0,
            max_iter=5_000,       # increase if it doesn’t converge
            random_state=13,
        )
        # 2) Wrap in CalibratedClassifierCV to get .predict_proba(...)
        #    “cv=3” will hold out 1/3 of the train set for calibration.
        return CalibratedClassifierCV(
            estimator=base_svc,
            cv=3,
            method="sigmoid",     # faster & usually fine for our use case
        )