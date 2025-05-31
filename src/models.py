# src/models.py
# -----------------------------------------------------------
"""
Mixture‐of‐Experts for Quora Duplicate Question Pairs (QQP).

Current experts (if available):
    • BertExpert      (textattack/bert-base-uncased-QQP)
    • RobertaExpert   (howey/roberta-large-qqp)
    • XLNetExpert     (textattack/xlnet-base-cased-QQP)  (requires sentencepiece)
    • SBertExpert     (sentence-transformers/all-MiniLM-L6-v2 + LogisticRegression)
    • CrossEncExpert  (cross-encoder/quora-roberta-large)

A lightweight gating network (Linear → Softmax) predicts weights w ∈ ℝᵏ (k = # of
chosen experts) per (q1, q2) pair; then the MoE output probability is ∑ wᵢ · pᵢ.

Public API
----------
    BertExpert, RobertaExpert, XLNetExpert (optional), SBertExpert, CrossEncExpert
        → .predict_prob(list[(q1, q2)]) → np.ndarray of duplicate‐probabilities ∈ [0,1]

    MoEClassifier(experts, lr, epochs)
        → .fit(pairs, y)
        → .predict_prob(pairs)
        → .evaluate(pairs, y)   (binary log‐loss)

Everything is implemented in PyTorch (gate) + HuggingFace Transformers + scikit‐learn.
"""

from __future__ import annotations
import os
import pickle
import logging
import warnings
from pathlib import Path
from typing import List, Tuple, Sequence

import numpy as np
import torch
import torch.nn as nn

# Silence HuggingFace INFO logs
logging.getLogger("transformers").setLevel(logging.ERROR)

# -----------------------------------------------------------------------------
# helpers ---------------------------------------------------------------------
# -----------------------------------------------------------------------------
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_CACHE  = Path("models/.hf_cache")
_CACHE.mkdir(parents=True, exist_ok=True)

Pair = Tuple[str, str]  # a (question1, question2) tuple

def _batchify(x: Sequence, batch_size: int = 32):
    """
    Yields consecutive slices of `x` of length ≤ batch_size.
    """
    for i in range(0, len(x), batch_size):
        yield x[i : i + batch_size]

# -----------------------------------------------------------------------------
# Abstract Expert -------------------------------------------------------------
# -----------------------------------------------------------------------------
class BaseExpert:
    """
    Base class for all experts. Subclasses must implement:
      - _load()            (load model/tokenizer into memory)
      - _predict_impl()    (given list[(q1,q2)], return np.ndarray(probs))
    """

    model_name: str
    batch_size: int = 32

    def __init__(self):
        self._loaded = False
        self._tokenizer = None
        self._model = None
        self._load()
        self._loaded = True

    def predict_prob(self, pairs: List[Pair]) -> np.ndarray:
        """
        Input : list[(q1, q2)] of raw strings
        Output: np.ndarray of shape (len(pairs),), dtype=float32, values in [0,1]
        """
        self._ensure_loaded()
        return self._predict_impl(pairs).astype("float32")

    def _load(self):
        """
        Subclasses must override this to load any necessary weights/tokenizers.
        """
        raise NotImplementedError

    def _predict_impl(self, pairs: List[Pair]) -> np.ndarray:
        """
        Subclasses must override this to implement inference logic.
        """
        raise NotImplementedError

    def _ensure_loaded(self):
        if not self._loaded:
            raise RuntimeError(f"{self.__class__.__name__} failed to load properly")

# -----------------------------------------------------------------------------
# 1) HF Sequence‐classification experts (BERT / RoBERTa / optional XLNet) -------
# -----------------------------------------------------------------------------
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class _HFAutoExpert(BaseExpert):
    """
    Shared loader for any HuggingFace checkpoint with a 2‐class head.
    Subclasses must set .model_name and .batch_size.
    """

    def _load(self):
        # Force Transformers to use our local cache directory
        os.environ["TRANSFORMERS_CACHE"] = str(_CACHE)

        # If XLNetExpert (model_name contains "xlnet"), sentencepiece is required.
        if "xlnet" in self.model_name.lower():
            try:
                import sentencepiece  # noqa: F401
            except ModuleNotFoundError:
                raise RuntimeError(
                    "XLNet tokenizer needs the `sentencepiece` package.\n"
                    "Install via:   pip install sentencepiece"
                )

        # Always load the **slow** tokenizer (avoid tiktoken/sentencepiece converters)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=False
        )
        self._model = (
            AutoModelForSequenceClassification.from_pretrained(self.model_name)
            .to(_DEVICE)
            .eval()
        )

    @torch.no_grad()
    def _predict_impl(self, pairs: List[Pair]) -> np.ndarray:
        """
        Tokenizes each batch of (q1,q2), runs the model, and returns class‐1 prob.
        """
        probs: List[float] = []
        for batch in _batchify(pairs, self.batch_size):
            toks = self._tokenizer(
                [p[0] for p in batch],
                [p[1] for p in batch],
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(_DEVICE)
            logits = self._model(**toks).logits  # (B, 2)
            # P(class=1) = duplicate probability
            p = torch.softmax(logits, dim=1)[:, 1]
            probs.extend(p.cpu().tolist())
        return np.array(probs, dtype="float32")


class BertExpert(_HFAutoExpert):
    """
    BERT fine‐tuned on QQP via TextAttack.
    (Model card: textattack/bert-base-uncased-QQP)
    """
    model_name = "textattack/bert-base-uncased-QQP"
    batch_size = 32


class RobertaExpert(_HFAutoExpert):
    """
    RoBERTa large fine‐tuned on QQP.
    (Model card: howey/roberta-large-qqp)
    """
    model_name = "howey/roberta-large-qqp"
    batch_size = 16  # larger model → smaller batch


class XLNetExpert(_HFAutoExpert):
    """
    XLNet base fine‐tuned on QQP (requires sentencepiece).
    (Model card: textattack/xlnet-base-cased-QQP)
    """
    model_name = "textattack/xlnet-base-cased-QQP"
    batch_size = 16

# -----------------------------------------------------------------------------
# 2) SBertExpert (MiniLM‐L6 + LogisticRegression on pairwise features) ---------
# -----------------------------------------------------------------------------
class SBertExpert(BaseExpert):
    """
    Uses MiniLM‐L6 sentence embeddings + a scikit‐learn LogisticRegression
    on [|u - v|, u · v].

    - On __init__, it must be provided:
        • emb_path:   path to question_embeddings.npy (numpy float32 array, shape=(n_questions, dim))
        • lr_path:    path to sbert_lr.pkl (pickle of a trained LogisticRegression)

    - If `sbert_lr.pkl` is missing, user must call .fit(...) once.

    Public methods:
        • .fit(qids1, qids2, y) → trains LogisticRegression on 80% train only
        • .predict_prob(pairs) → uses pretrained MiniLM to re‐encode fresh text & runs the LR
    """

    def __init__(
        self,
        emb_path: str,
        lr_path: str,
    ):
        """
        emb_path: file → 'question_embeddings.npy' (float32 array shape=(n_questions, 384))
        lr_path : file → 'sbert_lr.pkl'
        """
        self.emb_path = emb_path
        self.lr_path = Path(lr_path)
        super().__init__()

    def _load(self):
        from sentence_transformers import SentenceTransformer

        # 1) Load SentenceTransformer for on‐the‐fly encoding
        self._st_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", device=_DEVICE
        )
        # 2) Memory‐map precomputed embeddings
        self._emb = np.load(self.emb_path, mmap_mode="r")
        # 3) Load or set LogisticRegression to None
        if self.lr_path.exists():
            with open(self.lr_path, "rb") as f:
                self.lr = pickle.load(f)
        else:
            self.lr = None

    def fit(self, qids1: np.ndarray, qids2: np.ndarray, y: np.ndarray):
        """
        Train scikit‐learn LogisticRegression on:
          u = emb[qids1], v = emb[qids2]
          X = [|u - v|, u * v]  (shape: n_samples × (2 * dim))
        Then pickle‐dump to `sbert_lr.pkl`.
        """
        from sklearn.linear_model import LogisticRegression

        u = self._emb[qids1]  # (n_samples, 384)
        v = self._emb[qids2]  # (n_samples, 384)
        X = np.hstack([np.abs(u - v), u * v])  # (n_samples, 768)
        self.lr = LogisticRegression(max_iter=1000)
        self.lr.fit(X, y)
        with open(self.lr_path, "wb") as f:
            pickle.dump(self.lr, f)

    @torch.no_grad()
    def _predict_impl(self, pairs: List[Pair]) -> np.ndarray:
        if self.lr is None:
            raise RuntimeError(
                "SBertExpert: trained LR not found. Call .fit(...) first or ensure sbert_lr.pkl exists."
            )

        # 1) On‐the‐fly encode fresh questions (GPU if available):
        u = self._st_model.encode(
            [p[0] for p in pairs],
            batch_size=256,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        v = self._st_model.encode(
            [p[1] for p in pairs],
            batch_size=256,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        # 2) Build pairwise features and predict
        X = np.hstack([np.abs(u - v), u * v])  # (n_pairs, 768)
        prob = self.lr.predict_proba(X)[:, 1].astype("float32")
        return prob

# -----------------------------------------------------------------------------
# 3) Cross‐Encoder expert (already returns [0,1]) ----------------------------
# -----------------------------------------------------------------------------
from sentence_transformers import CrossEncoder

class CrossEncExpert(BaseExpert):
    """
    Uses “cross-encoder/quora-roberta-large” with Sigmoid activation to produce
    a P(duplicate) ∈ [0,1] for each (q1, q2). No extra post-processing.
    """
    model_name = "cross-encoder/quora-roberta-large"
    batch_size = 32

    def _load(self):
        os.environ["TRANSFORMERS_CACHE"] = str(_CACHE)
        # CrossEncoder with Sigmoid → outputs float32 array shape=(n_pairs,)
        self._model = CrossEncoder(
            self.model_name,
            device=_DEVICE,
            activation_fn=torch.nn.Sigmoid()
        )

    @torch.no_grad()
    def _predict_impl(self, pairs: List[Pair]) -> np.ndarray:
        probs = self._model.predict(pairs, batch_size=self.batch_size)
        return probs.astype("float32")

# -----------------------------------------------------------------------------
# 4) Mixture‐of‐Experts (gate + weighted blend) --------------------------------
# -----------------------------------------------------------------------------
class _GateNet(nn.Module):
    """Simple softmax gate over N experts → weights sum to 1."""

    def __init__(self, n_exp: int):
        super().__init__()
        self.lin = nn.Linear(n_exp, n_exp, bias=True)

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """
        p: (B, N) raw expert probabilities
        returns: (B, N) weights (softmaxed across dim=1)
        """
        w = self.lin(p)  # (B, N)
        return torch.softmax(w, dim=1)

class MoEClassifier:
    """
    Given K “frozen” experts, train a gating network to blend their outputs.

    Parameters
    ----------
    experts : List[BaseExpert]   (e.g., [BertExpert(), RobertaExpert(), SBertExpert(...), CrossEncExpert(), …])
    lr      : learning rate for gate’s parameters
    epochs  : number of epochs to train the gate

    Methods
    -------
      .fit(pairs, y)         – train gate on (q1, q2) pairs with binary labels y
      .predict_prob(pairs)   – return blended probabilities for new pairs
      .evaluate(pairs, y)    – return binary log‐loss on given dataset
    """

    def __init__(self, experts: List[BaseExpert], lr: float = 1e-2, epochs: int = 3):
        self.experts = experts
        self.epochs = epochs
        self.gate = _GateNet(len(experts)).to(_DEVICE)
        self.opt = torch.optim.Adam(self.gate.parameters(), lr=lr)
        self.loss_fn = nn.BCELoss()

    def fit(self, pairs: List[Pair], y: np.ndarray):
        """
        Train the gate on top of frozen experts.
        pairs: list[(q1, q2)]
        y    : np.ndarray binary labels (0/1)
        """
        y_t = torch.tensor(y, dtype=torch.float32).to(_DEVICE)
        B = 1024  # gate batch size

        for epoch in range(1, self.epochs + 1):
            perm = np.random.permutation(len(pairs))
            epoch_loss = 0.0

            for batch_idx in range(0, len(pairs), B):
                idx = perm[batch_idx : batch_idx + B]
                batch_pairs = [pairs[i] for i in idx]
                targets = y_t[idx]

                # 1) Collect frozen expert probabilities (no grad)
                with torch.no_grad():
                    probs = np.column_stack(
                        [exp.predict_prob(batch_pairs) for exp in self.experts]
                    )  # shape=(batch_size, K)
                    probs_t = torch.tensor(probs, dtype=torch.float32).to(_DEVICE)

                # 2) Gate forward
                weights = self.gate(probs_t)
                blended = (weights * probs_t).sum(dim=1)  # (batch_size,)
                loss = self.loss_fn(blended, targets)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                epoch_loss = loss.item()

            print(f"Epoch {epoch}/{self.epochs}  ·  loss {epoch_loss:.4f}")

    @torch.no_grad()
    def predict_prob(self, pairs: List[Pair]) -> np.ndarray:
        """
        For new pairs, return blended duplicate probabilities (shape=(n_pairs,)).
        """
        probs = np.column_stack([exp.predict_prob(pairs) for exp in self.experts])
        probs_t = torch.tensor(probs, dtype=torch.float32).to(_DEVICE)
        weights = self.gate(probs_t)            # (n_pairs, K)
        blended = (weights * probs_t).sum(dim=1) # (n_pairs,)
        return blended.cpu().numpy().astype("float32")

    def evaluate(self, pairs: List[Pair], y: np.ndarray) -> float:
        """
        Compute binary log‐loss between labels y and predicted probabilities.
        """
        from sklearn.metrics import log_loss
        p = self.predict_prob(pairs)
        return log_loss(y, p)

# -----------------------------------------------------------------------------
# 5) Convenience factory to instantiate all desired experts ---------------
# -----------------------------------------------------------------------------
def default_experts(
    emb_path: str       = "data/processed/question_embeddings.npy",
    lr_path: str        = "models/sbert_lr.pkl",
    embed_lr_ready: bool = True
) -> List[BaseExpert]:
    """
    Returns a list of expert instances in this order:
      [BertExpert, RobertaExpert, (XLNetExpert if available), SBertExpert, CrossEncExpert]

    Arguments:
      emb_path        : string path → precomputed question_embeddings.npy
      lr_path         : string path → sbert_lr.pkl (LogisticRegression)
      embed_lr_ready  : if False, SBertExpert will not error if no lr_path exists; you can call .fit() manually

    Example:
        experts = default_experts(
            emb_path="data/processed/question_embeddings.npy",
            lr_path="models/sbert_lr.pkl",
            embed_lr_ready=False
        )
    """
    exps: List[BaseExpert] = []

    # 1) BERT
    exps.append(BertExpert())

    # 2) RoBERTa
    exps.append(RobertaExpert())

    # 3) XLNet (optional)
    try:
        xl = XLNetExpert()
        exps.append(xl)
    except RuntimeError as e:
        warnings.warn(
            "Skipping XLNetExpert (sentencepiece not installed). "
            "To enable XLNetExpert, install sentencepiece:\n"
            "    pip install sentencepiece\n"
            f"Details: {e}"
        )

    # 4) SBERT (+ LR layer)
    sbert = SBertExpert(
        emb_path=emb_path,
        lr_path=lr_path
    )
    if not sbert.lr_path.exists() and not embed_lr_ready:
        # User must call sbert.fit(...) manually later
        pass
    exps.append(sbert)

    # 5) Cross‐Encoder
    exps.append(CrossEncExpert())

    return exps

# -----------------------------------------------------------------------------
# --- append to the end of src/models.py -------------------------------------
# -----------------------------------------------------------------------------
import joblib
import hashlib
import json

def _hash_pairs(pairs: List[Pair]) -> str:
    """SHA1 over the first/last 50 pairs so the cache filename is deterministic."""
    h = hashlib.sha1()
    sample = pairs[:50] + pairs[-50:]
    h.update(json.dumps(sample).encode())
    return h.hexdigest()[:8]

def get_predictions(
    experts   : List[BaseExpert],
    pairs     : List[Pair],
    split_tag : str,
    cache_dir : Path | str = "models/pred_cache"
) -> np.ndarray:
    """
    Forward-passes every expert exactly once on `pairs`, caches each column to
    <cache_dir>/<split_tag>_<expert_name>.npy, and returns a numpy matrix of
    shape (n_pairs,  n_experts).

    Next time you call it on the same split, probabilities are mmap-loaded – zero cost.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cols = []
    for exp in experts:
        fname = cache_dir / f"{split_tag}_{exp.__class__.__name__}.npy"
        if fname.exists():
            p = np.load(fname, mmap_mode="r")
        else:
            p = exp.predict_prob(pairs)
            np.save(fname, p)
        cols.append(p)
    return np.column_stack(cols).astype("float32")

# -----------------------------------------------------------------------------
# 6)  Convenience helpers for serialising the gate ----------------------------
# -----------------------------------------------------------------------------
def _subset_key(experts: list[BaseExpert]) -> str:
    """
    Canonical string that identifies a subset, e.g.
        "BertExpert+RobertaExpert+CrossEncExpert"
    The order inside `experts` is preserved.
    """
    return "+".join(e.__class__.__name__ for e in experts)

def save_gate(moe: "MoEClassifier", path: Path | str):
    """
    Persist the *trained* gate parameters only.
    """
    torch.save(moe.gate.state_dict(), str(path))

def load_gate(experts: list[BaseExpert], path: Path | str) -> "MoEClassifier":
    """
    Instantiate a MoEClassifier with the given experts and load the stored
    gate weights.
    """
    mdl = MoEClassifier(experts, lr=1.0, epochs=0)   # lr/epochs are dummies
    mdl.gate.load_state_dict(torch.load(str(path), map_location=_DEVICE))
    mdl.gate.eval()
    return mdl