# src/pretrained_models.py
# -----------------------------------------------------------
"""
Mixture-of-Experts for Quora Duplicate Question Pairs (QQP).

Current experts (if available):
    - BertExpert        (textattack/bert-base-uncased-QQP)
    - RobertaExpert     (howey/roberta-large-qqp)
    - XLNetExpert       (textattack/xlnet-base-cased-QQP)  (requires sentencepiece)
    - QuoraDistilExpert (distilbert-base-nli-stsb-quora-ranking + LogisticRegression)
    - CrossEncExpert    (cross-encoder/quora-roberta-large)

A lightweight gating network (Linear -> Softmax) predicts weights w ∈ ℝᵏ per (q₁, q₂) pair;
then MoE output probability = ∑ᵢ wᵢ · pᵢ.

Public API
----------
    BertExpert, RobertaExpert, XLNetExpert (optional),
    QuoraDistilExpert, CrossEncExpert
       -> .predict_prob(list[(q₁, q₂)]) -> np.ndarray of duplicate-probabilities ∈ [0,1]

    MoEClassifier(experts, lr, epochs)
       -> .fit(pairs, y)
       -> .predict_prob(pairs)
       -> .evaluate(pairs, y)   (binary log-loss)

Everything is implemented in PyTorch (gate) + HuggingFace Transformers + scikit-learn.
"""

from __future__ import annotations
import os
import pickle
import logging
import warnings
import time
from pathlib import Path
from typing import List, Tuple, Sequence

import numpy as np
import torch
import torch.nn as nn

# Force Transformers to operate (if needed) in offline mode:
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
# Silence HuggingFace INFO logs:
logging.getLogger("transformers").setLevel(logging.WARN)

# -----------------------------------------------------------------------------
# helpers ---------------------------------------------------------------------
# -----------------------------------------------------------------------------
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_CACHE  = Path("../models/.hf_cache")  # local cache for HF models
_CACHE.mkdir(parents=True, exist_ok=True)

Pair = Tuple[str, str]  # a (question₁, question₂) tuple

def _batchify(x: Sequence, batch_size: int = 256):
    """
    Yield consecutive slices of `x` up to length `batch_size`.
    This controls how many pairs we send through the model in each forward pass.
    Increasing `batch_size` reduces the number of total forward passes,
    which speeds up inference (no retraining of expert weights is needed).
    """
    for i in range(0, len(x), batch_size):
        yield x[i : i + batch_size]

# -----------------------------------------------------------------------------
# Abstract Expert -------------------------------------------------------------
# -----------------------------------------------------------------------------
class BaseExpert:
    """
    Base class for all “expert” models. Subclasses must implement:
      - _load()            (load model/tokenizer into memory)
      - _predict_impl()    (given list[(q₁,q₂)], return np.ndarray(probs))

    Once instantiated, .predict_prob(...) will run _predict_impl in minibatches
    of size `self.batch_size` (default=256).
    """

    model_name: str
    batch_size: int = 256   # default batch size; higher GPUs -> larger batch

    def __init__(self):
        self._loaded = False
        self._tokenizer = None  # some experts will need a tokenizer
        self._model = None      # underlying HF or SBERT model
        self._load()
        self._loaded = True

    def predict_prob(self, pairs: List[Pair]) -> np.ndarray:
        """
        Input : list[(q₁, q₂)] (raw text strings)
        Output: 1-D numpy array of shape (len(pairs),), dtype=float32 ∈ [0,1].

        Internally, it calls `_predict_impl` in minibatches of size `batch_size`.
        """
        self._ensure_loaded()
        return self._predict_impl(pairs).astype("float32")

    def _load(self):
        """ Subclasses override to load tokenizer/model onto _DEVICE. """
        raise NotImplementedError

    def _predict_impl(self, pairs: List[Pair]) -> np.ndarray:
        """ Subclasses override to implement actual inference logic. """
        raise NotImplementedError

    def _ensure_loaded(self):
        if not self._loaded:
            raise RuntimeError(f"{self.__class__.__name__} failed to load properly")


# -----------------------------------------------------------------------------
# 1) HF Sequence-classification experts (BERT / RoBERTa / optional XLNet) -------
# -----------------------------------------------------------------------------
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class _HFAutoExpert(BaseExpert):
    """
    Loads any HuggingFace AutoModelForSequenceClassification with a 2-class head.
    Subclasses set `.model_name` and may override `batch_size` (default=256).
    """

    def _load(self):
        # Force Transformers to use our local cache directory
        os.environ["TRANSFORMERS_CACHE"] = str(_CACHE)

        # If it’s XLNet, we need sentencepiece installed
        if "xlnet" in self.model_name.lower():
            try:
                import sentencepiece  # noqa: F401
            except ModuleNotFoundError:
                raise RuntimeError(
                    "XLNet tokenizer needs the `sentencepiece` package. "
                    "Install via: pip install sentencepiece"
                )

        # Retry logic for loading the tokenizer + model
        last_exc: Exception | None = None
        for attempt in range(10):
            try:
                # Always load the *slow* tokenizer to avoid any auto-fast behavior
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    use_fast=False,
                    cache_dir=str(_CACHE),
                )
                self._model = (
                    AutoModelForSequenceClassification.from_pretrained(
                        self.model_name,
                        cache_dir=str(_CACHE),
                    )
                    .to(_DEVICE)
                    .eval()
                )
                last_exc = None
                break
            except Exception as e:
                # Only retry on network-related errors
                if isinstance(e, ConnectionError) or isinstance(e, OSError):
                    last_exc = e
                    if attempt < 9:
                        print(
                            f"[pretrained_models.py] Warning: failed to load HF model/tokenizer "
                            f"'{self.model_name}' (attempt {attempt+1}/10). Retrying in 2s..."
                        )
                        time.sleep(2)
                        continue
                # If it's not a ConnectionError or exhausted retries, re-raise immediately
                raise

        if last_exc is not None:
            # If we exited the loop still with an exception, re-raise it
            raise last_exc

    @torch.no_grad()
    def _predict_impl(self, pairs: List[Pair]) -> np.ndarray:
        """
        For each batch of up to `batch_size` pairs:
          1. Tokenize both questions (padding/truncation -> token IDs).
          2. Run through sequence-classification head.
          3. Softmax on logits to get P(class=1) = “duplicate” probability.
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

            logits = self._model(**toks).logits  # shape: (B, 2)
            p = torch.softmax(logits, dim=1)[:, 1]  # P(class=1) (“duplicate”)
            probs.extend(p.cpu().tolist())

        return np.array(probs, dtype="float32")


class BertExpert(_HFAutoExpert):
    """
    BERT fine-tuned on QQP via TextAttack:
      textattack/bert-base-uncased-QQP
    """
    model_name = "textattack/bert-base-uncased-QQP"
    batch_size = 256   # increased from 128 -> 256 for faster inference


class RobertaExpert(_HFAutoExpert):
    """
    RoBERTa large fine-tuned on QQP:
      howey/roberta-large-qqp
    """
    model_name = "howey/roberta-large-qqp"
    batch_size = 256   # increased from 128 -> 256 for faster inference


class XLNetExpert(_HFAutoExpert):
    """
    XLNet base fine-tuned on QQP (requires sentencepiece):
      textattack/xlnet-base-cased-QQP
    """
    model_name = "textattack/xlnet-base-cased-QQP"
    batch_size = 256   # increased from 128 -> 256 for faster inference


# -----------------------------------------------------------------------------
# 2) QuoraDistilExpert (Quora-trained DistilBERT + LogisticRegression) --------
# -----------------------------------------------------------------------------
class QuoraDistilExpert(BaseExpert):
    """
    Uses Quora-trained DistilBERT (“distilbert-base-nli-stsb-quora-ranking”) to
    produce 768-dim embeddings, then a scikit-learn LogisticRegression on [|u−v|, u·v].

    - On __init__, supply:
        - emb_path: path to question_embeddings.npy (shape=(n_questions, 768))
                    (these embeddings should correspond to the same Quora DistilBERT backbone)
        - lr_path : path to quoradistil_lr.pkl (pickle of a pretrained LogisticRegression on 1536-dim features)

    - If `quoradistil_lr.pkl` is missing, user must call .fit(...) once (using precomputed embeddings).

    Public methods:
        - .fit(qids1, qids2, y) -> trains an LR on (|u−v|, u·v) features using loaded embeddings
        - .predict_prob(pairs) -> encodes fresh questions “on-the-fly” with the Quora DistilBERT,
                                 builds features [|u−v|, u·v], then runs LR -> P(duplicate)
    """

    def __init__(
        self,
        emb_path: str,
        lr_path: str,
    ):
        """
        emb_path: file -> 'question_embeddings_768.npy' (float32 array shape=(n_questions, 768))
        lr_path : file -> 'quoradistil_lr.pkl'
        """
        self.emb_path = emb_path
        self.lr_path = Path(lr_path)
        super().__init__()

    def _load(self):
        from sentence_transformers import SentenceTransformer

        # 1) Load the Quora-trained DistilBERT for on-the-fly encoding, with retries
        last_exc: Exception | None = None
        for attempt in range(10):
            try:
                self._st_model = SentenceTransformer(
                    "sentence-transformers/distilbert-base-nli-stsb-quora-ranking",
                    device=_DEVICE,
                    cache_folder=str(_CACHE),
                )
                last_exc = None
                break
            except Exception as e:
                if isinstance(e, (ConnectionError, OSError)) and attempt < 9:
                    last_exc = e
                    print(
                        f"[pretrained_models.py] Warning: failed to load SentenceTransformer "
                        f"'distilbert-base-nli-stsb-quora-ranking' (attempt {attempt+1}/10). Retrying in 2s..."
                    )
                    time.sleep(2)
                    continue
                raise
        if last_exc is not None:
            raise last_exc

        # 2) Memory-map precomputed embeddings (768-dim) if available
        self._emb = np.load(self.emb_path, mmap_mode="r")

        # 3) Load or set LogisticRegression to None
        if self.lr_path.exists():
            with open(self.lr_path, "rb") as f:
                self.lr = pickle.load(f)
        else:
            self.lr = None

    def fit(self, qids1: np.ndarray, qids2: np.ndarray, y: np.ndarray):
        """
        Train scikit-learn LogisticRegression on:
          u = emb[qids1]    (shape=(n_samples, 768))
          v = emb[qids2]    (shape=(n_samples, 768))
          X = [|u−v|, u·v]   (shape=(n_samples, 1536))
        Then pickle-dump to `quoradistil_lr.pkl`.
        """
        from sklearn.linear_model import LogisticRegression

        u = self._emb[qids1]  # (n_samples, 768)
        v = self._emb[qids2]  # (n_samples, 768)
        X = np.hstack([np.abs(u - v), u * v])  # (n_samples, 1536)

        self.lr = LogisticRegression(max_iter=1000)
        self.lr.fit(X, y)

        with open(self.lr_path, "wb") as f:
            pickle.dump(self.lr, f)

    @torch.no_grad()
    def _predict_impl(self, pairs: List[Pair]) -> np.ndarray:
        if self.lr is None:
            raise RuntimeError(
                "QuoraDistilExpert: trained LR not found. Call .fit(...) first or ensure quoradistil_lr.pkl exists."
            )

        # 1) On-the-fly encode fresh questions with Quora DistilBERT
        #    We use batch_size=256 for faster encoding.
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
        X = np.hstack([np.abs(u - v), u * v])  # shape=(n_pairs, 1536)
        prob = self.lr.predict_proba(X)[:, 1].astype("float32")
        return prob


# -----------------------------------------------------------------------------
# 3) Cross-Encoder expert (already returns [0,1]) ----------------------------
# -----------------------------------------------------------------------------
from sentence_transformers import CrossEncoder

class CrossEncExpert(BaseExpert):
    """
    Uses “cross-encoder/quora-roberta-large” with Sigmoid activation to produce
    P(duplicate) ∈ [0,1] for each (q₁, q₂). No extra post-processing.
    """
    model_name = "cross-encoder/quora-roberta-large"
    batch_size = 256    # increased from 128 -> 256 for faster inference

    def _load(self):
        os.environ["TRANSFORMERS_CACHE"] = str(_CACHE)

        # Retry logic for CrossEncoder instantiation
        last_exc: Exception | None = None
        for attempt in range(10):
            try:
                self._model = CrossEncoder(
                    self.model_name,
                    device=_DEVICE,
                    activation_fn=torch.nn.Sigmoid(),
                    cache_folder=str(_CACHE),
                )
                last_exc = None
                break
            except Exception as e:
                if isinstance(e, (ConnectionError, OSError)) and attempt < 9:
                    last_exc = e
                    print(
                        f"[pretrained_models.py] Warning: failed to load CrossEncoder "
                        f"'{self.model_name}' (attempt {attempt+1}/10). Retrying in 2s..."
                    )
                    time.sleep(2)
                    continue
                raise
        if last_exc is not None:
            raise last_exc

    @torch.no_grad()
    def _predict_impl(self, pairs: List[Pair]) -> np.ndarray:
        # Note: CrossEncoder handles its own internal batching up to batch_size
        probs = self._model.predict(pairs, batch_size=self.batch_size)
        return probs.astype("float32")


# -----------------------------------------------------------------------------
# 4) Mixture-of-Experts (gate + weighted blend) --------------------------------
# -----------------------------------------------------------------------------
class _GateNet(nn.Module):
    """Simple softmax gate over K experts -> weights sum to 1."""

    def __init__(self, n_exp: int):
        super().__init__()
        # We learn an K×K linear layer (bias=True). At inference, we apply
        #   w = softmax( lin(pᵢ) )   for each row of pᵢ, where pᵢ ∈ ℝᴷ are expert probabilities
        self.lin = nn.Linear(n_exp, n_exp, bias=True)

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """
        p: (B, K) float32 tensor of expert probabilities
        returns: (B, K) float32 tensor of softmaxed weights
        """
        w = self.lin(p)               # (B, K)
        return torch.softmax(w, dim=1) # (B, K)


class MoEClassifier:
    """
    Given K “frozen” experts, train a gating network to blend their outputs.

    Parameters
    ----------
    experts : List[BaseExpert]
    lr      : learning rate for gate’s parameters
    epochs  : number of epochs to train the gate

    Methods
    -------
      .fit(pairs, y)         – train the gate on (q₁, q₂) pairs & binary labels y
      .predict_prob(pairs)   – return blended probabilities for new pairs
      .evaluate(pairs, y)    – return binary log-loss on given dataset
    """

    def __init__(self, experts: List[BaseExpert], lr: float = 1e-2, epochs: int = 100):
        self.experts = experts
        self.epochs = epochs
        self.gate = _GateNet(len(experts)).to(_DEVICE)
        self.opt = torch.optim.Adam(self.gate.parameters(), lr=lr)
        self.loss_fn = nn.BCELoss()  # binary cross-entropy

        # LR scheduler: reduce LR on plateau (monitor train loss)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, mode="min", factor=0.5, patience=5
        )

    def fit(self, pairs: List[Pair], y: np.ndarray):
        """
        Train only the gating network on top of frozen experts.
        pairs: list[(q₁, q₂)]
        y    : np.ndarray binary labels (0/1)

        Internally:
         - We randomly shuffle the pairs each epoch.
         - We process in “gate batch size” B=1024.
         - Early stopping if training log-loss hasn’t improved for 10 epochs.
         - We use a ReduceLROnPlateau scheduler to decrease LR when plateaus occur.
         - We print the final loss of each epoch.
        """
        y_t = torch.tensor(y, dtype=torch.float32).to(_DEVICE)
        B = 1024  # gate batch size

        best_loss = float("inf")
        no_improve_count = 0
        patience = 10  # early stopping patience

        for epoch in range(1, self.epochs + 1):
            perm = np.random.permutation(len(pairs))
            epoch_loss = 0.0

            for batch_idx in range(0, len(pairs), B):
                idx = perm[batch_idx : batch_idx + B]
                batch_pairs = [pairs[i] for i in idx]
                targets = y_t[idx]

                # 1) Freeze experts, collect their predictions (no grad):
                with torch.no_grad():
                    probs = np.column_stack(
                        [exp.predict_prob(batch_pairs) for exp in self.experts]
                    )  # shape: (batch_size, K)
                    probs_t = torch.tensor(probs, dtype=torch.float32).to(_DEVICE)

                # 2) Gate forward:
                weights = self.gate(probs_t)            # (batch_size, K)
                blended = (weights * probs_t).sum(dim=1) # (batch_size,)
                loss = self.loss_fn(blended, targets)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                epoch_loss = loss.item()

            # Scheduler step on the epoch's loss
            self.scheduler.step(epoch_loss)

            print(f"Epoch {epoch}/{self.epochs}  ·  loss {epoch_loss:.4f}  ·  lr {self.opt.param_groups[0]['lr']:.6f}")

            # Early stopping check
            if epoch_loss < best_loss - 1e-4:
                best_loss = epoch_loss
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= patience:
                print(f"[MoEClassifier] Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
                break

    @torch.no_grad()
    def predict_prob(self, pairs: List[Pair]) -> np.ndarray:
        """
        For new pairs, return blended duplicate probabilities (shape=(n_pairs,)).
        Implementation:
         1. For each expert, call `.predict_prob(pairs)` -> yields an (n_pairs,) array.
         2. Stack these K columns -> (n_pairs, K).
         3. Convert to torch.Tensor, feed into gate -> get (n_pairs, K) weight matrix.
         4. blended = (weights * probs).sum(dim=1) -> (n_pairs,)
        """
        probs = np.column_stack([exp.predict_prob(pairs) for exp in self.experts])
        probs_t = torch.tensor(probs, dtype=torch.float32).to(_DEVICE)
        weights = self.gate(probs_t)            # (n_pairs, K)
        blended = (weights * probs_t).sum(dim=1) # (n_pairs,)
        return blended.cpu().numpy().astype("float32")

    def evaluate(self, pairs: List[Pair], y: np.ndarray) -> float:
        """
        Compute binary log-loss between true labels `y` and predicted probs.
        """
        from sklearn.metrics import log_loss
        p = self.predict_prob(pairs)
        return log_loss(y, p)


# -----------------------------------------------------------------------------
# 5) Convenience factory to instantiate all desired experts ---------------
# -----------------------------------------------------------------------------
def default_experts(
    emb_path: str       = "../data/processed/question_embeddings_768.npy",
    lr_path: str        = "../models/pretrained/quoradistil_lr.pkl",
    embed_lr_ready: bool = True
) -> List[BaseExpert]:
    """
    Returns a list of expert instances in this order:
      [BertExpert, RobertaExpert, (XLNetExpert if available), QuoraDistilExpert, CrossEncExpert]

    Arguments:
      emb_path        : string path -> precomputed question_embeddings_768.npy (768-dim from Quora DistilBERT)
      lr_path         : string path -> quoradistil_lr.pkl (LogisticRegression on 1536 features)
      embed_lr_ready  : if False, QuoraDistilExpert will not error if no lr_path exists;
                        you must call .fit() manually to train the LR.
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

    # 4) QuoraDistilExpert (+ LR) using Quora-trained DistilBERT
    quora = QuoraDistilExpert(
        emb_path=emb_path,
        lr_path=lr_path
    )
    if not quora.lr_path.exists() and not embed_lr_ready:
        # User must call quora.fit(...) manually later
        pass
    exps.append(quora)

    # 5) Cross-Encoder
    exps.append(CrossEncExpert())

    return exps


# -----------------------------------------------------------------------------
# --- append to the end of src/pretrained_models.py -------------------------------------
# -----------------------------------------------------------------------------
import hashlib
import json

def _hash_pairs(pairs: List[Pair]) -> str:
    """
    SHA1 over the first/last 50 pairs so cache filenames are deterministic.
    """
    h = hashlib.sha1()
    sample = pairs[:50] + pairs[-50:]
    h.update(json.dumps(sample).encode())
    return h.hexdigest()[:8]

def get_predictions(
    experts   : List[BaseExpert],
    pairs     : List[Pair],
    split_tag : str,
    cache_dir : Path | str = "../models/pred_cache"
) -> np.ndarray:
    """
    Forward-pass every expert exactly once on `pairs`, cache each column to
      <cache_dir>/<split_tag>_<expert_name>.npy

    Returns a (n_pairs, n_experts) float32 matrix.

    Next time you call it on the same split, the .npy is mmap-loaded -> zero cost.
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

        # Catch errors early
        if p.shape[0] != len(pairs):
            raise ValueError(
                f"[ERROR] {exp.__class__.__name__} returned {p.shape[0]} predictions "
                f"but expected {len(pairs)} for split='{split_tag}'"
            )

        cols.append(p)

    return np.column_stack(cols).astype("float32")


def _subset_key(experts: list[BaseExpert]) -> str:
    """
    Given a list of expert instances (in a fixed order), produce a canonical
    key string, e.g. "BertExpert+RobertaExpert+CrossEncExpert".
    """
    return "+".join(e.__class__.__name__ for e in experts)

def save_gate(moe: "MoEClassifier", path: Path | str):
    """
    Persist the *trained* gate parameters only.  (We do not save expert weights
    because experts are frozen and pretrained.)
    """
    torch.save(moe.gate.state_dict(), str(path))

def load_gate(experts: list[BaseExpert], path: Path | str) -> "MoEClassifier":
    """
    Instantiate a MoEClassifier with the given experts and load stored gate weights.
    lr/epochs are dummies because we won’t re-train:
    """
    mdl = MoEClassifier(experts, lr=1.0, epochs=0)
    mdl.gate.load_state_dict(torch.load(str(path), map_location=_DEVICE))
    mdl.gate.eval()
    return mdl