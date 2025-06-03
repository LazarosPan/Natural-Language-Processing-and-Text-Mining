# Natural-Language-Processing-and-Text-Mining
Quora Dataset - Determine if two questions ask the same thing or not

# Mixture-of-Experts Pipeline — Full Walkthrough

This project implements a Mixture-of-Experts (MoE) pipeline for Quora duplicate question detection. It combines classical ML and pretrained transformer models, and includes caching, ensemble gating, and benchmark evaluation. Below is a plain-text, notebook-by-notebook description of the workflow and why each step matters.

---

## 00_split.ipynb
- **Goal**: Fix data splits to avoid leakage.
- Load the full `quora.csv` (≈400MB), shuffle if needed.
- Drop rows with missing questions.
- Save to:
  - `data/splits/train.csv`
  - `data/splits/valid.csv`
  - `data/splits/test.csv`

**Why it matters**: All future steps assume static splits. Models will never train on valid/test data.

---

## 01_eda.ipynb
- **Goal**: Inspect label distribution, question lengths, and example pairs.
- Read split CSVs and produce histograms and word-overlap summaries.

**Why it matters**: Sanity check — no downstream outputs are produced.

---

## 02_preprocessing.ipynb
- **Goal**: Create reusable per-question artifacts.
- Aggregate all unique questions across splits.
- For each:
  - Save raw → `question`
  - Cleaned → `clean` (lowercased, stripped)
  - Lengths in characters/words → `len`, `words`
- Save to:
  - `data/processed/question_meta.csv`
  - `data/processed/clean_questions.npy`

- Then encode each unique question via pretrained SentenceTransformer:
  - Save to: `question_embeddings.npy` (n_questions × 768)

- Optionally: Precompute CrossEncoder scores per pair
  - Save to: `train/valid/test_cross_scores.npy`

**Why it matters**: Cleaned texts and embeddings are used by all later models; avoids recomputing expensive steps.

---

## 03_feature_engineering.ipynb
- **Goal**: Build 3 598-dimensional feature matrix + reduce it.
- Load split CSVs + cleaned questions + embeddings.
- Compute:
  - TF-IDF cosine similarities
  - TruncatedSVD on word & char TF-IDF (→ 500 dims)
  - Dense BERT pairwise vectors (→ 3 072 dims)
  - Fuzzy ratios, char/word diffs (→ 4 dims)
  - Jaccard token overlap (→ 1 dim)
  - Length buckets + frequency stats (→ 18 dims)
  - CrossEncoder score (→ 1 dim)

- Total = 3 598 dims per pair → apply IncrementalPCA (retain 95% variance)
- Save:
  - `X_train.npy`, `X_valid.npy`, `X_test.npy`
  - PCA model: `pca_95.pkl`
  - TF-IDF and SVD models: `tfidf_*.pkl`, `svd_*.pkl`

**Why it matters**: All feature-based models use these compact PCA-reduced features.

---

## 04_models.ipynb

### Purpose:
- Train all feature-based + transformer experts.
- Cache their predictions.
- Tune a Mixture-of-Experts gate over all expert combinations.

### Cell 1:
- Run `%run setup.py`, load `train.csv` and `valid.csv`
- Define `pairs_tr`, `y_tr`, `pairs_val`, `y_val`

### Cell 2:
- Create dirs:
  - `models/custom/` for classical models
  - `models/pretrained/` for QuoraDistil LR
  - `models/gates/` for gate checkpoints
  - `models/features/` for TF-IDF/SVD/PCA
  - `models/metric_logs.txt` with headers

### Cell 3–4: Train classical models
- Train:
  - `LRFeatureExpert`
  - `XGBFeatureExpert`
  - `LGBMFeatureExpert`
  - `KNNFeatureExpert`
  - `RFFeatureExpert`
  - `SVMFeatureExpert`

- Each:
  - Loads `X_train.npy` or builds features via `build_features(...)`
  - Fits `.clf.fit(X_train, y)`
  - Saves model to `models/custom/*.pkl`
  - Logs training time + log-loss to `metric_logs.txt`

### Cell 5: Load HF experts
- Instantiate:
  - `BertExpert`, `RobertaExpert`, `XLNetExpert`
  - `QuoraDistilExpert` → fits LogisticRegression on 768-dim embedding pairs
  - `CrossEncExpert`

- Save LR model under `models/pretrained/` if missing.

### Cell 6:
- Combine all experts:
```
experts = hf_experts + feature_experts
```


### Cell 7:
- If `models/pred_cache/train_*.npy` exists → load
- Else:
- For each expert:
  - Compute `exp.predict_prob(pairs_tr)` → save `train_*.npy`
  - Compute `exp.predict_prob(pairs_val)` → save `valid_*.npy`
- Stack into:
- `P_tr` : (n_train × K)
- `P_val`: (n_valid × K)

### Cell 8: Gate tuning
- For each nonempty subset of experts:
- Subset key = e.g. `Bert+XGB+SVM`
- If `gate_<key>.pt` exists → load and evaluate
- Else:
  - Train MoEClassifier (2 epochs) on `P_tr_sub`, `y_tr`
  - Validate on `P_val_sub`, `y_val`
  - Save gate checkpoint
- Log: `subset_key, status, train_time, val_time, log_loss` → `metric_logs.txt`

- Track best gate by validation loss.

### Cell 9: Final Gate on train+valid
- Retrain best gate on full (train+valid) split
- Save:
- `models/gates/final_moe_gate.pt`
- `models/gates/moe_selected_idxs.npy`

**Why it matters**:
- Avoids recomputation of expensive expert predictions
- Gate only blends cached outputs
- Every model + metric is cached and reproducible

---

## 05_benchmarks.ipynb

- Load final gate + expert subset
- Run `moe.predict_prob(pairs_test)`
- Evaluate on:
- `log_loss`
- `accuracy`
- `F1 score`

**Why it matters**: Final model evaluation on unseen test data.

---

## Summary of Outputs

### Created in `02_preprocessing.ipynb`:
- `data/processed/question_meta.csv`
- `clean_questions.npy`
- `question_embeddings.npy`

### Created in `03_feature_engineering.ipynb`:
- `X_train.npy`, `X_valid.npy`, `X_test.npy`
- `tfidf_w.pkl`, `tfidf_c.pkl`, `svd_w_150.pkl`, `svd_c_100.pkl`, `pca_95.pkl`

### Created in `04_models.ipynb`:
- `models/custom/*.pkl` → feature models
- `models/pretrained/*.pkl` → DistilBERT LR
- `models/pred_cache/*.npy` → expert predictions
- `models/gates/*.pt` → MoE gates
- `models/gates/final_moe_gate.pt`, `moe_selected_idxs.npy`
- `models/metric_logs.txt`

**Every file has a purpose**:
- `00_split` → fix splits
- `01_eda` → sanity check
- `02_preprocessing` → question-wise cleaning + embedding
- `03_feature_engineering` → high-dimensional → PCA
- `04_models` → train experts + tune MoE gate
- `05_benchmarks` → final test evaluation