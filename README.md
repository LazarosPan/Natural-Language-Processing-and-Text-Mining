# Natural-Language-Processing-and-Text-Mining
Quora Dataset — Determine if two questions ask the same thing

# Mixture-of-Experts Pipeline (384D & 768D Paths) — Full Walkthrough

This repository implements a scalable, modular, and reproducible Mixture-of-Experts (MoE) pipeline for Quora duplicate question detection. The system mixes feature-based models, transformer-based experts, and dimensionality-reduced versions using PCA/UMAP. Models are cached and evaluated with a unified gate mechanism.

---

## Project Structure

```
Natural-Language-Processing-and-Text-Mining/
├── data/
│   ├── quora.csv
│   ├── splits/
│   │   ├── train.csv
│   │   ├── valid.csv
│   │   └── test.csv
│   └── processed/
│       ├── question_meta.csv
│       ├── clean_questions.npy
│       └── question_embeddings.npy
│
├── models/
│   ├── custom/           # Classical ML models (e.g., LR, XGB, LGBM, SVM)
│   ├── pretrained/       # QuoraDistilExpert logistic regression, pretrained assets
│   ├── pred_cache/       # Cached prediction outputs from experts
│   ├── gates/            # MoE gate weights, indices of top experts
│   └── features/         # Saved TF-IDF, SVD, PCA models
│
├── notebooks/
│   ├── 00_split.ipynb            # Data splitting
│   ├── 01_eda.ipynb              # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb    # Text cleaning, embedding
│   ├── 03_feature_engineering.ipynb
│   ├── 04_models.ipynb           # Expert training + MoE gate tuning
│   └── 05_benchmarks.ipynb       # Final test evaluation
│
├── src/
│   ├── preprocessing.py      # Cleaning functions, text normalizers
│   ├── features.py           # Feature builders, vectorizers
│   ├── modeling.py           # Utility functions for model wrappers
│   ├── pretrained_models.py  # Pretrained expert classes (BERT, CrossEnc, etc.)
│   ├── custom_models.py      # Classical ML expert classes (LR, SVM, etc.)
│   └── logs.py               # Unified logging interface
│
└── metric_logs/
    ├── features.csv      # Logs from 03_feature_engineering
    ├── models.csv        # Logs from individual expert training
    ├── gates.csv         # Logs from gate tuning and evaluation
    └── benchmarks.csv    # Final test performance (accuracy, F1, etc.)

```


---

## Notebook-by-Notebook Overview

### 00_split.ipynb
- Fixes reproducible splits (train/valid/test)
- Drops rows with nulls
- Saves `train.csv`, `valid.csv`, `test.csv` to `data/splits/`

### 01_eda.ipynb
- Explores class balance, length distributions, top tokens
- Logs EDA statistics to `metric_logs/eda.csv`

### 02_preprocessing.ipynb
- Creates per-question artifacts:
  - Raw, cleaned, lowercased
  - Lengths (chars/words)
- Sentence-transformer embeddings (768D)
- Saves:
  - `question_meta.csv`
  - `clean_questions.npy`
  - `question_embeddings.npy`

### 03_feature_engineering.ipynb
- Computes 3,598-dimensional features:
  - TF-IDF word/char
  - SVD word/char
  - Fuzzy scores
  - Embedding distances
  - Graph-based stats
- Dimensionality reduction:
  - PCA (retain 95% variance)
  - UMAP (n_neighbors=15, min_dist=0.1)
  - Combo: PCA → UMAP (for smooth manifold)
- Saves 3 versions for both `dim=384` and `dim=768`:
  - IPCA only (X_*_ipca.npy)
  - UMAP only (X_*_umap.npy)
  - IPCA + UMAP (X_*_ipca_umap.npy)

### 04_models.ipynb
- Two branches:
  - **Pretrained experts** use full (768D or 384D) sentence embeddings
  - **Custom experts** use reduced 3D (PCA/UMAP) engineered features

- Experts trained:
  - `BertExpert`
  - `RobertaExpert`
  - `XLNetExpert` *(optional)*
  - `CrossEncExpert`
  - `QuoraDistilExpert` with LR on [|u−v|, u·v]
  - `LRFeatureExpert`, `XGB`, `LGBM`, `KNN`, `RF`, `SVM`

- MoE Gate:
  - Gated softmax over expert outputs
  - Trained on all expert combinations
  - Logs all results with validation log-loss
  - Top-10 subsets are retrained on Train+Valid

### 05_benchmarks.ipynb
- Loads top-10 MoE gates from `models/gates/`
- Loads corresponding `moe_*_idxs.npy` subsets
- Runs `.predict_prob()` on `test.csv`
- Outputs:
  - `test_LL`, `test_ACC`, `test_F1`, `test_PREC`, `test_REC`, `test_AUC`, `seconds`
- Logs to: `metric_logs/benchmarks.csv`

---

## Evaluation Pipeline

- All metrics are logged via `log_event(...)` to `.csv` files
- You can use `05_benchmarks.ipynb` or the plotting tool to visualize:
  - Top-10 performance comparisons
  - Time vs log-loss tradeoffs
  - Accuracy/F1 per MoE gate
  - Correlation heatmaps of metric interactions

---

## Requirements (minimal)

Save this to `requirements.txt`:
```
numpy
pandas
scikit-learn
matplotlib
seaborn
sentence-transformers
transformers
xgboost
lightgbm
torch
umap-learn
rapidfuzz
networkx
```

Optional:
```
sentencepiece # required for XLNetExpert
```


---

## How to Reproduce

1. Download the `quora.csv` dataset into `data/`
2. Run the notebooks in order:
   - `00_split.ipynb`
   - `01_eda.ipynb`
   - `02_preprocessing.ipynb`
   - `03_feature_engineering.ipynb`
   - `04_models.ipynb`
   - `05_benchmarks.ipynb`
3. Inspect results in `metric_logs/` or plot from CSVs
4. Run ablations by altering feature reduction or expert list

---

## Notes for Academic Use

- All experiments are reproducible (seed fixed)
- Feature extraction, model training, and predictions are cached
- MoE architecture supports heterogeneous expert types (BERT + LR + CrossEnc)
- Dimensionality reduction and custom experts allow performance vs cost analysis
- Logs are flat and parseable for analysis or paper inclusion

---