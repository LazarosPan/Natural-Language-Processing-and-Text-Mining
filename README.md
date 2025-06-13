# Natural-Language-Processing-and-Text-Mining
Quora Dataset — Determine if two questions ask the same thing

# Overview

This repository presents a modular and reproducible pipeline for detecting semantically equivalent questions in the Quora Question Pairs (QQP) dataset. It implements a Mixture-of-Experts (MoE) architecture that blends predictions from both classical machine learning models and pretrained transformer-based models through a softmax-trained gating mechanism. Extensive preprocessing, diverse feature engineering, and dimensionality reduction (IPCA & UMAP) contribute to robust and generalizable performance.

---

## Key Highlights

- Multi-resolution sentence embeddings (MiniLM, MPNet)

- Lexical, semantic, structural, and fuzzy features (~3600 features)

- Dimensionality reduction using IPCA (k95) and UMAP

- Multiple expert types: LR, SVM, XGB, LGBM, BERT, RoBERTa, CrossEnc, etc.

- Learnable soft-gate over expert logits (trained on validation log-loss)

- Cached embeddings, predictions, models, and gate weights

- Evaluation logged via lightweight .csv logger for reproducibility


---

## Project Structure

```
Natural-Language-Processing-and-Text-Mining/
├── data/
│   ├── quora.csv
│   ├── splits/                 # train/valid/test (no question leakage)
│   └── processed/              # clean text, embeddings, features
│       ├── question_meta.csv
│       ├── clean_questions.npy
│       ├── question_embeddings_384.npy / 768.npy
│       └── X_*_{ipca, umap}.npy
│
├── models/
│   ├── custom/                 # Pickled classical experts (LR, SVM, etc.)
│   ├── pretrained/             # Logistic head weights for DistilBERT
│   ├── pred_cache/             # Per-expert .npy prediction logs
│   ├── features*/              # TF-IDF, SVD, PCA caches
│   └── gates/                  # MoE gate weights and expert subset indices
│
├── notebooks/
│   ├── 0_split.ipynb
│   ├── 1_eda.ipynb
│   ├── 2_preprocessing.ipynb
│   ├── 3_feature_engineering.ipynb
│   ├── 4_models.ipynb
│   └── 5_benchmarks_lr{X}_ep{Y}.ipynb
│
├── src/
│   ├── preprocessing.py        # Cleaners, SBERT cache, len/word stats
│   ├── features.py             # Feature blocks, IPCA/UMAP
│   ├── custom_models.py        # LR/XGB/LGBM/etc. experts
│   ├── pretrained_models.py    # BERT, RoBERTa, Distil, CrossEnc, MoE
│   └── logs.py                 # CSV logger by event type
│
└── metric_logs/
    ├── splits.csv
    ├── eda.csv / eda_summary.csv
    ├── preprocessing.csv
    ├── features.csv
    ├── models.csv
    ├── gates.csv
    └── benchmarks_lrX_epY.csv

```


---

## Notebook-by-Notebook Overview

### 0_split.ipynb
- Fixes reproducible splits (train/valid/test)
- Drops rows with nulls
- Saves `train.csv`, `valid.csv`, `test.csv` to `data/splits/`

### 1_eda.ipynb
- Explores class balance, length distributions, top tokens
- Logs EDA statistics to `metric_logs/eda.csv`

### 2_preprocessing.ipynb
- Creates per-question artifacts:
  - Raw, cleaned, lowercased
  - Lengths (chars/words)
- Sentence-transformer embeddings (768D)
- Saves:
  - `question_meta.csv`
  - `clean_questions.npy`
  - `question_embeddings.npy`

### 3_feature_engineering.ipynb
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

### 4_models.ipynb
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

### 5_benchmarks*.ipynb
- Loads top-10 MoE gates from `models/gates/`
- Loads corresponding `moe_*_idxs.npy` subsets
- Runs `.predict_prob()` on `test.csv`
- Outputs:
  - `test_LL`, `test_ACC`, `test_F1`, `test_PREC`, `test_REC`, `test_AUC`, `seconds`
- Logs to: `metric_logs/benchmarks.csv`


### 6_summary.ipynb
- Result tables of the benchmarks.
- Correlation between metrics and hyperparameters.
---

## Evaluation Pipeline

- Evaluation metrics: Log-Loss, Accuracy, F1, Precision, Recall, ROC-AUC, Inference Time
- Gated combinations evaluated on test set under 3 configurations:
  - lr=0.001, epochs=1
  - lr=0.01, epochs=2
  - lr=0.05, epochs=10
- Results are stored in `metric_logs/benchmarks_lrX_epY.csv`
- Heatmap visualizations and correlation analyses assess metric interdependence
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
2. Run `main.py`or the notebooks in order:
   - `0_split.ipynb`
   - `1_eda.ipynb`
   - `2_preprocessing.ipynb`
   - `3_feature_engineering.ipynb`
   - `4_models.ipynb`
   - `5_benchmarks*.ipynb`
   - `6_summary.ipynb`
3. Inspect results in `metric_logs/` or plot from CSVs
4. Run ablations by altering feature reduction or expert list

---

## Academic Use

This repository was developed as part of a course assignment. It includes:

  - Modular architecture with clear separation of concerns

  - Feature-based + transformer-based modeling synergy

  - Log-based tracking for reproducibility

  - Validation-driven MoE tuning

  - Full support for ablation and metric correlation analysis

---