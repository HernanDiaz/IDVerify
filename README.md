# DocVerify — Multi-Task CNN for Forged ID Document Detection

DocVerify is a multi-task deep learning system that simultaneously detects forged identity documents (binary classification) and localises the altered regions (segmentation mask). Its core contribution is a **multi-objective Pareto HPO framework** (Optuna / MOTPE) that jointly optimises PR-AUC and Dice instead of collapsing both objectives into a single scalar loss weight, making the classification–segmentation trade-off explicit and avoiding arbitrary design choices.

Evaluated on the [FantasyID dataset](https://www.idiap.ch/paper/fantasyid/), the system achieves results competitive with top-3 submissions to the [DeepID Challenge (ICCV 2025)](https://deepid-iccv.github.io/) while training exclusively on FantasyID and using a fully custom architecture — no TruFor pretraining, no external data.

| Metric | Ours | AG/EdgeDoc (3rd, challenge) | Sunlight (1st, challenge) |
|---|---|---|---|
| PR-AUC (nested CV) | 0.9921 ± 0.0058 | — | — |
| F1 detection (thr=0.5) | 0.9687 ± 0.0137 | 0.958 | 0.991 |
| F1 localisation (per-image) | 0.807 ± 0.096 | 0.686 | 0.784 |
| Dice (blind test, 30 seeds) | 0.875 ± 0.018 | — | — |

> AG/EdgeDoc is the most comparable team: same training data (FantasyID only), own architecture.  
> Sunlight used 60K+ external images and multi-stage pretraining.

---

## Architecture

```
Input image (224×224)
    └─► Encoder: Patel CNN (6 conv blocks, 8→256 filters)
            ├─► Classification head: GlobalAvgPool → 4×Dense+Dropout → logit
            └─► Decoder: U-Net with skip connections → Conv 1×1 → mask logit
```

- **Loss:** `w_cls · BCEWithLogitsLoss + w_mask · (BCE + Dice)` on logits
- **HPO:** MOTPE sampler, multi-objective `maximize(PR-AUC, Dice)`, Pareto-optimal trial selection by minimum Euclidean distance to ideal point (1, 1)
- **Validation:** Nested CV (10 outer × 5 inner folds), early stopping (patience=12), ReduceLROnPlateau

---

## Requirements

- Python 3.11
- NVIDIA GPU with CUDA support (tested on RTX 5060 Ti 16 GB, CUDA 13.0)
- FantasyID dataset

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-org>/docverify.git
cd docverify

# 2. Create and activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

# 3. Install dependencies
#    PyTorch CUDA 12.4 binaries are compatible with CUDA 13.0 (Blackwell)
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu124
```

### Dataset setup

Download [FantasyID](https://www.idiap.ch/paper/fantasyid/) and extract it so the directory structure looks like:

```
DocVerify/
└── FantasyID/
    ├── train/
    │   ├── bonafide/<device>/<stem>.jpg + <stem>.json
    │   └── attack/<attack_type>/<device>/<stem>.jpg + <stem>.json
    └── test/
        └── (same structure)
```

If the dataset is in a different location, set the environment variable before running:

```bash
# Windows
set DATASET_ROOT=C:\path\to\FantasyID
# Linux / macOS
export DATASET_ROOT=/path/to/FantasyID
```

---

## Running the Experiments

### 1 · Full pipeline — Nested CV + Blind Test

Runs the complete experiment from scratch: nested cross-validation with multi-objective HPO, blind test with 4 ablation variants (30 seeds each), and statistical tests.

```bash
python main.py
```

**Estimated time:** ~59 h on RTX 5060 Ti 16 GB.  
**Outputs:** `exports_hpo_pareto_nested/`

Key configuration parameters (edit `config.py` or pass as environment variables):

| Parameter | Default | Description |
|---|---|---|
| `N_OUTER` | 10 | Number of outer CV folds |
| `N_INNER` | 5 | Number of inner CV folds (HPO) |
| `N_TRIALS` | 50 | Optuna trials per outer fold |
| `N_FINAL_SEEDS` | 30 | Seeds for the blind test |
| `MAX_EPOCHS_FINAL` | 100 | Max epochs for final models |
| `BATCH_SIZE` | 64 | Batch size (increase to 256 on A100/H100) |

```bash
# Example: quick smoke test with reduced settings
set N_OUTER=2 && set N_TRIALS=2 && set MAX_EPOCHS_TRIAL=1 && python main.py
```

---

### 2 · Scalar experiment (classical scalarisation baseline)

Compares multi-objective Pareto HPO against mono-objective selection over a fixed `loss_w_mask` grid `{0.5, 1.0, 1.5, 2.0, 2.5, 3.0}`. Requires the full pipeline to have been run first (reuses the same splits and seeds).

```bash
python scalar_experiment.py
```

**Estimated time:** ~4.6 h on RTX 5060 Ti 16 GB.  
**Outputs:** `exports_hpo_pareto_nested/scalar_experiment/`

---

### 3 · DeepID Challenge metrics (re-evaluation, no retraining)

Computes the two metrics used by the DeepID Challenge ranking — F1 at fixed threshold 0.5 (Track 1) and per-image pixel-wise F1 (Track 2) — by loading the already-trained `.pt` models. Does not retrain anything.

```bash
python evaluate_challenge_metrics.py
```

**Prerequisite:** `python main.py` must have completed (needs the `.pt` models in `exports_hpo_pareto_nested/models/`).  
**Estimated time:** ~15 min on RTX 5060 Ti 16 GB (inference only, 30 seeds).  
**Outputs:** `exports_hpo_pareto_nested/challenge_metrics.csv` and `challenge_metrics_summary.csv`

---

## Results and Outputs

All results are written to `exports_hpo_pareto_nested/`. A timestamped `.zip` of all CSVs (models excluded) is created automatically at the end of each run.

| File | Generated by | Contents |
|---|---|---|
| `nested_outer_results.csv` | `main.py` | Per-fold metrics from the nested CV outer loop |
| `final_blind_test_multiseed.csv` | `main.py` | Blind test metrics per variant and seed |
| `stat_tests.csv` | `main.py` | Wilcoxon + Holm-Bonferroni + Cohen's d |
| `optuna_trials_nested.csv` | `main.py` | All HPO trials |
| `pareto_front_trials.csv` | `main.py` | Non-dominated trials per outer fold |
| `scalar_experiment/scalar_grid_full.csv` | `scalar_experiment.py` | Full grid results (w_mask × fold) |
| `scalar_experiment/scalar_grid_selected.csv` | `scalar_experiment.py` | Selection per criterion |
| `scalar_experiment/scalar_stats.csv` | `scalar_experiment.py` | Statistical tests for scalar experiment |
| `challenge_metrics.csv` | `evaluate_challenge_metrics.py` | F1@0.5 and per-image F1 per seed |
| `challenge_metrics_summary.csv` | `evaluate_challenge_metrics.py` | Summary statistics for challenge metrics |

The Optuna study is stored in `optuna_nested_outer.sqlite3` and can be explored interactively with [DB Browser for SQLite](https://sqlitebrowser.org/) or the Optuna Dashboard:

```bash
pip install optuna-dashboard
optuna-dashboard sqlite:///exports_hpo_pareto_nested/optuna_nested_outer.sqlite3
```

---

## Project Structure

```
DocVerify/
├── config.py                       — Global configuration (paths, hyperparameters, flags)
├── dataset.py                      — Dataset indexing, JSON parsing, VRAMCache, DataLoader
├── model.py                        — Architecture, losses, model factory
├── evaluate.py                     — Internal metrics (PR-AUC, Dice, mIoU, BACC, …)
├── evaluate_challenge_metrics.py   — DeepID Challenge metrics (F1@0.5, per-image F1)
├── train.py                        — Nested CV, HPO, early stopping, blind test, statistics
├── main.py                         — Main entry point
├── scalar_experiment.py            — Classical scalarisation experiment
└── requirements.txt
```

---

## License

This project is for research purposes. The FantasyID dataset is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
