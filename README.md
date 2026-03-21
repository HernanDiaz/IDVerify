# DocVerify — Multi-Task CNN for Forged ID Document Detection

DocVerify is a multi-task deep learning system that simultaneously detects forged identity documents (binary classification) and localises the altered regions (pixel-level segmentation). Its core contribution is a **multi-objective Pareto HPO framework** (Optuna / MOTPE) that jointly optimises PR-AUC and Dice instead of collapsing both objectives into a single scalar loss weight.

Evaluated on the [FantasyID dataset](https://zenodo.org/records/17063366), the system achieves results competitive with top-3 submissions to the [DeepID 2025 Challenge (ICCV)](https://deepid-iccv.github.io/) while training exclusively on FantasyID with a fully custom architecture — no external data, no TruFor pretraining.

| Metric | DocVerify | AG/EdgeDoc (3rd, FantasyID-only) | Sunlight (1st, 60K+ external) |
|---|---|---|---|
| PR-AUC (nested CV) | **0.9921 ± 0.0058** | — | — |
| F1 detection (thr=0.5) | **0.969 ± 0.014** | 0.958 | 0.991 |
| F1 localisation (per-image) | **0.807 ± 0.096** | 0.686 | 0.784 |
| Dice (blind test, 30 seeds) | **0.875 ± 0.018** | — | — |

> DocVerify was developed independently after the DeepID 2025 Challenge deadline; results are on an internal 15% holdout, not the official test set.

---

## Architecture

```
Input image (224×224)
    └─► Encoder: Patel CNN (6 conv blocks, 8→256 filters, LeakyReLU + BN)
            ├─► Classification head: GlobalAvgPool → 4×Dense+Dropout → logit
            └─► Decoder: U-Net with skip connections → Conv 1×1 → mask logit
```

- **Loss:** `BCEWithLogitsLoss(cls) + λ_mask · (BCE + Dice)(seg)`
- **HPO:** MOTPE sampler, multi-objective `maximize(PR-AUC, Dice)`, Pareto-optimal selection by minimum Euclidean distance to ideal point (1,1)
- **Validation:** Nested CV (10 outer × 5 inner folds, 50 MOTPE trials/fold = 500 unique configs, 2,500 individual model trainings)

---

## Requirements

- Python 3.11
- NVIDIA GPU with CUDA support (tested: RTX 5060 Ti 16 GB, AMD Ryzen 5 3400G, 32 GB RAM, CUDA 13.0, Windows 11)
- FantasyID dataset

---

## Installation

```bash
git clone https://github.com/HernanDiaz/IDVerify.git
cd IDVerify
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu124
```

### Dataset setup

Download [FantasyID](https://zenodo.org/records/17063366) and extract to:

```
DocVerify/
└── FantasyID/
    ├── train/
    │   ├── bonafide/<device>/<stem>.jpg + <stem>.json
    │   └── attack/<attack_type>/<device>/<stem>.jpg + <stem>.json
    └── test/  (same structure)
```

Custom location: `set DATASET_ROOT=C:\path\to\FantasyID`

---

## Running the Experiments

### 1 · Full pipeline — Nested CV + Blind Test (~59 h)
```bash
python main.py
```
Outputs: `exports_hpo_pareto_nested/`

### 2 · Scalar HPO baseline (~4.6 h)
```bash
python scalar_experiment.py
```

### 3 · DeepID Challenge metrics re-evaluation (~15 min, inference only)
```bash
python evaluate_challenge_metrics.py
```

---

## Project Structure

```
DocVerify/
├── config.py                       — Global configuration
├── dataset.py                      — Dataset indexing, JSON parsing, VRAMCache
├── model.py                        — Architecture + losses
├── evaluate.py                     — Metrics (PR-AUC, Dice, BACC, …)
├── evaluate_challenge_metrics.py   — DeepID Challenge metrics (F1@0.5, F1 per-image)
├── train.py                        — Nested CV, HPO, early stopping, blind test
├── main.py                         — Entry point
├── scalar_experiment.py            — Classical scalarisation baseline
├── dataset_sidtd.py                — SIDTD dataset parser (tested, excluded from paper)
├── eval_sidtd.py                   — SIDTD evaluation script
├── finetune_sidtd.py               — SIDTD fine-tuning script
├── requirements.txt
├── exports_hpo_pareto_nested/      — Experiment results (CSVs + Optuna DB)
│   ├── nested_outer_results.csv    — Per-fold NCV metrics (Table 2)
│   ├── optuna_trials_nested.csv    — All 2,500 HPO trials
│   ├── optuna_nested_outer.sqlite3 — Full Optuna study database
│   ├── final_blind_test_multiseed.csv — Ablation study, 30 seeds (Table 3)
│   ├── scalar_experiment/          — Pareto vs scalar comparison (Table 4)
│   ├── challenge_metrics*.csv      — Track 1/2 challenge metrics (Table 5)
│   └── stat_tests.csv              — Wilcoxon + Cohen's d statistical tests
├── paper/
│   ├── README_paper.md             — Paper build instructions
│   ├── tifs/                       — IEEE T-IFS version (12 pages, IEEEtran)
│   └── prltemplate/                — Pattern Recognition Letters version (7 pages, elsarticle) — SUBMITTED
├── paper_figures/                  — Figure generation scripts (matplotlib)
└── sota_comparison/                — TruFor comparison pipeline
    ├── 00_export_holdout.py        — Export holdout images
    ├── 01_run_trufor.py            — TruFor zero-shot inference
    ├── 02_eval_comparison.py       — Metrics + comparison table generation
    ├── run_trufor_finetune.py      — TruFor fine-tuning on FantasyID
    ├── run_trufor_finetuned_inference.py — Fine-tuned model inference
    └── *.csv                       — Result files
```

---

## Paper

| Version | Venue | Pages | Status |
|---|---|---|---|
| `paper/tifs/` | IEEE T-IFS (Q1, IF ~6.8) | 12 | Draft — compiles, no errors |
| `paper/prltemplate/` | Pattern Recognition Letters (Q1) | 7 | **Submitted — 2026-03-21** |

See `paper/README_paper.md` for build instructions.

---

## License

Research purposes only. The FantasyID dataset is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
