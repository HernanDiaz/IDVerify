"""
data_loader.py — Loads all experiment CSVs once and exposes them as a PaperData object.

Dependency Inversion: figure generators receive a PaperData instance rather than
loading files themselves, keeping I/O concerns separate from plotting concerns.
"""

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from config import EXPORT_DIR


@dataclass
class PaperData:
    """Container for all DataFrames needed by the paper figures."""

    pareto_front:       pd.DataFrame   # pareto_front_trials.csv       (51 rows)
    all_trials:         pd.DataFrame   # optuna_trials_nested.csv       (501 rows)
    nested_cv:          pd.DataFrame   # nested_outer_results.csv       (10 rows)
    blind_test:         pd.DataFrame   # final_blind_test_multiseed.csv (120 rows)
    stat_tests:         pd.DataFrame   # stat_tests.csv
    challenge_summary:  pd.DataFrame   # challenge_metrics_summary.csv
    challenge_detail:   pd.DataFrame   # challenge_metrics.csv
    scalar_selected:    pd.DataFrame   # scalar_experiment/scalar_grid_selected.csv
    scalar_stats:       pd.DataFrame   # scalar_experiment/scalar_stats.csv


def load_paper_data(export_dir: Path = EXPORT_DIR) -> PaperData:
    """
    Read all experiment CSVs from export_dir and return a PaperData instance.

    Args:
        export_dir: path to the exports_hpo_pareto_nested/ directory.

    Returns:
        PaperData with all DataFrames loaded.
    """
    scalar_dir = export_dir / "scalar_experiment"

    return PaperData(
        pareto_front      = pd.read_csv(export_dir / "pareto_front_trials.csv"),
        all_trials        = pd.read_csv(export_dir / "optuna_trials_nested.csv"),
        nested_cv         = pd.read_csv(export_dir / "nested_outer_results.csv"),
        blind_test        = pd.read_csv(export_dir / "final_blind_test_multiseed.csv"),
        stat_tests        = pd.read_csv(export_dir / "stat_tests.csv"),
        challenge_summary = pd.read_csv(export_dir / "challenge_metrics_summary.csv"),
        challenge_detail  = pd.read_csv(export_dir / "challenge_metrics.csv"),
        scalar_selected   = pd.read_csv(scalar_dir  / "scalar_grid_selected.csv"),
        scalar_stats      = pd.read_csv(scalar_dir  / "scalar_stats.csv"),
    )
