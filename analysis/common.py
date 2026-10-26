"""Shared helpers for the analysis scripts.

Every script in this folder reads only the files in ``data/`` and needs no GPU.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# (display name, file slug, instruction-tuned?) in the order used in the paper.
MODELS = [
    ("DistilGPT-2", "distilgpt2", False),
    ("Qwen2.5-0.5B", "qwen2_5_0_5b", False),
    ("Qwen2.5-0.5B-Instruct", "qwen2_5_0_5b_instruct", True),
    ("Qwen2.5-7B", "qwen2_5_7b", False),
    ("Qwen2.5-7B-Instruct", "qwen2_5_7b_instruct", True),
    ("Llama-3.1-8B", "llama_3_1_8b", False),
    ("Llama-3.1-8B-Instruct", "llama_3_1_8b_instruct", True),
]
MODEL_NAMES = [m[0] for m in MODELS]
INSTRUCT = [m[0] for m in MODELS if m[2]]
BASE = [m[0] for m in MODELS if not m[2]]

# Base / instruction-tuned pairs that share an architecture and scale.
WITHIN_FAMILY_PAIRS = [
    ("Qwen2.5-0.5B", "Qwen2.5-0.5B-Instruct"),
    ("Qwen2.5-7B", "Qwen2.5-7B-Instruct"),
    ("Llama-3.1-8B", "Llama-3.1-8B-Instruct"),
]

SEED = 42


def _slug(name: str) -> str:
    for display, slug, _ in MODELS:
        if display == name:
            return slug
    raise KeyError(name)


def load_full_results(name: str) -> pd.DataFrame:
    """Per-instance results: 127 tasks x 8 families x 5 paraphrase variants = 5,080 rows."""
    return pd.read_csv(DATA_DIR / "full_results" / f"{_slug(name)}_full_results.csv")


def load_per_family(name: str) -> pd.DataFrame:
    """Per-family means of the per-instance descriptors (8 rows)."""
    return pd.read_csv(DATA_DIR / "per_family" / f"{_slug(name)}_per_family.csv")


def load_jsd_matrix(name: str, zero_diagonal: bool = True) -> pd.DataFrame:
    """8x8 first-token Jensen-Shannon distance matrix between prompt families.

    The stored diagonal holds the mean pairwise distance *within* a family, which is a
    different quantity from the between-family distances. None of the statistics use
    the diagonal; by default it is set to zero so the matrix is a proper distance matrix.
    """
    df = pd.read_csv(DATA_DIR / "jsd_matrices" / f"{_slug(name)}_jsd_matrix.csv", index_col=0)
    if zero_diagonal:
        values = df.values.astype(float).copy()
        np.fill_diagonal(values, 0.0)
        df = pd.DataFrame(values, index=df.index, columns=df.columns)
    return df


def zscore(x) -> np.ndarray:
    """Population z-score (ddof = 0), as used throughout the pipeline."""
    x = np.asarray(x, dtype=float)
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)


def save_table(df: pd.DataFrame, filename: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    df.to_csv(path, index=False)
    return path


def header(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)
