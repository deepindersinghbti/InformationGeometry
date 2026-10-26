"""Fisher trace vs PQS per prompt family, base vs instruction-tuned (Figs. 8 and 9).

Each panel plots the 8 family means of the Fisher trace (x) against the family mean
PQS (y), with a least-squares line and the non-circular Spearman rho(Phi, s) in the
corner. Family means are computed from data/full_results/.

The submitted PDFs in figures/pdf/ were drawn from the same family means with the
plotting cell of the per-model experiment notebooks (re-styled for the journal).
This script redraws them from the shared data so the figure can be checked without
a GPU; the plotted values are identical, minor typographic details may differ.

Run:  python figures/scripts/fisher_vs_pqs.py      -> figures/regenerated/*.pdf
"""

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))
from common import load_full_results  # noqa: E402

matplotlib.rcParams["pdf.fonttype"] = 42
OUT = ROOT / "figures" / "regenerated"

PANELS = [
    ("Qwen2.5-7B", "Qwen2.5-7B (base)", "qwen2_5_7b_fig_fisher_pqs.pdf"),
    ("Qwen2.5-7B-Instruct", "Qwen2.5-7B-Instruct", "qwen2_5_7b_instruct_fig_fisher_pqs.pdf"),
    ("Llama-3.1-8B", "Llama-3.1-8B (base)", "llama_3_1_8b_fig2_fisher_pqs.pdf"),
    ("Llama-3.1-8B-Instruct", "Llama-3.1-8B-Instruct", "llama_3_1_8b_instruct_fig2_fisher_pqs.pdf"),
]


def draw(model: str, title: str, filename: str) -> Path:
    df = load_full_results(model)
    fm = df.groupby("family").agg(fisher=("fisher_trace", "mean"),
                                  pqs=("PQS", "mean"),
                                  sim=("semantic_similarity", "mean")).reset_index()
    rho = spearmanr(fm.fisher, fm.sim).correlation

    fig, ax = plt.subplots(figsize=(4.55, 3.6))
    ax.scatter(fm.fisher, fm.pqs, s=40, c="#264653", edgecolor="white", linewidth=0.8, zorder=3)
    coef = np.polyfit(fm.fisher, fm.pqs, 1)
    xs = np.linspace(fm.fisher.min(), fm.fisher.max(), 50)
    ax.plot(xs, np.polyval(coef, xs), "--", color="#e76f51", lw=1.1, zorder=2)
    for x, y, name in zip(fm.fisher, fm.pqs, fm.family):
        ax.annotate(name, (x, y), xytext=(4, 3), textcoords="offset points",
                    fontsize=6.5, fontweight="bold")
    ax.margins(0.12)
    ax.text(0.97, 0.04, f"Spearman $\\rho$(Fisher,sim)={rho:+.2f}", transform=ax.transAxes,
            ha="right", fontsize=6.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))
    ax.set_xlabel("Fisher trace (local sensitivity)", fontsize=7.5)
    ax.set_ylabel("PQS", fontsize=7.5)
    ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=8.5, fontweight="bold")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / filename
    fig.savefig(path)
    plt.close(fig)
    print(f"{title:<24} rho(Phi, s) = {rho:+.2f}  -> {path.relative_to(ROOT)}")
    return path


if __name__ == "__main__":
    for args in PANELS:
        draw(*args)
