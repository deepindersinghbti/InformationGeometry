"""Fisher trace vs a quality measure per prompt family, base vs instruction-tuned (Figs. 8 and 9).

Each panel plots the 8 family means of the Fisher trace Phi (x) against the family mean
of the chosen quality measure (y), with a least-squares line, and is annotated with the
Spearman correlation between exactly those two plotted quantities (n = 8 families):

    --y s     (default) semantic fidelity s; annotation rho(Phi, s). Non-circular; these
              values are the ones in Table 4 of the paper.
    --y pqs   PQS of Eq. (3); annotation rho(Phi, PQS). Circular: PQS contains -z(Phi).
    --y pqsf  Fisher-free PQS^-F of Eq. (4); annotation rho(Phi, PQS^-F) (Table 6).

Family means are computed from data/full_results/ (PQS^-F exactly as in
analysis/fisher_free_pqs.py).

Note on the submitted PDFs in figures/pdf/fig8*, fig9*: they plot PQS on the y axis but
are annotated with rho(Phi, s) ("rho(Fisher,sim)"), i.e. the annotated statistic does
not describe the plotted points. This script no longer produces that combination.

Run:  python figures/scripts/fisher_vs_pqs.py                 -> figures/regenerated/*.pdf
      python figures/scripts/fisher_vs_pqs.py --y pqsf --outdir some/dir
Output file names are those used by the manuscript source; for --y pqs / pqsf a suffix
(_y-pqs, _y-pqsf) is added.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))
from common import load_full_results  # noqa: E402
from fisher_free_pqs import pqs_minus_f  # noqa: E402

matplotlib.rcParams["pdf.fonttype"] = 42
DEFAULT_OUT = ROOT / "figures" / "regenerated"

PANELS = [
    ("Qwen2.5-7B", "Qwen2.5-7B (base)", "qwen2_5_7b_fig_fisher_pqs"),
    ("Qwen2.5-7B-Instruct", "Qwen2.5-7B-Instruct", "qwen2_5_7b_instruct_fig_fisher_pqs"),
    ("Llama-3.1-8B", "Llama-3.1-8B (base)", "llama_3_1_8b_fig2_fisher_pqs"),
    ("Llama-3.1-8B-Instruct", "Llama-3.1-8B-Instruct", "llama_3_1_8b_instruct_fig2_fisher_pqs"),
]

# y choice -> (column of the family table, y-axis label, symbol used in the annotation)
Y_OPTIONS = {
    "s": ("sim", "Family-mean semantic fidelity $s$", "s"),
    "pqs": ("pqs", "Family-mean PQS (Eq. 3, includes $-z(\\Phi)$)", "\\mathrm{PQS}"),
    "pqsf": ("pqs_f", "Family-mean PQS$^{-F}$ (Eq. 4, Fisher-free)", "\\mathrm{PQS}^{-F}"),
}

# Label offsets (points) where two family labels would otherwise overlap: (model, y, family).
LABEL_OFFSETS = {("Qwen2.5-7B", "s", "overconstrained"): (4, -9)}


def family_means(model: str):
    df = load_full_results(model).copy()
    df["pqs_f"] = pqs_minus_f(df)
    return df.groupby("family").agg(fisher=("fisher_trace", "mean"),
                                    pqs=("PQS", "mean"),
                                    pqs_f=("pqs_f", "mean"),
                                    sim=("semantic_similarity", "mean")).reset_index()


def draw(model: str, title: str, stem: str, y: str, outdir: Path) -> dict:
    col, ylabel, sym = Y_OPTIONS[y]
    fm = family_means(model)
    yv = fm[col]
    rho, p = spearmanr(fm.fisher, yv)

    fig, ax = plt.subplots(figsize=(4.55, 3.6))
    ax.scatter(fm.fisher, yv, s=40, c="#264653", edgecolor="white", linewidth=0.8, zorder=3)
    coef = np.polyfit(fm.fisher, yv, 1)
    xs = np.linspace(fm.fisher.min(), fm.fisher.max(), 50)
    # Least-squares line (for orientation only; the annotated statistic is rank-based).
    ax.plot(xs, np.polyval(coef, xs), "--", color="#e76f51", lw=1.1, zorder=2)
    for xv, yy, name in zip(fm.fisher, yv, fm.family):
        ax.annotate(name, (xv, yy), xytext=LABEL_OFFSETS.get((model, y, name), (4, 3)),
                    textcoords="offset points",
                    fontsize=6.5, fontweight="bold")
    ax.margins(0.12)
    ax.text(0.97, 0.04,
            f"Spearman $\\rho(\\Phi, {sym})$ = {rho:+.2f}, $p$ = {p:.3f} ($n$ = {len(fm)})",
            transform=ax.transAxes, ha="right", fontsize=6.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))
    ax.set_xlabel("Family-mean Fisher trace $\\Phi$ (local sensitivity)", fontsize=7.5)
    ax.set_ylabel(ylabel, fontsize=7.5)
    ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=8.5, fontweight="bold")
    fig.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / (stem + ("" if y == "s" else f"_y-{y}") + ".pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"{title:<24} rho(Phi, {y}) = {rho:+.3f}  p = {p:.4f}  -> {path}")
    return {"model": model, "y": y, "rho": rho, "p": p, "file": str(path)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--y", choices=sorted(Y_OPTIONS), default="s",
                    help="quantity on the y axis (default: s)")
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args()
    for args in PANELS:
        draw(*args, y=a.y, outdir=a.outdir)
