"""Representation sensitivity (Fisher trace, Phi) versus semantic fidelity (s).

Reproduces the family-level, non-circular Fisher-fidelity table:

* Spearman rho(Phi, s) over the 8 prompt-family means, with its p-value;
* a 2,000-sample bootstrap 95% CI (instances resampled with replacement, family
  means recomputed, seed 42) -- the same routine as the per-model experiment notebooks;
* the highest- and lowest-PQS family;
* the base vs instruction-tuned group means of rho(Phi, s), with an exact two-sided
  permutation test over all C(7,3) = 35 ways of labelling three models as instruct.

Run:  python analysis/fisher_fidelity.py
"""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import MODELS, SEED, header, load_full_results, save_table

N_BOOT = 2000


def family_means(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("family").agg(
        fisher=("fisher_trace", "mean"),
        sim=("semantic_similarity", "mean"),
        pqs=("PQS", "mean"),
    )


def bootstrap_ci(df: pd.DataFrame, n_boot: int = N_BOOT, seed: int = SEED):
    """Instance-level bootstrap of rho(Phi, s), identical to the experiment notebooks."""
    rng = np.random.default_rng(seed)
    inst = df[["family", "fisher_trace", "semantic_similarity"]].copy()
    boot = []
    for _ in range(n_boot):
        s = inst.sample(frac=1.0, replace=True, random_state=rng.integers(1e9))
        fm = s.groupby("family").agg(f=("fisher_trace", "mean"), q=("semantic_similarity", "mean"))
        if fm.f.nunique() > 1:
            boot.append(spearmanr(fm.f, fm.q).correlation)
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return lo, hi


def main():
    header("Fisher trace vs semantic fidelity (family level, n = 8 families per model)")
    rows = []
    for name, _, is_inst in MODELS:
        df = load_full_results(name)
        fm = family_means(df)
        rho, p = spearmanr(fm.fisher, fm.sim)
        rho_circ, p_circ = spearmanr(fm.fisher, fm.pqs)
        lo, hi = bootstrap_ci(df)
        rows.append({
            "model": name,
            "instruct": "yes" if is_inst else "no",
            "rho_fisher_vs_s": rho,
            "p": p,
            "ci95_low": lo,
            "ci95_high": hi,
            "rho_fisher_vs_pqs_circular": rho_circ,
            "top_pqs_family": fm.pqs.idxmax(),
            "bottom_pqs_family": fm.pqs.idxmin(),
        })
        print(f"{name:<23} rho = {rho:+.2f}  p = {p:.3f}  95% CI [{lo:+.2f}, {hi:+.2f}]"
              f"  top = {fm.pqs.idxmax():<15} bottom = {fm.pqs.idxmin()}")
    res = pd.DataFrame(rows)

    # Group comparison: base vs instruction-tuned.
    rho = dict(zip(res.model, res.rho_fisher_vs_s))
    names = list(rho)
    inst = [n for n, _, i in MODELS if i]
    base = [n for n in names if n not in inst]
    obs = np.mean([rho[n] for n in base]) - np.mean([rho[n] for n in inst])
    diffs = []
    for chosen in combinations(names, len(inst)):
        others = [n for n in names if n not in chosen]
        diffs.append(np.mean([rho[n] for n in others]) - np.mean([rho[n] for n in chosen]))
    diffs = np.array(diffs)
    p_perm = np.mean(np.abs(diffs) >= abs(obs) - 1e-12)

    print()
    print(f"Mean rho(Phi, s), base models           : {np.mean([rho[n] for n in base]):+.2f}")
    print(f"Mean rho(Phi, s), instruction-tuned     : {np.mean([rho[n] for n in inst]):+.2f}")
    print(f"Difference (base - instruct)            : {obs:+.2f}")
    print(f"Exact permutation p (two-sided, 35 labellings): {p_perm:.2f}")

    path = save_table(res, "fisher_fidelity.csv")
    print(f"\nSaved {path.relative_to(path.parents[2])}")


if __name__ == "__main__":
    main()
