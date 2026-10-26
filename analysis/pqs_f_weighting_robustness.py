"""Robustness of the instruction-tuned PQS^-F sign pattern to the term weights.

PQS^-F = z(s) + z(r) - z(NLL) - z(H) - z(sigma^2)  (Eq. 4; no Phi term), each term
z-scored over the 5,080 instances of its model and averaged per family.

The five term weights are redrawn from a symmetric Dirichlet(1,1,1,1,1) (scaled by 5)
and rho(Phi, PQS^-F) is recomputed for every model. Three different statements are
tested and kept separate:

    A  all three instruction-tuned models are individually negative;
    B  the instruction-tuned group mean is negative;
    C  the instruction-tuned group mean is below the base-model group mean.

With 8 families, Spearman rho is discrete and can be exactly 0 (this happens for
Llama-3.1-8B-Instruct in a few percent of draws). An exact zero is not counted as
negative; |rho| < 1e-12 is snapped to 0 so floating-point noise cannot flip the sign.
A vectorised routine is checked against a plain SciPy loop before it is used.

Run:  python analysis/pqs_f_weighting_robustness.py           (5,000 and 200,000 draws)
      python analysis/pqs_f_weighting_robustness.py --quick   (5,000 draws only)
"""

import argparse

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from common import INSTRUCT, MODEL_NAMES, header, load_full_results, save_table, zscore

SEEDS = [42, 1, 2026]
ZERO_TOL = 1e-12
PQS_F_TERMS = [("semantic_similarity", +1), ("rouge_l", +1), ("ref_nll", -1),
               ("entropy", -1), ("paraphrase_var", -1)]


def build():
    comp, fisher = [], []
    for name in MODEL_NAMES:
        df = load_full_results(name)
        signed = np.column_stack([s * zscore(df[c].values) for c, s in PQS_F_TERMS])
        fams = sorted(df["family"].unique())
        fi = df["family"].map({f: i for i, f in enumerate(fams)}).values.astype(int)
        counts = np.bincount(fi, minlength=len(fams)).astype(float)
        comp.append(np.stack([np.bincount(fi, weights=signed[:, j], minlength=len(fams)) / counts
                              for j in range(len(PQS_F_TERMS))], axis=1))
        fisher.append(np.array([df.loc[fi == i, "fisher_trace"].mean() for i in range(len(fams))]))
    return np.array(comp), np.array(fisher)          # (7, 8, 5), (7, 8)


def main(quick: bool):
    comp, fisher = build()

    header("PQS^-F weighting robustness")
    print("Equal-weight rho(Phi, PQS^-F):")
    for m, name in enumerate(MODEL_NAMES):
        print(f"  {name:<23} {spearmanr(fisher[m], comp[m] @ np.ones(5))[0]:+.4f}")

    Rf = np.array([rankdata(f) for f in fisher])
    Rc = Rf - Rf.mean(1, keepdims=True)
    Rn = Rc / np.linalg.norm(Rc, axis=1, keepdims=True)

    def rho_fast(W):
        s = np.einsum("mfj,kj->kmf", comp, W)
        r = np.apply_along_axis(rankdata, 2, s)
        rc = r - r.mean(2, keepdims=True)
        rn = rc / np.linalg.norm(rc, axis=2, keepdims=True)
        rho = np.einsum("kmf,mf->km", rn, Rn)
        return np.where(np.abs(rho) < ZERO_TOL, 0.0, rho)

    # Guard: vectorised routine vs SciPy reference.
    rng = np.random.default_rng(20260821)
    W_chk = rng.dirichlet(np.ones(5), size=300) * 5
    fast = rho_fast(W_chk)
    ref = np.array([[spearmanr(fisher[m], comp[m] @ w)[0] for m in range(7)] for w in W_chk])
    ref = np.where(np.abs(ref) < ZERO_TOL, 0.0, ref)
    assert np.abs(fast - ref).max() < 1e-9 and ((fast < 0) == (ref < 0)).all()
    print("\nVectorised routine agrees with the SciPy reference on 300 draws.")

    i_inst = [MODEL_NAMES.index(m) for m in INSTRUCT]
    i_base = [i for i in range(7) if i not in i_inst]

    def run(n_draws, seed, batch=4000):
        rng = np.random.default_rng(seed)
        a = b = c = done = 0
        while done < n_draws:
            k = min(batch, n_draws - done)
            done += k
            rho = rho_fast(rng.dirichlet(np.ones(5), size=k) * 5)
            iv, bv = rho[:, i_inst], rho[:, i_base]
            a += int((iv < 0).all(1).sum())
            b += int((iv.mean(1) < 0).sum())
            c += int((iv.mean(1) < bv.mean(1)).sum())
        return 100 * a / n_draws, 100 * b / n_draws, 100 * c / n_draws

    draws = [5000] if quick else [5000, 200000]
    rows = []
    for n in draws:
        for s in SEEDS:
            a, b, c = run(n, s)
            rows.append({"draws": n, "seed": s, "A_all_three_instruct_negative_pct": a,
                         "B_instruct_mean_negative_pct": b, "C_instruct_below_base_pct": c})
    res = pd.DataFrame(rows)
    print()
    print(res.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\nAt 5,000 draws the Monte-Carlo SE on A is about 0.6 percentage points, so the "
          "decimal is seed-dependent; the 200,000-draw rows give the converged values.")

    path = save_table(res, "pqs_f_weighting_robustness.csv")
    print(f"\nSaved {path.relative_to(path.parents[2])}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="only the 5,000-draw runs")
    main(ap.parse_args().quick)
