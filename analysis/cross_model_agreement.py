"""Cross-model agreement: scalar PQS rankings (Kendall) vs prompt geometry (Mantel).

Kendall part (per-family PQS, 7 models x 8 families)
    * Kendall's W with tie correction, chi-square p, and a permutation p
      (9,999 permutations, seed 42);
    * the 21 pairwise Kendall tau values and their mean.

Mantel part (8x8 first-token JSD matrices, strict upper triangle, diagonal unused)
    * Mantel r for all 21 model pairs;
    * a Monte-Carlo p-value (9,999 row-and-column permutations, seed 42, two-sided);
    * the exact p-value from enumerating all 8! = 40,320 relabellings;
    * summaries: mean r over 21 pairs, the instruction-tuned trio, the three
      base/instruct pairs, and each model's mean between-family JSD.

The Mantel section is printed twice: once with the JSD matrices used in the manuscript
and once replacing Llama-3.1-8B-Instruct with the matrix from its A100 bf16 rerun
(data/jsd_matrices/supplementary/). See data/README.md for why both exist.

Run:  python analysis/cross_model_agreement.py
"""

from itertools import combinations, permutations

import numpy as np
import pandas as pd
from scipy.stats import chi2, kendalltau, pearsonr

from common import (DATA_DIR, INSTRUCT, MODEL_NAMES, SEED, WITHIN_FAMILY_PAIRS,
                    header, load_jsd_matrix, load_per_family, save_table)

N_PERM = 9999


# --------------------------------------------------------------------------- Kendall
def tie_term(col) -> float:
    _, counts = np.unique(col, return_counts=True)
    return float((counts ** 3 - counts).sum())


def kendall_w(ranks: pd.DataFrame) -> float:
    k, n = ranks.shape[1], ranks.shape[0]
    T = sum(tie_term(ranks.iloc[:, j].values) for j in range(k))
    rsum = ranks.sum(axis=1)
    S = ((rsum - rsum.mean()) ** 2).sum()
    return 12 * S / (k ** 2 * (n ** 3 - n) - k * T)


def kendall_section():
    header("Kendall agreement of per-family PQS rankings (7 models x 8 families)")
    scores = pd.DataFrame(
        {name: load_per_family(name).set_index("family")["PQS"] for name in MODEL_NAMES}
    ).sort_index()
    ranks = scores.rank(axis=0, method="average")
    k, n = ranks.shape[1], ranks.shape[0]
    W = kendall_w(ranks)
    chi_sq = k * (n - 1) * W
    p_chi = chi2.sf(chi_sq, df=n - 1)

    rng = np.random.default_rng(SEED)
    count = 1
    for _ in range(N_PERM):
        perm = ranks.apply(lambda c: rng.permutation(c.values), axis=0, result_type="broadcast")
        if kendall_w(perm) >= W:
            count += 1
    p_perm = count / (N_PERM + 1)

    pairs = []
    for a, b in combinations(MODEL_NAMES, 2):
        pairs.append({"model_1": a, "model_2": b,
                      "kendall_tau": kendalltau(scores[a], scores[b])[0]})
    pairs = pd.DataFrame(pairs)

    print(f"Kendall's W = {W:.3f}   chi2({n - 1}) = {chi_sq:.2f}, p = {p_chi:.3f}   "
          f"permutation p = {p_perm:.2f} ({N_PERM:,} permutations, seed {SEED})")
    print(f"Mean pairwise Kendall tau (21 pairs) = {pairs.kendall_tau.mean():.2f}")
    save_table(pairs, "kendall_pairwise.csv")
    return W, p_chi, p_perm


# --------------------------------------------------------------------------- Mantel
IU = np.triu_indices(8, k=1)
ALL_PERMS = np.array(list(permutations(range(8))))  # 40,320 x 8


def mantel_mc(A: np.ndarray, B: np.ndarray, n_perm: int = N_PERM, seed: int = SEED):
    """Row-and-column permutation Mantel test (two-sided), fresh RNG per pair."""
    rng = np.random.default_rng(seed)
    aa = A[IU]
    r_obs = pearsonr(aa, B[IU])[0]
    count = 1
    for _ in range(n_perm):
        perm = rng.permutation(A.shape[0])
        if abs(pearsonr(aa, B[perm][:, perm][IU])[0]) >= abs(r_obs):
            count += 1
    return r_obs, count / (n_perm + 1)


def mantel_exact(A: np.ndarray, B: np.ndarray):
    """Exact two-sided p over all 8! relabellings of B (identity included)."""
    aa = A[IU]
    r_obs = pearsonr(aa, B[IU])[0]
    rows, cols = IU
    bp = B[ALL_PERMS[:, rows], ALL_PERMS[:, cols]]          # 40,320 x 28
    a_c = aa - aa.mean()
    b_c = bp - bp.mean(axis=1, keepdims=True)
    r_all = (b_c @ a_c) / (np.linalg.norm(b_c, axis=1) * np.linalg.norm(a_c))
    p = np.mean(np.abs(r_all) >= abs(r_obs) - 1e-12)
    return r_obs, p


def mantel_section(matrices: dict, label: str, filename: str):
    header(f"Mantel agreement of first-token JSD geometry -- {label}")
    rows = []
    for a, b in combinations(MODEL_NAMES, 2):
        r, p_mc = mantel_mc(matrices[a], matrices[b])
        _, p_ex = mantel_exact(matrices[a], matrices[b])
        rows.append({"model_1": a, "model_2": b, "mantel_r": r,
                     "p_monte_carlo": p_mc, "p_exact": p_ex})
    res = pd.DataFrame(rows)

    def r_of(a, b):
        hit = res[((res.model_1 == a) & (res.model_2 == b)) | ((res.model_1 == b) & (res.model_2 == a))]
        return hit.iloc[0]

    print(f"{'pair':<50}{'r':>7}{'p (MC)':>9}{'p (exact)':>11}")
    for x in res.itertuples():
        print(f"{x.model_1 + ' vs ' + x.model_2:<50}{x.mantel_r:>+7.3f}"
              f"{x.p_monte_carlo:>9.4f}{x.p_exact:>11.4f}")

    trio = [r_of(a, b).mantel_r for a, b in combinations(INSTRUCT, 2)]
    print()
    print(f"Mean Mantel r over all 21 pairs       : {res.mantel_r.mean():.2f} "
          f"(range {res.mantel_r.min():+.2f} to {res.mantel_r.max():+.2f})")
    print(f"Mean Mantel r, instruction-tuned trio : {np.mean(trio):.2f}")
    print("Base vs instruct, same architecture and scale:")
    for a, b in WITHIN_FAMILY_PAIRS:
        x = r_of(a, b)
        print(f"  {a:<14} vs {b:<23} r = {x.mantel_r:.2f}, exact p = {x.p_exact:.3f}")

    print("\nMean between-family JSD (off-diagonal mean of each 8x8 matrix):")
    off = ~np.eye(8, dtype=bool)
    for name in MODEL_NAMES:
        print(f"  {name:<23} {matrices[name][off].mean():.2f}")

    save_table(res, filename)
    return res


def main():
    kendall_section()

    paper = {name: load_jsd_matrix(name).values for name in MODEL_NAMES}
    mantel_section(paper, "JSD matrices used in the manuscript", "mantel_pairwise_results.csv")

    rerun_path = (DATA_DIR / "jsd_matrices" / "supplementary"
                  / "llama_3_1_8b_instruct_jsd_matrix_a100_bf16_rerun.csv")
    rerun = pd.read_csv(rerun_path, index_col=0).values.astype(float)
    np.fill_diagonal(rerun, 0.0)
    alt = dict(paper)
    alt["Llama-3.1-8B-Instruct"] = rerun
    mantel_section(alt, "Llama-3.1-8B-Instruct replaced by its A100 bf16 rerun",
                   "mantel_pairwise_results_llama_instruct_a100_rerun.csv")


if __name__ == "__main__":
    main()
