"""De-circularisation test with the Fisher-free quality score PQS^-F.

PQS contains a -z(Phi) term, so correlating Phi with PQS is partly circular. PQS^-F
drops that term:

    PQS^-F = z(s) + z(r) - z(NLL) - z(H) - z(sigma^2)

with s = semantic similarity, r = ROUGE-L, NLL = reference negative log-likelihood,
H = output entropy, sigma^2 = paraphrase variance. Each descriptor is z-scored over the
5,080 instances of its model (population SD), then averaged per prompt family.

Reported per model (n = 8 families):
    rho(Phi, PQS^-F) and its p-value, the circular rho(Phi, PQS), the non-circular
    rho(Phi, s), and rho_align = rho(PQS^-F, PQS).

Run:  python analysis/fisher_free_pqs.py
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import INSTRUCT, MODELS, header, load_full_results, save_table, zscore

# (column, sign) for Eq. (4). No fisher_trace term.
PQS_F_TERMS = [
    ("semantic_similarity", +1),
    ("rouge_l", +1),
    ("ref_nll", -1),
    ("entropy", -1),
    ("paraphrase_var", -1),
]


def pqs_minus_f(df: pd.DataFrame) -> np.ndarray:
    return sum(sign * zscore(df[col].values) for col, sign in PQS_F_TERMS)


def main():
    header("De-circularisation test: rho(Phi, PQS^-F), n = 8 families per model")
    rows = []
    for name, _, is_inst in MODELS:
        df = load_full_results(name).copy()
        df["pqs_f"] = pqs_minus_f(df)
        fam = df.groupby("family").agg(
            fisher=("fisher_trace", "mean"),
            pqs_f=("pqs_f", "mean"),
            pqs=("PQS", "mean"),
            sim=("semantic_similarity", "mean"),
        )
        rho_f, p_f = spearmanr(fam.fisher, fam.pqs_f)
        rows.append({
            "model": name,
            "instruct": "yes" if is_inst else "no",
            "rho_fisher_vs_pqs_f": rho_f,
            "p": p_f,
            "rho_fisher_vs_pqs_circular": spearmanr(fam.fisher, fam.pqs)[0],
            "rho_fisher_vs_s": spearmanr(fam.fisher, fam.sim)[0],
            "rho_align_pqs_f_vs_pqs": spearmanr(fam.pqs_f, fam.pqs)[0],
        })
    res = pd.DataFrame(rows)

    print(f"{'model':<23}{'rho(Phi,PQS-F)':>15}{'p':>8}{'rho(Phi,PQS)':>14}"
          f"{'rho(Phi,s)':>12}{'rho_align':>11}")
    for r in res.itertuples():
        print(f"{r.model:<23}{r.rho_fisher_vs_pqs_f:>+15.2f}{r.p:>8.3f}"
              f"{r.rho_fisher_vs_pqs_circular:>+14.2f}{r.rho_fisher_vs_s:>+12.2f}"
              f"{r.rho_align_pqs_f_vs_pqs:>+11.2f}")

    inst = res[res.model.isin(INSTRUCT)]
    print(f"\nInstruction-tuned models with negative rho(Phi, PQS^-F): "
          f"{(inst.rho_fisher_vs_pqs_f < 0).sum()}/{len(inst)}")
    print("Note: with n = 8 families each estimate is noisy; read the sign pattern, "
          "not single p-values.")

    path = save_table(res, "fisher_free_pqs.csv")
    print(f"\nSaved {path.relative_to(path.parents[2])}")


if __name__ == "__main__":
    main()
