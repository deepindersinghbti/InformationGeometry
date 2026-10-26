"""Spread of the per-family PQS within each model (Sections 5.2 and 5.7).

For each model: SD across the 8 family-mean PQS values (sample SD, ddof = 1), range
(max - min), interquartile range, and the best and worst family. For the three
base/instruct pairs: the SD ratio and the Kendall tau between the two family rankings.

Run:  python analysis/pqs_dispersion.py
"""

import pandas as pd
from scipy.stats import kendalltau

from common import MODEL_NAMES, WITHIN_FAMILY_PAIRS, header, load_per_family, save_table


def main():
    header("Per-family PQS spread within each model")
    pqs = {m: load_per_family(m).set_index("family")["PQS"] for m in MODEL_NAMES}
    rows = []
    for m in MODEL_NAMES:
        p = pqs[m]
        rows.append({"model": m, "sd": p.std(ddof=1), "range": p.max() - p.min(),
                     "iqr": p.quantile(0.75) - p.quantile(0.25),
                     "best_family": p.idxmax(), "worst_family": p.idxmin()})
    res = pd.DataFrame(rows)
    print(res.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    print("\nBase vs instruction-tuned pairs:")
    for b, i in WITHIN_FAMILY_PAIRS:
        tau = kendalltau(pqs[b].sort_index(), pqs[i].sort_index())[0]
        sd_b, sd_i = pqs[b].std(ddof=1), pqs[i].std(ddof=1)
        print(f"  {b:<13} SD {sd_b:.2f} -> {sd_i:.2f} (x{sd_i / sd_b:.1f}),  "
              f"range {pqs[b].max() - pqs[b].min():.2f} -> {pqs[i].max() - pqs[i].min():.2f},  "
              f"Kendall tau of family rankings = {tau:+.2f}")

    path = save_table(res, "pqs_dispersion.csv")
    print(f"\nSaved {path.relative_to(path.parents[2])}")


if __name__ == "__main__":
    main()
