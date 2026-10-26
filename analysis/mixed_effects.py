"""Instance-level mixed-effects test of the Fisher-fidelity coupling.

For each model, fits  semantic_similarity ~ fisher_z  with a random intercept per task
(the 127 distinct questions), using all 5,080 instances. fisher_z is the Fisher trace
standardised within the model. p-values are Holm-corrected across the seven models.

Run:  python analysis/mixed_effects.py
Output: analysis/results/mixed_effects.csv
"""

import warnings

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from common import MODELS, header, load_full_results, save_table

warnings.filterwarnings("ignore")


def fit(name: str, is_instruct: bool) -> dict:
    df = load_full_results(name)
    # Task identifier = the question text (127 tasks), not the 6 broad task categories.
    df["task_id"] = df["question"].astype(str)
    f = df["fisher_trace"]
    df["fisher_z"] = (f - f.mean()) / f.std()
    m = smf.mixedlm("semantic_similarity ~ fisher_z", df, groups="task_id").fit()
    lo, hi = m.conf_int().loc["fisher_z"]
    return {
        "model": name,
        "instruct": "yes" if is_instruct else "no",
        "n_instances": len(df),
        "slope": m.params["fisher_z"],
        "ci_low": lo,
        "ci_high": hi,
        "p_raw": m.pvalues["fisher_z"],
    }


def main():
    header("Mixed-effects model: semantic_similarity ~ fisher_z + (1 | task)")
    res = pd.DataFrame([fit(name, inst) for name, _, inst in MODELS])
    res["p_holm"] = multipletests(res["p_raw"], method="holm")[1]

    show = res.copy()
    for c in ["slope", "ci_low", "ci_high", "p_raw", "p_holm"]:
        show[c] = show[c].round(4)
    show["sign"] = ["negative" if s < 0 else "positive" for s in res["slope"]]
    show = show[["model", "instruct", "n_instances", "slope", "sign",
                 "ci_low", "ci_high", "p_raw", "p_holm"]]
    print(show.to_string(index=False))

    path = save_table(show, "mixed_effects.csv")
    print(f"\nSaved {path.relative_to(path.parents[2])}")


if __name__ == "__main__":
    main()
