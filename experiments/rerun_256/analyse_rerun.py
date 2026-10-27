"""CPU analysis of the max_new_tokens re-run versus the original 64-token data.

For every model that has outputs/<tag>_mnt<N>.csv:
  * per family: % capped, mean s, mean ROUGE-L, exact-match accuracy (arithmetic, factual QA,
    both), for the ORIGINAL 64-token outputs (data/full_results, re-scored with scoring.py)
    and for the NEW outputs;
  * family rankings by s, by exact match, by PQS and by PQS^-F, 64 vs N;
  * PQS and PQS^-F recomputed with the new s, ROUGE-L and sigma^2_s (np.var over the 5
    variants of s within task x family, copied to the 5 rows -- notebook cell 10), keeping the
    stored Phi (fisher_trace), ref_nll and entropy; z-scores per model, population SD (ddof 0),
    exactly as notebook cell 11 / analysis/common.zscore;
  * rho(Phi, s) over the 8 family means (Spearman), and the mixed model
    s ~ z(Phi) + (1 | task) (statsmodels MixedLM, z with ddof 1, groups = question), as in
    analysis/mixed_effects.py. Also rho(Phi, exact match) and the mixed model on exact match.

% capped for the ORIGINAL data is estimated by re-tokenising the stored text with the model's
tokenizer and counting >= 60 tokens (the 26 Oct audit rule; the decoded+stripped text can
re-tokenise to a few tokens fewer than were generated -- see README for the check of this
rule against an exact 64-token re-run). For the NEW data hit_cap is exact.

Run:  python analyse_rerun.py [--mnt 256]
Writes results/*.csv and prints the tables.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import kendalltau, spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from scoring import score_row  # noqa: E402

warnings.filterwarnings("ignore")
DATA = HERE.parent.parent / "data"
RES = HERE / "results"

MODELS = [  # display, slug (data file), output tag, tokenizer, instruct
    ("DistilGPT-2", "distilgpt2", "distilgpt2", "distilgpt2", False),
    ("Qwen2.5-0.5B", "qwen2_5_0_5b", "qwen2_5_0_5b", "Qwen/Qwen2.5-0.5B", False),
    ("Qwen2.5-0.5B-Instruct", "qwen2_5_0_5b_instruct", "qwen2_5_0_5b_instruct",
     "Qwen/Qwen2.5-0.5B-Instruct", True),
    ("Qwen2.5-7B", "qwen2_5_7b", "qwen2_5_7b", "Qwen/Qwen2.5-7B", False),
    ("Qwen2.5-7B-Instruct", "qwen2_5_7b_instruct", "qwen2_5_7b_instruct", "Qwen/Qwen2.5-7B", True),
    ("Llama-3.1-8B", "llama_3_1_8b", "llama_3_1_8b", "meta-llama/Llama-3.1-8B", False),
    ("Llama-3.1-8B-Instruct", "llama_3_1_8b_instruct", "llama_3_1_8b_instruct",
     "meta-llama/Llama-3.1-8B", True),
]
FAMS = ["base", "expert", "cot", "structured", "concise", "overconstrained", "robust", "formatting"]
KEY = ["question", "family", "variant"]


def z(x):  # notebook cell 11 / common.zscore
    x = np.asarray(x, dtype=float)
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    sc = pd.DataFrame([score_row(a, b, c, d) for a, b, c, d in
                       zip(df.task_type, df.question, df.reference, df.output.fillna(""))],
                      index=df.index)
    for c in sc.columns:
        df[c] = sc[c]
    return df


def pqs_cols(df: pd.DataFrame) -> pd.DataFrame:
    df["paraphrase_var_re"] = df.groupby(["question", "family"]).semantic_similarity.transform(
        lambda s: np.var(s.values))
    base = z(df.semantic_similarity) + z(df.rouge_l) - z(df.ref_nll) - z(df.entropy) \
        - z(df.paraphrase_var_re)
    df["PQS_re"] = base - z(df.fisher_trace)
    df["PQS_F_re"] = base
    return df


def mixed(df: pd.DataFrame, y: str) -> dict:
    d = df.dropna(subset=[y]).copy()
    d[y] = d[y].astype(float)
    d["task_id"] = d.question.astype(str)
    f = d.fisher_trace
    d["fisher_z"] = (f - f.mean()) / f.std()
    m = smf.mixedlm(f"{y} ~ fisher_z", d, groups="task_id").fit()
    lo, hi = m.conf_int().loc["fisher_z"]
    return dict(slope=m.params["fisher_z"], ci_low=lo, ci_high=hi, p=m.pvalues["fisher_z"], n=len(d))


def order(s: pd.Series) -> str:
    return " > ".join(s.sort_values(ascending=False).index)


def family_table(df: pd.DataFrame, cap_col: str) -> pd.DataFrame:
    g = df.groupby("family")
    t = pd.DataFrame({
        "pct_capped": 100 * g[cap_col].mean(),
        "s": g.semantic_similarity.mean(),
        "rouge_l": g.rouge_l.mean(),
        "em_arith": 100 * df[df.task_type == "arithmetic"].groupby("family").arith_correct.mean(),
        "gold_present_arith": 100 * df[df.task_type == "arithmetic"].groupby("family")
        .arith_gold_present.mean(),
        "em_factual": 100 * df[df.task_type == "factual_qa"].groupby("family").factual_correct.mean(),
        "em_both": 100 * df[df.task_type.isin(["arithmetic", "factual_qa"])].groupby("family")
        .answer_correct.apply(lambda x: x.astype(float).mean()),
        "PQS": g.PQS_re.mean(),
        "PQS_F": g.PQS_F_re.mean(),
        "fisher": g.fisher_trace.mean(),
    })
    return t.loc[FAMS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mnt", type=int, default=256)
    ap.add_argument("--outdir", default=str(HERE / "outputs"))
    args = ap.parse_args()
    from transformers import AutoTokenizer

    RES.mkdir(exist_ok=True)
    fam_rows, rank_rows, stat_rows = [], [], []
    for name, slug, tag, tokname, inst in MODELS:
        new_p = Path(args.outdir) / f"{tag}_mnt{args.mnt}.csv"
        if not new_p.exists():
            continue
        orig = pd.read_csv(DATA / "full_results" / f"{slug}_full_results.csv")
        orig["output"] = orig.output.fillna("")
        refs = pd.read_csv(DATA / "prompts" / "evaluation_tasks.csv")[["question", "reference"]]
        orig = orig.merge(refs, on="question", how="left")
        # check our PQS recomputation reproduces the stored PQS on the original data
        o = pqs_cols(orig.copy())
        assert np.allclose(o.paraphrase_var_re, orig.paraphrase_var, atol=1e-9), "sigma2 def"
        assert np.allclose(o.PQS_re, orig.PQS, atol=1e-6), "PQS def"
        orig = add_scores(o)
        tok = AutoTokenizer.from_pretrained(tokname)
        orig["n_tok_retok"] = [len(tok(t, add_special_tokens=False).input_ids) for t in orig.output]
        orig["capped60"] = orig.n_tok_retok >= 60

        new = pd.read_csv(new_p)
        new["output"] = new.output.fillna("")
        if len(new) != 5080:
            print(f"WARNING {name}: {len(new)} rows in {new_p.name} (incomplete); skipping")
            continue
        keep = orig[KEY + ["fisher_trace", "ref_nll", "entropy"]]
        new = new.drop(columns=[c for c in ["fisher_trace", "ref_nll", "entropy"] if c in new])
        new = new.merge(keep, on=KEY, how="left", validate="one_to_one")
        new = pqs_cols(add_scores(new))
        new.to_csv(RES / f"{tag}_mnt{args.mnt}_scored.csv", index=False)

        t64 = family_table(orig, "capped60").add_suffix("_64")
        tN = family_table(new, "hit_cap").add_suffix(f"_{args.mnt}")
        ft = pd.concat([t64, tN], axis=1)
        ft.insert(0, "model", name)
        fam_rows.append(ft.reset_index().rename(columns={"index": "family"}))

        for lab, d, cap in [("64 (original)", orig, "capped60"), (f"{args.mnt} (rerun)", new, "hit_cap")]:
            t = family_table(d, cap)
            rank_rows.append(dict(model=name, budget=lab,
                                  by_s=order(t.s), by_em_both=order(t.em_both),
                                  by_em_arith=order(t.em_arith), by_PQS=order(t.PQS),
                                  by_PQS_F=order(t.PQS_F)))
            rho, p = spearmanr(t.fisher, t.s)
            rho_em, p_em = spearmanr(t.fisher, t.em_both)
            ms = mixed(d, "semantic_similarity")
            sub = d[d.task_type.isin(["arithmetic", "factual_qa"])].copy()
            sub["em"] = sub.answer_correct.astype(float)
            me = mixed(sub, "em")
            stat_rows.append(dict(
                model=name, budget=lab, overall_pct_capped=100 * d[cap].mean(),
                rho_phi_s=rho, p_phi_s=p, rho_phi_em=rho_em, p_phi_em=p_em,
                rho_phi_pqsF=spearmanr(t.fisher, t.PQS_F)[0], p_phi_pqsF=spearmanr(t.fisher, t.PQS_F)[1],
                rho_phi_pqs=spearmanr(t.fisher, t.PQS)[0],
                mixed_s_slope=ms["slope"], mixed_s_ci_low=ms["ci_low"], mixed_s_ci_high=ms["ci_high"],
                mixed_s_p=ms["p"], mixed_em_slope=me["slope"], mixed_em_p=me["p"], mixed_em_n=me["n"],
                top_PQS=t.PQS.idxmax(), bottom_PQS=t.PQS.idxmin(),
                top_s=t.s.idxmax(), bottom_s=t.s.idxmin(),
                top_em=t.em_both.idxmax(), bottom_em=t.em_both.idxmin()))
        # rank agreement 64 vs N
        t64r, tNr = family_table(orig, "capped60"), family_table(new, "hit_cap")
        for col in ["s", "em_both", "PQS", "PQS_F"]:
            tau, p = kendalltau(t64r[col], tNr[col])
            stat_rows[-1][f"kendall_64_vs_{args.mnt}_{col}"] = tau

    if not fam_rows:
        print("no complete re-run outputs found"); return
    F = pd.concat(fam_rows); R = pd.DataFrame(rank_rows); S = pd.DataFrame(stat_rows)
    F.to_csv(RES / f"per_family_64_vs_{args.mnt}.csv", index=False)
    R.to_csv(RES / f"rank_orders_64_vs_{args.mnt}.csv", index=False)
    S.to_csv(RES / f"stats_64_vs_{args.mnt}.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
    pd.set_option("display.max_colwidth", 120)
    cols = ["model", "family"] + [f"{c}_{b}" for c in ["pct_capped", "s", "rouge_l", "em_arith",
                                                       "em_factual", "PQS", "PQS_F"]
                                  for b in ["64", str(args.mnt)]]
    print(F[cols].round(3).to_string(index=False))
    print(R.to_string(index=False))
    print(S.round(4).T.to_string())


if __name__ == "__main__":
    main()
