"""Sensitivity of the conclusions to the equal weighting of the PQS terms.

PQS gives equal weight to six z-scored terms:

    PQS = z(s) + z(r) - z(NLL) - z(H) - z(Phi) - z(sigma^2)

Signs are fixed by meaning; only the magnitudes are varied. Two statistics are tracked:

    A  Kendall's W of the 8-family PQS ranking across the 7 models (cross-model agreement);
    B  the median over models of rho(Phi, PQS^-F), where PQS^-F drops the Phi term.

Weighting schemes: equal weights (baseline), leave-one-term-out, 5,000 symmetric
Dirichlet draws (seed 42), and first-principal-component loadings.

Statistic B is a single summary over all seven models. The claim about the
instruction-tuned models specifically is tested in pqs_f_weighting_robustness.py.

Run:  python analysis/pqs_weighting_ablation.py
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

from common import MODEL_NAMES, SEED, header, load_full_results, save_table

N_DRAWS = 5000

# (column, sign, label)
COMPONENTS_FULL = [
    ("semantic_similarity", +1, "sem_sim"),
    ("rouge_l", +1, "rouge_l"),
    ("ref_nll", -1, "ref_nll"),
    ("entropy", -1, "entropy"),
    ("fisher_trace", -1, "fisher"),
    ("paraphrase_var", -1, "par_var"),
]
COMPONENTS_NF = [c for c in COMPONENTS_FULL if c[0] != "fisher_trace"]


def z(x):
    x = np.asarray(x, dtype=np.float64)
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)


def compute_pqs(df, components, weights=None):
    if weights is None:
        weights = np.ones(len(components))
    pqs = np.zeros(len(df))
    for (col, sign, _), w in zip(components, weights):
        pqs += sign * w * z(df[col].values)
    return pqs


def kendall_w(rank_matrix):
    """Kendall's W from an (n_raters x n_items) rank matrix (no tie correction)."""
    n_raters, n_items = rank_matrix.shape
    rank_sums = rank_matrix.sum(axis=0)
    S = np.sum((rank_sums - rank_sums.mean()) ** 2)
    return float(12 * S / (n_raters ** 2 * (n_items ** 3 - n_items)))


def compute_stats(data, w_full=None, w_nf=None):
    models = list(data)
    families = sorted(data[models[0]]["family"].unique())
    ranks = np.zeros((len(models), len(families)))
    rhos = []
    for i, m in enumerate(models):
        g = data[m]
        fam = g.assign(_p=compute_pqs(g, COMPONENTS_FULL, w_full)).groupby("family")["_p"].mean()
        r = fam.rank(ascending=False)
        ranks[i] = [r.get(f, np.nan) for f in families]

        t = g.assign(_q=compute_pqs(g, COMPONENTS_NF, w_nf)).groupby("family").agg(
            fisher=("fisher_trace", "mean"), q=("_q", "mean"))
        if t.fisher.nunique() > 1:
            rhos.append(spearmanr(t.fisher, t.q)[0])
    med = float(np.median(rhos))
    return kendall_w(ranks), med, "+" if med >= 0 else "-"


def family_means_np(v, fam_idx, n_fam):
    return np.bincount(fam_idx, weights=v, minlength=n_fam) / (
        np.bincount(fam_idx, minlength=n_fam).astype(float) + 1e-12)


def main():
    frames = []
    for name in MODEL_NAMES:
        df = load_full_results(name)
        df["model"] = name
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    data = {m: g.reset_index(drop=True) for m, g in all_df.groupby("model")}

    header("PQS weighting ablation")
    err = max(np.max(np.abs(compute_pqs(g, COMPONENTS_FULL) - g["PQS"].values)) for g in data.values())
    print(f"Equal-weight PQS reconstructs the stored PQS column (max |error| = {err:.1e})")

    base_W, base_rho, base_sign = compute_stats(data)
    print(f"\nBaseline (equal weights): Kendall's W = {base_W:.4f}, "
          f"median rho(Phi, PQS^-F) = {base_rho:+.4f}")

    # Leave one term out.
    loo = []
    nf_cols = [c[0] for c in COMPONENTS_NF]
    for k, (col, _, label) in enumerate(COMPONENTS_FULL):
        w_full = np.ones(len(COMPONENTS_FULL))
        w_full[k] = 0.0
        w_nf = np.ones(len(COMPONENTS_NF))
        if col in nf_cols:
            w_nf[nf_cols.index(col)] = 0.0
        W, rho, sign = compute_stats(data, w_full, w_nf)
        loo.append({"dropped_term": label, "kendall_W": W, "delta_W": W - base_W,
                    "median_rho": rho, "sign_preserved": sign == base_sign})
    loo = pd.DataFrame(loo)
    print("\nLeave-one-term-out:")
    print(loo.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    # Random Dirichlet reweighting.
    arrays = {}
    for m, g in data.items():
        fams = sorted(g["family"].unique())
        idx = g["family"].map({f: i for i, f in enumerate(fams)}).values.astype(int)
        arrays[m] = dict(
            z_full=np.column_stack([s * z(g[c].values) for c, s, _ in COMPONENTS_FULL]),
            z_nf=np.column_stack([s * z(g[c].values) for c, s, _ in COMPONENTS_NF]),
            idx=idx, n=len(fams),
            fisher=np.array([g.loc[idx == i, "fisher_trace"].mean() for i in range(len(fams))]),
        )
    models = list(arrays)
    rng = np.random.default_rng(SEED)
    W_vals = np.empty(N_DRAWS)
    rho_vals = np.empty(N_DRAWS)
    for d in range(N_DRAWS):
        w_full = rng.dirichlet(np.ones(len(COMPONENTS_FULL))) * len(COMPONENTS_FULL)
        w_nf = rng.dirichlet(np.ones(len(COMPONENTS_NF))) * len(COMPONENTS_NF)
        R = np.zeros((len(models), arrays[models[0]]["n"]))
        rhos = []
        for i, m in enumerate(models):
            a = arrays[m]
            fm = family_means_np(a["z_full"] @ w_full, a["idx"], a["n"])
            order = np.argsort(fm)[::-1]
            r = np.empty_like(order)
            r[order] = np.arange(1, len(order) + 1)
            R[i] = r
            q = family_means_np(a["z_nf"] @ w_nf, a["idx"], a["n"])
            if np.unique(a["fisher"]).size > 1:
                rhos.append(spearmanr(a["fisher"], q)[0])
        W_vals[d] = kendall_w(R)
        rho_vals[d] = float(np.median(rhos))
    sign_pct = 100.0 * np.mean(np.sign(rho_vals) == np.sign(base_rho))
    print(f"\nRandom reweighting ({N_DRAWS:,} Dirichlet draws, seed {SEED}):")
    print(f"  Kendall's W        : mean {W_vals.mean():.3f} +/- {W_vals.std():.3f} (SD), "
          f"range [{W_vals.min():.3f}, {W_vals.max():.3f}]")
    print(f"  median rho sign    : same as baseline in {sign_pct:.1f}% of draws")

    # PCA-derived weights.
    Z = np.vstack([np.column_stack([s * z(g[c].values) for c, s, _ in COMPONENTS_FULL])
                   for g in data.values()])
    load = np.abs(PCA(n_components=1).fit(Z).components_[0])
    nf_idx = [i for i, c in enumerate(COMPONENTS_FULL) if c[0] != "fisher_trace"]
    W_pca, rho_pca, sign_pca = compute_stats(data, load, load[nf_idx])
    print(f"\nPCA (first-component loadings): Kendall's W = {W_pca:.3f}, "
          f"median rho = {rho_pca:+.3f}, sign preserved: {sign_pca == base_sign}")

    summary = pd.DataFrame(
        [{"scheme": "equal weights (baseline)", "kendall_W": base_W, "kendall_W_sd": np.nan,
          "median_rho": base_rho, "sign_preserved": "baseline"}]
        + [{"scheme": f"leave out {r.dropped_term}", "kendall_W": r.kendall_W, "kendall_W_sd": np.nan,
            "median_rho": r.median_rho, "sign_preserved": str(r.sign_preserved)} for r in loo.itertuples()]
        + [{"scheme": f"random Dirichlet (N={N_DRAWS}, seed {SEED})", "kendall_W": W_vals.mean(),
            "kendall_W_sd": W_vals.std(), "median_rho": rho_vals.mean(),
            "sign_preserved": f"{sign_pct:.1f}% of draws"},
           {"scheme": "PCA loadings", "kendall_W": W_pca, "kendall_W_sd": np.nan,
            "median_rho": rho_pca, "sign_preserved": str(sign_pca == base_sign)}]
    )
    path = save_table(summary, "pqs_weighting_ablation.csv")
    print(f"\nSaved {path.relative_to(path.parents[2])}")


if __name__ == "__main__":
    main()
