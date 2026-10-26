"""Surface (prompt) similarity vs behavioural (output) similarity, 16 prompts.

Input: data/surface_vs_behaviour/recursion_real_outputs.csv -- the 16 prompts and the
outputs Qwen2.5-7B-Instruct produced for them (greedy decoding, max 512 new tokens,
bf16, seed 42; see provenance_manifest.json next to it).

Both texts are embedded with sentence-transformers/all-MiniLM-L6-v2 (normalised), the
two 16x16 cosine matrices are built, and their 120 strict-upper-triangle pairs are
compared with Pearson and Spearman correlations and a Mantel permutation test
(999,999 row-and-column permutations, seed 42, two-sided). 16! relabellings are too
many to enumerate, so the Mantel p is Monte Carlo and its third decimal depends on the
seed; the permutation count and seed are therefore fixed here.

Run:  python analysis/surface_vs_behaviour.py                 (needs sentence-transformers;
                                                               downloads the 90 MB embedder)
      python analysis/surface_vs_behaviour.py --from-matrices (no download; uses the saved
                                                               2-decimal matrices, so values
                                                               are close but not exact)
      add --n-perm 9999 for a fast check
"""

import argparse

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from common import DATA_DIR, header

SV_DIR = DATA_DIR / "surface_vs_behaviour"
EMBEDDER = "all-MiniLM-L6-v2"


def matrices_from_embeddings():
    from sentence_transformers import SentenceTransformer

    df = pd.read_csv(SV_DIR / "recursion_real_outputs.csv")
    df = df.sort_values("prompt_id").reset_index(drop=True)
    assert len(df) == 16
    model = SentenceTransformer(EMBEDDER)
    pe = model.encode(df["prompt_text"].tolist(), normalize_embeddings=True,
                      convert_to_numpy=True).astype(np.float64)
    oe = model.encode(df["output_text"].tolist(), normalize_embeddings=True,
                      convert_to_numpy=True).astype(np.float64)
    return pe @ pe.T, oe @ oe.T


def matrices_from_text():
    P = np.loadtxt(SV_DIR / "prompt_similarities.txt")
    O = np.loadtxt(SV_DIR / "output_similarities.txt")
    return P, O


def main(from_matrices: bool, n_perm: int, seed: int):
    if from_matrices:
        header("Surface vs behavioural similarity (saved 2-decimal matrices)")
        P, O = matrices_from_text()
    else:
        header(f"Surface vs behavioural similarity (re-embedded with {EMBEDDER})")
        P, O = matrices_from_embeddings()

    n = P.shape[0]
    iu = np.triu_indices(n, k=1)
    x, y = P[iu], O[iu]
    r, p_r = pearsonr(x, y)
    rho, p_rho = spearmanr(x, y)
    print(f"pairs                  : {len(x)}")
    print(f"mean prompt-pair cosine: {x.mean():.3f}")
    print(f"mean output-pair cosine: {y.mean():.3f}")
    print(f"Pearson r              : {r:.3f}  (p = {p_r:.1e}, R^2 = {r * r:.2f})")
    print(f"Spearman rho           : {rho:.3f}  (p = {p_rho:.3f})")

    rng = np.random.default_rng(seed)
    obs = float(np.corrcoef(x, y)[0, 1])
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        if abs(np.corrcoef(x, O[np.ix_(perm, perm)][iu])[0, 1]) >= abs(obs) - 1e-12:
            count += 1
    p_mantel = (count + 1) / (n_perm + 1)
    se = np.sqrt(p_mantel * (1 - p_mantel) / n_perm)
    print(f"Mantel r = {obs:.3f}, two-sided p = {p_mantel:.4f} (MC SE {se:.4f}; "
          f"{n_perm:,} permutations, seed {seed})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-matrices", action="store_true")
    ap.add_argument("--n-perm", type=int, default=999_999)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    main(a.from_matrices, a.n_perm, a.seed)
