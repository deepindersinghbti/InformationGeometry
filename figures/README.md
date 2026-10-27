# Figures

`pdf/` holds the exact figure files of the submitted manuscript. The other folders hold the code that draws them; every figure is drawn from files in `data/` (or, for Fig. 5, from `analysis/results/`).

| Fig. | Content | Submitted file (`pdf/`) | Code | Input |
|---|---|---|---|---|
| 1 | Pipeline overview | `fig1_pipeline.pdf` | `source/fig1_pipeline.drawio` (open with [draw.io](https://app.diagrams.net)) | — |
| 2 | Prompt / output cosine-similarity matrices (16 × 16) | `fig2a_prompt_similarity.pdf`, `fig2b_output_similarity.pdf` | `notebooks/fig2_cosine_similarity_heatmaps.ipynb` | `data/surface_vs_behaviour/*_similarities.txt` |
| 3 | Surface vs. behavioural similarity scatter | `fig3_similarity_scatter.pdf` | `notebooks/fig3_similarity_scatter.ipynb` | `data/surface_vs_behaviour/recursion_real_outputs.csv` |
| 4 | Per-family PQS, seven models | `fig4_pqs_by_family.pdf` | `notebooks/fig4_pqs_by_family.ipynb` | `data/per_family/` |
| 5 | Cross-model Kendall τ and Mantel *r* | `fig5_agreement_heatmaps.pdf` | `notebooks/fig5_agreement_heatmaps.ipynb` | `data/per_family/`, `analysis/results/mantel_pairwise_results.csv` |
| 6 | First-token JSD heatmaps, seven models | `fig6_jsd_heatmaps.pdf` | `notebooks/fig6_jsd_heatmaps.ipynb` | `data/jsd_matrices/` |
| 7 | MDS prompt manifolds, seven models | `fig7_mds_manifolds.pdf` | `notebooks/fig7_mds_manifolds.ipynb` | `data/jsd_matrices/` |
| 8 | Fisher trace Φ vs semantic fidelity *s* (family means), Qwen2.5-7B base / instruct | `fig8a_…`, `fig8b_…` | `scripts/fisher_vs_pqs.py` | `data/full_results/` |
| 9 | Fisher trace Φ vs semantic fidelity *s* (family means), Llama-3.1-8B base / instruct | `fig9a_…`, `fig9b_…` | `scripts/fisher_vs_pqs.py` | `data/full_results/` |

## Running the notebooks

The figure notebooks were run in Google Colab and keep their stored outputs (integrity checks, printed statistics and previews). In Colab, *Runtime → Run all* and upload the input files listed above when prompted. Locally, Figs. 4, 6 and 7 read from a folder set in their config cell (`LOCAL_CSV_DIR` or `DATA_DIR`). Figs. 2 and 3 look for their inputs next to the notebook. Fig. 5 uses the Colab upload dialog only. Each notebook writes a vector PDF; Figs. 3, 6 and 7 also write a JSON manifest with input SHA-256 hashes and library versions.

The code cells are exactly as run. Only markdown text and notebook metadata were edited for publication.

Notes:

* **Fig. 3** re-embeds the prompts and outputs at full precision (needs `sentence-transformers` and a 90 MB model download). It checks the result against the values of the original GPU run, then computes the Mantel test (999,999 permutations, seed 42).
* **Fig. 6** shows the stored JSD matrices including their diagonal, which holds the mean within-family distance (see `data/README.md`). Fig. 7 sets the diagonal to zero before MDS.
* **Fig. 7**: MDS solutions are defined only up to rotation and reflection. Each panel is the lowest-stress solution over a seed sweep, rotated onto a reference orientation for readability. The notebook is saved with `LABEL_MODE = "index"` (numbered points); the submitted figure uses `LABEL_MODE = "full"`.
* **Figs. 8–9**: the submitted PDFs in `pdf/` were drawn with the plotting cell of the experiment notebooks and plot family-mean **PQS** on the y axis while the annotation gives Spearman ρ(Φ, *s*) ("ρ(Fisher,sim)"), so the annotated statistic does not describe the plotted points. `scripts/fisher_vs_pqs.py` now draws the y quantity and the annotation consistently. By default (`--y s`) it plots family-mean Φ vs family-mean *s* annotated with ρ(Φ, *s*) and its *p* (n = 8), the non-circular values of Table 4 (Qwen2.5-7B +0.60, Qwen2.5-7B-Instruct −0.26, Llama-3.1-8B +0.48, Llama-3.1-8B-Instruct +0.07). `--y pqs` plots PQS with ρ(Φ, PQS) (circular, because PQS contains −z(Φ)), and `--y pqsf` plots PQS⁻ᶠ with ρ(Φ, PQS⁻ᶠ) (Table 6). Output goes to `figures/regenerated/` (or `--outdir`), using the file names of the manuscript source. The dashed line is a least-squares fit for orientation only; the statistic is rank-based.
* `fig5`–`fig7` (Figs. 3–5 of the revised manuscript) use the A100 bf16 JSD matrix for Llama-3.1-8B-Instruct. The first submission used the matrix from an earlier 4-bit NF4 run (`experiments/main_experiment/llama_3_1_8b_instruct_rtx5070_nf4.ipynb`), now in `data/jsd_matrices/supplementary/`; see `data/README.md`.
* File names keep the numbering of the first submission: `fig2`/`fig3` are Figs. A1/A2 of the revised manuscript and `fig4`–`fig9` are Figs. 2–7.
