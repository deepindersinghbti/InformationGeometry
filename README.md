# The Information Geometry of Prompting — code and data

Code, data and figures for the manuscript **"The Information Geometry of Prompting: A Representation-Sensitivity Account of Prompt Quality Across Language Models"** (under review).

The study passes 127 tasks × 8 prompt families × 5 paraphrases (5,080 prompt instances) through seven open-weight checkpoints. For every instance it records representation-geometry descriptors (Fisher trace Φ, effective dimension, trajectory curvature), first-token output distributions, and behavioural descriptors (semantic fidelity, ROUGE-L, reference NLL, entropy, paraphrase variance). It then compares the models in two ways: on scalar quality rankings (PQS) and on the relational geometry of prompt families (first-token Jensen–Shannon distances).

## Contents

```
data/                      inputs for every analysis and figure (see data/README.md)
  prompts/                 127 evaluation tasks with reference answers; 40 prompt templates
  full_results/            one row per instance, 5,080 rows x 7 models
  per_family/              family means, 8 rows x 7 models
  jsd_matrices/            8x8 first-token JSD matrices, 7 models (+ supplementary/)
  surface_vs_behaviour/    Section 5.1: 16 prompts, model outputs, cosine matrices, manifest
experiments/               GPU notebooks that produced data/ (stored outputs = the actual runs)
  main_experiment/         5 notebooks covering the 7 checkpoints, plus the earlier
                           Llama-3.1-8B-Instruct run that produced its JSD matrix
  surface_vs_behaviour/    Section 5.1 generation run (Qwen2.5-7B-Instruct)
analysis/                  CPU scripts that reproduce every table and statistic from data/
  results/                 the tables they write (committed for comparison)
figures/                   submitted figure PDFs and the code that draws them (see figures/README.md)
requirements.txt           CPU analysis + figures
requirements-experiments.txt   GPU experiments
```

## Quick start (CPU only, no model downloads)

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python analysis/run_all.py            # about 10 minutes; add --quick for about 4 minutes
```

Each script can also be run on its own, e.g. `python analysis/mixed_effects.py`. The scripts read only `data/` and write CSV tables to `analysis/results/`.

## Where each result in the manuscript comes from

| Manuscript | Result | Reproduced by |
|---|---|---|
| Tables 1–2, §4 | models, prompt families, 127 tasks | `data/README.md`, `data/prompts/` |
| §5.1, Table 3, Figs. 2–3 | mean prompt / output cosine 0.694 / 0.671; Pearson *r* = 0.484; Spearman ρ = 0.144 (*p* = 0.116); Mantel *p* = 0.060 (999,999 permutations, seed 42) | `analysis/surface_vs_behaviour.py`, `figures/notebooks/fig3_similarity_scatter.ipynb` |
| §5.2, Fig. 4 | per-family PQS; SD across families 1.06→1.30, 0.55→1.46, 0.33→1.85; ranges 0.28 to 5.61 | `analysis/pqs_dispersion.py` |
| §5.3, Table 4 | ρ(Φ, *s*) per model; base mean +0.19 vs instruct −0.30, exact permutation *p* = 0.23 | `analysis/fisher_fidelity.py` |
| §5.3, Table 5 | mixed-effects slopes, 95% CIs, Holm-corrected *p* | `analysis/mixed_effects.py` |
| §5.4, Table 6 | ρ(Φ, PQS⁻ᶠ), *p*, ρ(Φ, PQS), ρ(Φ, *s*), ρ_align | `analysis/fisher_free_pqs.py` |
| §5.5, Figs. 5–7 | Kendall's *W* = 0.185 (χ² = 9.05, df = 7, *p* = 0.249), mean τ = 0.05; Mantel *r̄* = 0.38, instruct trio 0.82, max 0.91; mean inter-family JSD 0.10 to 0.57 | `analysis/cross_model_agreement.py` |
| §5.6 | *W* under reweighting 0.228 ± 0.112, PCA weights 0.428; instruct group mean negative 99.8%, below base 99.3%, all three negative 21.3% (5,000 draws, seed 42) | `analysis/pqs_weighting_ablation.py`, `analysis/pqs_f_weighting_robustness.py` |
| §5.7, Figs. 8–9 | within-pair τ +0.14 / 0.00 / −0.50; SD ratio ×1.2 / ×2.6 / ×5.6; Mantel *r* 0.30 / 0.21 / 0.52 (exact *p* 0.166 / 0.377 / 0.064) | `analysis/pqs_dispersion.py`, `analysis/cross_model_agreement.py`, `figures/scripts/fisher_vs_pqs.py` |

Every script above was re-run from this repository, on the files in `data/`, before release.

## Experiments (GPU)

| Notebook (`experiments/`) | Checkpoint(s) | Hardware | Precision | Run time |
|---|---|---|---|---|
| `main_experiment/distilgpt2.ipynb` | `distilgpt2` | RTX 5070, 12 GB | fp32 | 41 min |
| `main_experiment/qwen2_5_0_5b.ipynb` | `Qwen/Qwen2.5-0.5B` | RTX 5070, 12 GB | bf16 | 1.5 h |
| `main_experiment/qwen2_5_0_5b_instruct.ipynb` | `Qwen/Qwen2.5-0.5B-Instruct` | RTX 5070, 12 GB | bf16 | 1.5 h |
| `main_experiment/qwen2_5_7b.ipynb` | `Qwen/Qwen2.5-7B` (stored run) and `Qwen/Qwen2.5-7B-Instruct` | A100-PCIE-40GB | bf16 | 6.6 h |
| `main_experiment/llama_3_1_8b_instruct.ipynb` | `meta-llama/Llama-3.1-8B-Instruct` (stored run) and `meta-llama/Llama-3.1-8B` | A100-PCIE-40GB | bf16 | 10.6 h |
| `main_experiment/llama_3_1_8b_instruct_rtx5070_nf4.ipynb` | `meta-llama/Llama-3.1-8B-Instruct`, earlier run; source of its JSD matrix only | RTX 5070, 12 GB | 4-bit NF4 | 5.9 h |
| `surface_vs_behaviour/surface_vs_behaviour_qwen2_5_7b_instruct.ipynb` | `Qwen/Qwen2.5-7B-Instruct` (16 prompts) | A100-PCIE-40GB | bf16 | minutes |

```bash
pip install -r requirements-experiments.txt     # after installing a torch build for your GPU
```

Open a notebook and run all cells. To run the second checkpoint listed for a notebook, make its `MODEL_NAME` line the active one. Each run writes `<model>_full_results.csv`, `<model>_per_family.csv` and `<model>_jsd_matrix.csv` to `results_<model>/`; the copies used in the paper are in `data/`. The dataset, prompt-family, generation and metric cells are identical in all six main-experiment notebooks. The Llama models are gated: accept the licence on Hugging Face and export `HF_TOKEN`. No token is stored anywhere in this repository.

All runs use seed 42. Greedy generations are deterministic on a given GPU, dtype and library version. A re-run on other hardware or library versions can change individual generated texts and the values derived from them. The CSVs in `data/` are the exact outputs analysed in the paper.

## Notes for reviewers

* **Llama-3.1-8B-Instruct JSD matrix.** The JSD matrix used in the manuscript for this model comes from an earlier run of the same pipeline on the RTX 5070 in 4-bit NF4 (notebook included), not from the A100 bf16 run that produced the rest of its data. The A100 run's own matrix is included and every affected statistic is printed for both; see `data/README.md`.
* **DistilGPT-2 precision.** The notebook loads models of this size in full precision, so DistilGPT-2 ran in fp32; all other checkpoints ran in bf16.
* **JSD diagonals.** Stored matrices have non-zero diagonals (mean within-family distance). No statistic uses them; the analysis code sets them to zero.
* **Permutation p-values.** With 8 families, Mantel *p*-values are exact (all 8! = 40,320 relabellings); the Monte-Carlo values (9,999 permutations) are printed alongside. Section 5.1 has 16 prompts, so its Mantel test is Monte Carlo and the third decimal depends on the seed; the count and seed are fixed in the code. The §5.6 percentages at 5,000 draws vary in the first decimal with the seed; `pqs_f_weighting_robustness.py` also prints 200,000-draw values.

## Software

The analysis scripts were checked with Python 3.13, numpy 2.5, pandas 3.0, scipy 1.18, statsmodels 0.15 and scikit-learn 1.9; earlier versions within the ranges in `requirements.txt` gave identical results where tested.

## Licence and use

No licence is granted yet. The repository is shared so that editors and reviewers can evaluate the manuscript; please do not redistribute it before publication. Model outputs in `data/` are also subject to the licences of the models that generated them.

## Citation

If you use this code or data, please cite the manuscript above (full reference to be added on publication).
