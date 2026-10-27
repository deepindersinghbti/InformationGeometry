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
  main_experiment/         5 notebooks with stored runs for 5 of the 7 checkpoints, plus
                           the earlier 4-bit NF4 Llama-3.1-8B-Instruct run that produced
                           its JSD matrix (no stored runs for Qwen2.5-7B-Instruct and
                           Llama-3.1-8B base; see "Notes for reviewers")
  surface_vs_behaviour/    Section 5.1 generation run (Qwen2.5-7B-Instruct)
analysis/                  CPU scripts that reproduce every table and statistic from data/
  results/                 the tables they write (committed for comparison)
figures/                   submitted figure PDFs and the code that draws them (see figures/README.md)
requirements.txt           CPU analysis + figures (lower bounds)
requirements-lock.txt      CPU analysis + figures, exact versions verified to reproduce analysis/results/
requirements-experiments.txt   GPU experiments (see experiments/ENVIRONMENT.md for the recorded versions)
LICENSE                    MIT licence (code)
LICENSE-DATA.md            CC BY 4.0 (data, figures) and upstream model licences
CITATION.cff               citation metadata
```

## Quick start (CPU only, no model downloads)

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-lock.txt   # exact versions verified (Python 3.12); or: -r requirements.txt
python analysis/run_all.py            # about 10 minutes; add --quick for about 4 minutes
```

Each script can also be run on its own, e.g. `python analysis/mixed_effects.py`. The scripts read only `data/` and write CSV tables to `analysis/results/`.

## Where each result in the manuscript comes from

| Manuscript | Result | Reproduced by |
|---|---|---|
| Tables 1–2, §4 | models, prompt families, 127 tasks | `data/README.md`, `data/prompts/` |
| §5.1, Table 3, Figs. 2–3 | mean prompt / output cosine 0.694 / 0.671; Pearson *r* = 0.484; Spearman ρ = 0.144 (*p* = 0.116); Mantel *p* = 0.060 (999,999 permutations, seed 42) | `analysis/surface_vs_behaviour.py` (→ `results/surface_vs_behaviour.csv`), `figures/notebooks/fig3_similarity_scatter.ipynb` |
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
| `main_experiment/qwen2_5_7b.ipynb` | `Qwen/Qwen2.5-7B` (stored run) and `Qwen/Qwen2.5-7B-Instruct` (**run not stored**) | A100-PCIE-40GB | bf16 | 6.6 h |
| `main_experiment/llama_3_1_8b_instruct.ipynb` | `meta-llama/Llama-3.1-8B-Instruct` (stored run) and `meta-llama/Llama-3.1-8B` (**run not stored**) | A100-PCIE-40GB | bf16 | 10.6 h |
| `main_experiment/llama_3_1_8b_instruct_rtx5070_nf4.ipynb` | `meta-llama/Llama-3.1-8B-Instruct`, earlier run; source of its JSD matrix used in the paper (only) | RTX 5070, 12 GB | **4-bit NF4** | 5.9 h |
| `surface_vs_behaviour/surface_vs_behaviour_qwen2_5_7b_instruct.ipynb` | `Qwen/Qwen2.5-7B-Instruct` (16 prompts) | A100-PCIE-40GB | bf16 | minutes |

```bash
pip install -r requirements-experiments.txt     # after installing a torch build for your GPU
```

Open a notebook and run all cells. To run the second checkpoint listed for a notebook, make its `MODEL_NAME` line the active one. Each run writes `<model>_full_results.csv`, `<model>_per_family.csv` and `<model>_jsd_matrix.csv` to `results_<model>/`; the copies used in the paper are in `data/`. The dataset, prompt-family, generation and metric cells are identical in all six main-experiment notebooks. The Llama models are gated: accept the licence on Hugging Face and export `HF_TOKEN`. No token is stored anywhere in this repository.

All runs use seed 42. Greedy generations are deterministic on a given GPU, dtype and library version. A re-run on other hardware or library versions can change individual generated texts and the values derived from them. The CSVs in `data/` are the exact outputs analysed in the paper.

## Notes for reviewers

**Precision and provenance at a glance.** All runs used bf16 with two exceptions: **DistilGPT-2 ran in fp32**, and the **Llama-3.1-8B-Instruct JSD matrix used in the main analysis (Figs. 5–7, every Mantel statistic, mean inter-family JSD) comes from a 4-bit NF4 run**; the bf16 A100 matrix of the same model is in `data/jsd_matrices/supplementary/`, and `analysis/cross_model_agreement.py` reports both (`mantel_pairwise_results_llama_instruct_a100_rerun.csv`). Executed notebooks exist for five checkpoints; **for Qwen2.5-7B-Instruct and Llama-3.1-8B (base) no executed notebook output is stored**, so their CSVs in `data/` cannot be traced to a stored run.

| Model | Precision of `full_results/`, `per_family/` | Precision of JSD matrix used | Stored executed run |
|---|---|---|---|
| DistilGPT-2 | **fp32** | fp32 | `distilgpt2.ipynb` |
| Qwen2.5-0.5B | bf16 | bf16 | `qwen2_5_0_5b.ipynb` |
| Qwen2.5-0.5B-Instruct | bf16 | bf16 | `qwen2_5_0_5b_instruct.ipynb` |
| Qwen2.5-7B | bf16 | bf16 | `qwen2_5_7b.ipynb` |
| Qwen2.5-7B-Instruct | bf16 | bf16 | **none** |
| Llama-3.1-8B | bf16 | bf16 | **none** |
| Llama-3.1-8B-Instruct | bf16 | **4-bit NF4** (bf16 matrix in `supplementary/`) | `llama_3_1_8b_instruct.ipynb` (bf16), `llama_3_1_8b_instruct_rtx5070_nf4.ipynb` (NF4) |

* **Llama-3.1-8B-Instruct JSD matrix.** The JSD matrix used in the manuscript for this model comes from an earlier run of the same pipeline on the RTX 5070 in 4-bit NF4 (notebook included), not from the A100 bf16 run that produced the rest of its data. The A100 run's own matrix is included and every affected statistic is printed for both; see `data/README.md`.
* **DistilGPT-2 precision.** The notebook loads models of this size in full precision, so DistilGPT-2 ran in fp32 (the stored run prints "Loaded fp32 (full precision)"); all other checkpoints ran in bf16, except for the Llama-3.1-8B-Instruct JSD matrix (4-bit NF4, above).
* **JSD diagonals.** Stored matrices have non-zero diagonals (mean within-family distance). No statistic uses them; the analysis code sets them to zero.
* **Permutation p-values.** With 8 families, Mantel *p*-values are exact (all 8! = 40,320 relabellings); the Monte-Carlo values (9,999 permutations) are printed alongside. Section 5.1 has 16 prompts, so its Mantel test is Monte Carlo and the third decimal depends on the seed; the count and seed are fixed in the code. The §5.6 percentages at 5,000 draws vary in the first decimal with the seed; `pqs_f_weighting_robustness.py` also prints 200,000-draw values.

## Software

`python analysis/run_all.py` reproduces every committed table in `analysis/results/` bit-for-bit (largest difference 2.2e-16) with Python 3.12.3 on Linux x86_64 and the exact versions in `requirements-lock.txt` (numpy 2.5.4, pandas 3.1.0, scipy 1.18.1, statsmodels 0.15.0, scikit-learn 1.9.1, sentence-transformers 6.2.0, torch 2.14.1+cpu). `requirements.txt` lists looser lower bounds; other versions within them are expected to give the same tables but have not all been verified.

The GPU experiments ran in different environments (Python 3.10.20 / 3.12.3, torch 2.11.0+cu130 / 2.12.0+cu130). `experiments/ENVIRONMENT.md` lists what each notebook recorded and what was not recorded (transformers version of the main runs, NVIDIA driver).

## Licence and use

Code is released under the MIT License (`LICENSE`). Data in `data/` (including the prompts) and the figure files in `figures/pdf/` are released under CC BY 4.0 (`LICENSE-DATA.md`). Model outputs in `data/` are also subject to the licences of the models that generated them: the Llama 3.1 Community License Agreement for the Llama-3.1-8B models and Apache-2.0 for the Qwen2.5 models and DistilGPT-2 (details in `LICENSE-DATA.md`).

## Citation

If you use this code or data, please cite the manuscript above (full reference to be added on publication). `CITATION.cff` holds the citation metadata (GitHub shows it under "Cite this repository").
