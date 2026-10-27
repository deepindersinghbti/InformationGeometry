# Data

Everything the analyses and figures read. All files are plain CSV/TXT/JSON; `SHA256SUMS` lists a checksum for each one; verify them with `sh scripts/verify_checksums.sh` from the repository root (or `sha256sum -c SHA256SUMS` from this folder). Checksums are over the LF-terminated files as stored in git; `.gitattributes` forces LF on checkout, so they verify on every OS.

## `prompts/` — evaluation set and prompt templates

| File | Rows | Contents |
|---|---|---|
| `evaluation_tasks.csv` | 127 | `task_id`, `task_type` (arithmetic 24, factual_qa 23, reasoning 22, safety_grounding 20, instruction_following 20, summarization 18), `question`, `reference` (the short ideal answer used for *s*, ROUGE-L and reference NLL) |
| `prompt_templates.csv` | 40 | `family` (8), `variant` (0–4), `canonical` (variant 0, used for the first-token JSD), `instruction` |

These are exported verbatim from the dataset and prompt-family cells of the experiment notebooks, which are identical in all five notebooks. Every model saw every task × template: 127 × 8 × 5 = 5,080 instances.

Prompt construction (from the notebooks): base models get plain text `"{instruction}\n\nQuestion: {question}\nAnswer:"`; instruction-tuned models get their chat template with system prompt `"You are a helpful assistant."` and user turn `"{instruction}\n\nQuestion: {question}"`.

## `full_results/` — one row per instance (5,080 rows per model)

| Column | Meaning (symbol in the paper) |
|---|---|
| `task_type`, `question` | the task (join to `prompts/evaluation_tasks.csv` on `question`) |
| `family`, `variant` | prompt family and paraphrase variant (0 = canonical) |
| `semantic_similarity` | cosine similarity between the greedy output and the reference, all-MiniLM-L6-v2 embeddings (*s*) |
| `rouge_l` | ROUGE-L F-measure of the greedy output against the reference (*r*) |
| `ref_nll` | negative log-likelihood of the reference given the prompt (ℓ) |
| `entropy` | mean per-token entropy over 3 sampled generations at T = 0.7, 0.8, 0.9 (*H_e*) |
| `samp_stability` | SD of the cosine similarity of those 3 samples to the reference |
| `fisher_trace` | trace of the covariance of layer-to-layer hidden-state displacements + 1e-4·I (Φ) |
| `d_eff` | participation ratio of that covariance's eigenvalues (effective dimension) |
| `robustness_index` | trajectory-curvature robustness, 1 / (1 + mean \|2nd derivative\|) over the first 64 hidden dimensions |
| `output` | the greedy output text (up to 64 new tokens) |
| `paraphrase_var` | variance of *s* over the 5 variants of the same task and family (σ²_s); identical within a task × family |
| `adv_robustness` | perturbation robustness *R*: noise-word and jailbreak-prefix variants of the canonical prompt on the first 8 tasks; one value per family, repeated on every row |
| `PQS` | z(s) + z(r) − z(ℓ) − z(H_e) − z(Φ) − z(σ²_s), each term z-scored over the 5,080 instances (population SD), Eq. (3) |

## `per_family/` — family means (8 rows per model)

Means of the per-instance columns by `family`, rounded to 4 decimals (`sem_sim`, `rouge_l`, `ref_nll`, `entropy`, `samp_stab`, `fisher_trace`, `paraphrase_var`, `adv_robustness`, `PQS`). They equal the group means of `full_results/` to within 5e-5.

## `jsd_matrices/` — first-token geometry (8 × 8 per model)

Off-diagonal entry (*i*, *j*): Jensen–Shannon distance between the mean first-token distributions of families *i* and *j* (canonical variant, 127 tasks). Diagonal entry (*i*, *i*): mean pairwise distance among the 127 first-token distributions *within* family *i*. This is a different quantity, so the diagonal is not zero. No statistic in the paper uses the diagonal; the analysis code sets it to zero on load. Values are stored rounded to 3 decimals.

### Provenance note: Llama-3.1-8B-Instruct

For six models, the full results, family means and JSD matrix come from the same run of the notebook in `experiments/main_experiment/`; for two of them (Qwen2.5-7B-Instruct and Llama-3.1-8B base) that run's executed output was not saved, so no stored notebook output backs their files (see the Models table). For **Llama-3.1-8B-Instruct** the files come from two different runs:

* `full_results/` and `per_family/` come from the A100 bf16 run whose outputs are stored in `experiments/main_experiment/llama_3_1_8b_instruct.ipynb` (22 June 2026).
* `jsd_matrices/llama_3_1_8b_instruct_jsd_matrix.csv` is the JSD matrix from the A100 bf16 run itself. The revised manuscript uses this matrix.
* `jsd_matrices/supplementary/llama_3_1_8b_instruct_jsd_matrix_rtx5070_nf4.csv` comes from an earlier run of the same pipeline code (identical dataset, prompt, generation and metric cells) on a local RTX 5070 (12 GB) on 15 June 2026, with the model loaded in **4-bit NF4** quantisation (`LOAD_IN_4BIT = True`). That run's notebook is included as `experiments/main_experiment/llama_3_1_8b_instruct_rtx5070_nf4.ipynb`, and the matrix here is byte-identical to the file it wrote.

The first submission used the NF4 matrix; the revised manuscript uses the bf16 matrix for the JSD heatmaps, MDS manifolds, Mantel statistics and mean inter-family distances. `analysis/cross_model_agreement.py` prints every Mantel statistic for both:

| Quantity | Earlier NF4 matrix | A100 bf16 matrix (used) |
|---|---|---|
| Mean inter-family JSD, Llama-3.1-8B-Instruct | 0.49 | 0.50 |
| Mantel *r*, Llama-3.1-8B vs Instruct (exact *p*) | 0.52 (0.064) | 0.55 (0.033) |
| Mantel *r*, Qwen2.5-0.5B-Instruct vs Llama-3.1-8B-Instruct | 0.73 | 0.79 |
| Mantel *r*, Qwen2.5-7B-Instruct vs Llama-3.1-8B-Instruct | 0.91 | 0.91 |
| Mean Mantel *r*, instruction-tuned trio | 0.82 | 0.84 |
| Mean Mantel *r*, all 21 pairs | 0.38 | 0.38 |

## `surface_vs_behaviour/` — Appendix A preliminary experiment

| File | Contents |
|---|---|
| `recursion_real_outputs.csv` | the 16 prompts (`prompt_id`, `category`, `prompt_text`) and Qwen2.5-7B-Instruct's greedy outputs (`output_text`, up to 512 new tokens) |
| `provenance_manifest.json` | model, dtype (bf16), decoding settings, embedder, headline statistics, library versions and GPU of that run |
| `prompt_similarities.txt`, `output_similarities.txt` | the two 16 × 16 cosine matrices, rounded to 2 decimals (as printed in Fig. A1) |

The statistics in the paper use full-precision embeddings recomputed from `recursion_real_outputs.csv`; the 2-decimal matrices reproduce them only approximately (Pearson *r* 0.4875 vs 0.4837).

## Models

| Model | Hugging Face id | Hardware | Precision | Stored executed run |
|---|---|---|---|---|
| DistilGPT-2 | `distilgpt2` | RTX 5070 12 GB | **fp32** | yes |
| Qwen2.5-0.5B | `Qwen/Qwen2.5-0.5B` | RTX 5070 12 GB | bf16 | yes |
| Qwen2.5-0.5B-Instruct | `Qwen/Qwen2.5-0.5B-Instruct` | RTX 5070 12 GB | bf16 | yes |
| Qwen2.5-7B | `Qwen/Qwen2.5-7B` | A100-PCIE-40GB | bf16 | yes |
| Qwen2.5-7B-Instruct | `Qwen/Qwen2.5-7B-Instruct` | A100-PCIE-40GB | bf16 | **no** |
| Llama-3.1-8B | `meta-llama/Llama-3.1-8B` | A100-PCIE-40GB | bf16 | **no** |
| Llama-3.1-8B-Instruct | `meta-llama/Llama-3.1-8B-Instruct` | A100-PCIE-40GB | bf16 (an earlier RTX 5070 4-bit NF4 JSD matrix is in `jsd_matrices/supplementary/`; see the note above) | yes (both runs) |

"Stored executed run": whether a notebook in `experiments/main_experiment/` holds the executed output of the run that produced the files. For Qwen2.5-7B-Instruct and Llama-3.1-8B the run was made by switching `MODEL_NAME` in `qwen2_5_7b.ipynb` and `llama_3_1_8b_instruct.ipynb` respectively, but those executions were not saved.

Generated text in this folder is model output; the licence of each model (see its Hugging Face model card, e.g. the Llama 3.1 Community License) also applies to it.
