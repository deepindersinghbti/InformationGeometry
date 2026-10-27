# Generation-length check: re-run with `max_new_tokens = 256`

The main experiment capped every generation at 64 new tokens. In the instruction-tuned models
this cuts off most `cot` and `structured` answers (67–88 %) and almost no `concise` answers.
This folder re-runs the greedy generation with a 256-token budget, scores arithmetic and
factual-QA answers by exact match, and compares rankings and statistics with the 64-token run.
It backs limitation (vi) and the §5.1 caveat of the revised manuscript.

Done so far: DistilGPT-2, Qwen2.5-0.5B, Qwen2.5-0.5B-Instruct (RTX 4050 6 GB).
**Not yet run: the four 7B/8B checkpoints** (`run_large_a100.sh`, needs a ≥ 24 GB GPU).

## Files

| File | What |
|---|---|
| `rerun_generation.py` | Re-runs the notebooks' greedy generation and computes s, ROUGE-L and the answer scores (`--model --max-new-tokens --outdir --batch-size --limit/--indices`; resumable). |
| `scoring.py` | Answer extraction: arithmetic exact match, factual-QA normalised containment. Self-test: `python scoring.py`. |
| `compare_sanity.py` | 64-token re-run vs the stored outputs; batch size 1 vs 32; 64 vs 256 prefix check. |
| `analyse_rerun.py` | CPU analysis: per-family tables, rankings, recomputed PQS / PQS⁻ᶠ, ρ(Φ, s), ρ(Φ, accuracy), mixed model, for both budgets. |
| `sanity.sh` | 100-instance sanity and batching check (Qwen2.5-0.5B-Instruct). |
| `full_runs.sh` | Full 5,080-instance runs of the three small models. |
| `run_large_a100.sh` | Full runs of Qwen2.5-7B(-Instruct) and Llama-3.1-8B(-Instruct), then the analysis. |
| `outputs/` | Per-instance re-run CSVs (`<model>_mnt256.csv`) and `_meta.json` (generation config, library versions, GPU). |
| `outputs/sanity/` | The 100-instance sanity runs. |
| `results/` | `per_family_64_vs_256.csv`, `rank_orders_64_vs_256.csv`, `stats_64_vs_256.csv`. `analyse_rerun.py` also writes per-instance `*_scored.csv` files (not committed; regenerated in seconds). |
| `logs/` | `analysis.txt` (all tables), `compare_sanity.txt`, `sanity.log`. |
| `pip_freeze_gpu.txt` | Exact environment of the runs. |

## How to run

```bash
pip install -r requirements-experiments.txt     # from the repository root; GPU torch build
cd experiments/rerun_256
python scoring.py                               # self-test
PY=python ./sanity.sh                           # optional, ~5 min on a small GPU
PY=python ./full_runs.sh                        # three small models
export HF_TOKEN=...                             # Llama 3.1 is gated: accept the licence first
PY=python ./run_large_a100.sh                   # four 7B/8B models, then the analysis
python analyse_rerun.py --mnt 256 > logs/analysis.txt
```

`analyse_rerun.py` reads the stored 64-token results from `../../data/full_results/` and the
prompts from `../../data/prompts/`, and analyses every model whose `outputs/<model>_mnt256.csv`
exists. Estimated time on an A100 40 GB at batch size 32: about 20–40 min per 7B/8B model.

## What is reproduced exactly from the notebooks

* Instances: `data/prompts/evaluation_tasks.csv` × `prompt_templates.csv` = 127 × 8 × 5 = 5,080, in
  notebook row order (checked against the notebook cells and `data/full_results`).
* Prompt: instruct models (`"instruct"` in the model id) get the chat template with system message
  "You are a helpful assistant." and user turn `"{instruction}\n\nQuestion: {question}"`,
  `add_generation_prompt=True`; base models get `"{instruction}\n\nQuestion: {question}\nAnswer:"`.
* bf16 (fp32 for DistilGPT-2, as in its notebook), `do_sample=False`, `pad_token_id=eos`, otherwise
  the model's own `generation_config`. For Qwen2.5-Instruct this means `repetition_penalty = 1.1`
  is active under greedy decoding, as in the original runs.
* Output = `decode(new tokens, skip_special_tokens=True).strip()`; s = cosine of all-MiniLM-L6-v2
  vectors; ROUGE-L = `rouge_score` rougeL F with stemming.
* The only differences are `max_new_tokens` and batching (left padding; see the sanity check).

## Not recomputed

Φ, reference NLL and the first-token JSD come from single forward passes over the prompt (and the
reference), so they do not depend on the budget. **Entropy (H_e) and sample stability do**: they
come from three sampled generations (T = 0.7 / 0.8 / 0.9, top-k 50) that were also capped at 64
tokens. PQS and PQS⁻ᶠ recomputed here therefore still contain a 64-token entropy term.

## Results (27 Oct 2026, three small models)

* **Sanity:** the 64-token re-run reproduces 70 % of the stored outputs verbatim (differences are
  mostly in long `cot` outputs: GPU and library numerics). Batch size 32 vs 1 changes the answer
  score in 3–8 % of instances.
* **Qwen2.5-0.5B-Instruct:** capped share 36 % → 4 %. Arithmetic exact match for `cot` 13 % → 70 %
  (`concise` 69 % → 70 %). Family ranking by s, PQS and PQS⁻ᶠ unchanged (Kendall τ = 1);
  ρ(Φ, s) = −0.71 at both budgets; mixed-model slope −0.033 → −0.028. Ranked by exact match,
  `concise` moves to last (τ = −0.11 with the 64-token ranking); ρ(Φ, accuracy) −0.79 (p = 0.02)
  → +0.48 (p = 0.23).
* **Qwen2.5-0.5B:** capped share 29 % → 10 %; s ranking unchanged; accuracy ranking τ = 0.47;
  Φ–accuracy association null at both budgets.
* **DistilGPT-2:** capped in every family at both budgets (never emits EOS); accuracy at floor (0–7 %).
* **Conclusion:** results based on s are robust to the budget, but s rewards short answers and does
  not measure correctness. The "concise best, cot/structured worst" ordering in instruction-tuned
  models is a property of the score, not of answer correctness.
