#!/bin/bash
# 256-token re-run of the four 7B/8B checkpoints (not yet run). Needs a GPU with >= 24 GB
# (A100 40 GB recommended) and access to the gated Llama 3.1 models: accept the licence on
# Hugging Face and `export HF_TOKEN=...` first. Resumable: rerun the same command to continue.
# Expected time on an A100 40 GB at batch size 32: roughly 20-40 min per model.
set -e
cd "$(dirname "$0")"
PY=${PY:-python}
BS=${BS:-32}
for M in Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-7B meta-llama/Llama-3.1-8B-Instruct meta-llama/Llama-3.1-8B; do
  $PY rerun_generation.py --model $M --max-new-tokens 256 --outdir outputs --batch-size $BS
done
$PY analyse_rerun.py --mnt 256 > logs/analysis.txt 2>&1
echo "Done. Commit outputs/*_mnt256.csv, outputs/*_meta.json, results/*.csv and logs/analysis.txt."
