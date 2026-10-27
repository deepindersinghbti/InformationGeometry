#!/bin/bash
# Full 5,080-instance re-runs at max_new_tokens=256 (resumable; rerun the same command to continue).
set -e
cd "$(dirname "$0")"
PY=${PY:-python}
BS=${BS:-32}
for M in Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-0.5B distilgpt2; do
  $PY rerun_generation.py --model $M --max-new-tokens 256 --outdir outputs --batch-size $BS
done
