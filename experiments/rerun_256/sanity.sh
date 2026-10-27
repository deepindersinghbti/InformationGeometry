#!/bin/bash
# Sanity + batching check on 100 instances spread over all tasks/families (idx = 0,51,...,5049):
# 64 and 256 tokens, batch size 1 and 32, then compare with the stored 64-token outputs.
set -e
cd "$(dirname "$0")"
PY=${PY:-python}
M=Qwen/Qwen2.5-0.5B-Instruct
IDX=$(seq -s, 0 51 5049)
for MNT in 64 256; do for BS in 1 32; do
  $PY rerun_generation.py --model $M --max-new-tokens $MNT --outdir outputs/sanity --indices $IDX --batch-size $BS --suffix _bs$BS
done; done
$PY compare_sanity.py > logs/compare_sanity.txt 2>&1
