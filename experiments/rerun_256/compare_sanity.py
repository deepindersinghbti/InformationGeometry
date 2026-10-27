"""Compare sanity re-runs with the stored data and with each other (bs=1 vs batched).

Run: python compare_sanity.py   (reads outputs/sanity/*.csv)
"""
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
S = HERE / "outputs" / "sanity"
ORIG = HERE.parent.parent / "data/full_results/qwen2_5_0_5b_instruct_full_results.csv"
K = ["question", "family", "variant"]
tag = "qwen2_5_0_5b_instruct"


def cmp(a, b, la, lb):
    m = a.merge(b, on=K, suffixes=("_a", "_b"))
    same = (m.output_a.fillna("") == m.output_b.fillna(""))
    ds = (m.semantic_similarity_a - m.semantic_similarity_b).abs()
    dr = (m.rouge_l_a - m.rouge_l_b).abs()
    print(f"{la} vs {lb}: n={len(m)} identical text {same.mean()*100:.1f}% | "
          f"|ds| mean {ds.mean():.4f} max {ds.max():.4f} | |s diff|<1e-4: {(ds<1e-4).mean()*100:.1f}% "
          f"| |dROUGE| mean {dr.mean():.4f}")
    if "answer_correct_a" in m and "answer_correct_b" in m:
        x = m.dropna(subset=["answer_correct_a", "answer_correct_b"])
        print(f"    answer-score agreement {(x.answer_correct_a == x.answer_correct_b).mean()*100:.1f}% (n={len(x)})")
    # where do differing outputs first diverge?
    diff = m[~same]
    pref = [len(__import__("os").path.commonprefix([x, y])) for x, y in zip(diff.output_a.fillna(""), diff.output_b.fillna(""))]
    if pref:
        print(f"    differing: {len(diff)}; common prefix chars median {pd.Series(pref).median():.0f}; "
              f"families {diff.family.value_counts().to_dict()}")
    return m


orig = pd.read_csv(ORIG)
r = {f"{n}_{b}": pd.read_csv(S / f"{tag}_mnt{n}_{b}.csv") for n in (64, 256) for b in ("bs1", "bs32")
     if (S / f"{tag}_mnt{n}_{b}.csv").exists()}
if "64_bs1" in r:
    cmp(r["64_bs1"], orig, "rerun64 bs1", "original")
if "64_bs32" in r:
    cmp(r["64_bs32"], orig, "rerun64 bs32", "original")
if "64_bs1" in r and "64_bs32" in r:
    cmp(r["64_bs1"], r["64_bs32"], "rerun64 bs1", "rerun64 bs32")
if "256_bs1" in r and "256_bs32" in r:
    cmp(r["256_bs1"], r["256_bs32"], "rerun256 bs1", "rerun256 bs32")
if "64_bs1" in r and "256_bs1" in r:
    a, b = r["64_bs1"], r["256_bs1"]
    m = a.merge(b, on=K, suffixes=("_64", "_256"))
    pre = [y.startswith(x) for x, y in zip(m.output_64.fillna(""), m.output_256.fillna(""))]
    print(f"rerun64 output is a prefix of rerun256 output (same bs=1): {pd.Series(pre).mean()*100:.1f}%")
# validate the >=60 re-tokenisation proxy for 'capped' against the exact flag
if "64_bs1" in r:
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    a = r["64_bs1"]
    n = [len(tok(t, add_special_tokens=False).input_ids) for t in a.output.fillna("")]
    proxy = pd.Series(n) >= 60
    print(f"capped proxy (retok>=60) vs exact hit_cap at 64: agreement {(proxy.values == a.hit_cap.values).mean()*100:.1f}% "
          f"(exact capped {a.hit_cap.mean()*100:.1f}%, proxy {proxy.mean()*100:.1f}%)")
