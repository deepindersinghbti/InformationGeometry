"""Re-run the GREEDY generation of the main experiment with a different max_new_tokens.

Faithful to experiments/main_experiment/*.ipynb (cells 3-8, 10) except for max_new_tokens and
(optionally) batching:
  * prompts: instruct models -> tokenizer.apply_chat_template([system "You are a helpful
    assistant.", user "{instruction}\n\nQuestion: {question}"], add_generation_prompt=True);
    base models -> "{instruction}\n\nQuestion: {question}\nAnswer:". Instruct is detected
    exactly as in the notebook: "instruct" in model name (lower-cased).
  * tokenisation: tokenizer(text) with default add_special_tokens (as in the notebook).
  * model: AutoModelForCausalLM, bf16 (fp32 for distilgpt2, as in its notebook), eval.
  * decoding: model.generate(do_sample=False, max_new_tokens=N, pad_token_id=eos), i.e. the
    model's own generation_config otherwise (Qwen-Instruct: repetition_penalty=1.1 applies).
  * output text: decode(new tokens, skip_special_tokens=True).strip()
  * s: cosine(all-MiniLM-L6-v2 emb(output), emb(reference)), embedder.encode() as notebook.
  * ROUGE-L: rouge_score RougeScorer(["rougeL"], use_stemmer=True).score(ref, pred).fmeasure
  * instances: data/prompts/evaluation_tasks.csv x prompt_templates.csv, 127 x 8 x 5 = 5080,
    in the notebook's row order (task, family, variant).
Phi, entropy, NLL, JSD are NOT recomputed.

Batching (--batch-size > 1) uses left padding with an attention mask. Instances are sorted by
prompt length so a batch has little padding; results are written in the notebook's order.

Resumable: rows are appended to <outdir>/<tag>_mnt<N>.csv every --checkpoint-every instances;
on restart, instances already present are skipped.

Usage:
  python rerun_generation.py --model Qwen/Qwen2.5-0.5B-Instruct --max-new-tokens 256 \
      --outdir outputs --batch-size 32
  python rerun_generation.py ... --limit 100        # first 100 instances (notebook order)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from scoring import score_row  # noqa: E402

DATA = HERE.parent.parent / "data" / "prompts"
SYSTEM_PROMPT = "You are a helpful assistant."
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SEED = 42


def model_tag(name: str) -> str:
    return name.split("/")[-1].replace(".", "_").replace("-", "_").lower()


def load_instances() -> pd.DataFrame:
    tasks = pd.read_csv(DATA / "evaluation_tasks.csv")
    tmpl = pd.read_csv(DATA / "prompt_templates.csv")
    fam_order = list(dict.fromkeys(tmpl.family))
    rows = []
    for t in tasks.itertuples():
        for fam in fam_order:
            for v in tmpl[tmpl.family == fam].sort_values("variant").itertuples():
                rows.append(dict(task_id=t.task_id, task_type=t.task_type, question=t.question,
                                 reference=t.reference, family=fam, variant=int(v.variant),
                                 template_id=f"{fam}_{int(v.variant)}",
                                 instruction=v.instruction))
    df = pd.DataFrame(rows)
    df.insert(0, "idx", np.arange(len(df)))
    assert len(df) == 5080
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--outdir", default=str(HERE / "outputs"))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="first N instances only")
    ap.add_argument("--indices", default=None, help="comma-separated instance idx (overrides --limit)")
    ap.add_argument("--checkpoint-every", type=int, default=256)
    ap.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp32"],
                    help="auto = fp32 for distilgpt2 (as its notebook), bf16 otherwise")
    ap.add_argument("--no-sort", action="store_true", help="keep notebook order inside batches")
    ap.add_argument("--suffix", default="", help="extra tag for the output file name")
    args = ap.parse_args()

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sentence_transformers import SentenceTransformer
    from rouge_score import rouge_scorer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_instruct = "instruct" in args.model.lower()
    tag = model_tag(args.model)
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}.get(args.dtype)
    if dtype is None:
        dtype = torch.float32 if "distilgpt2" in args.model.lower() else torch.bfloat16

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir / f"{tag}_mnt{args.max_new_tokens}{args.suffix}.csv"

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device).eval()
    embedder = SentenceTransformer(EMBED_MODEL, device=str(device))
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    gc = model.generation_config
    eos_ids = gc.eos_token_id if isinstance(gc.eos_token_id, list) else [gc.eos_token_id]
    eos_ids = set(int(e) for e in eos_ids if e is not None) | {int(tok.eos_token_id)}

    inst = load_instances()
    if args.indices:
        inst = inst[inst.idx.isin([int(i) for i in args.indices.split(",")])]
    elif args.limit:
        inst = inst.head(args.limit)

    def build(instruction, question):
        if is_instruct:
            msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"{instruction}\n\nQuestion: {question}"}]
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return f"{instruction}\n\nQuestion: {question}\nAnswer:"

    inst = inst.copy()
    inst["prompt_text"] = [build(a, b) for a, b in zip(inst.instruction, inst.question)]
    inst["prompt_tokens"] = [len(tok(p).input_ids) for p in inst.prompt_text]

    done = set()
    if out_csv.exists():
        done = set(pd.read_csv(out_csv, usecols=["idx"]).idx)
    todo = inst[~inst.idx.isin(done)]
    if not args.no_sort:
        todo = todo.sort_values(["prompt_tokens", "idx"], kind="stable")
    print(f"model={args.model} instruct={is_instruct} dtype={dtype} mnt={args.max_new_tokens} "
          f"bs={args.batch_size} todo={len(todo)} done={len(done)} -> {out_csv}", flush=True)

    meta = dict(model=args.model, is_instruct=is_instruct, dtype=str(dtype),
                max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
                sorted_by_length=not args.no_sort, generation_config=gc.to_dict(),
                eos_ids=sorted(eos_ids), torch=torch.__version__,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    (outdir / f"{tag}_mnt{args.max_new_tokens}{args.suffix}_meta.json").write_text(
        json.dumps(meta, indent=2, default=str))

    ref_emb = {r: embedder.encode(r) for r in inst.reference.unique()}
    buf, t0, n_done_run = [], time.time(), 0
    rows = list(todo.itertuples())

    def flush():
        nonlocal buf
        if buf:
            pd.DataFrame(buf).to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)
            buf = []

    for b in range(0, len(rows), args.batch_size):
        batch = rows[b:b + args.batch_size]
        enc = tok([r.prompt_text for r in batch], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        new = out[:, enc.input_ids.shape[1]:].cpu()
        for r, ids in zip(batch, new):
            ids = ids.tolist()
            stop = next((i for i, t in enumerate(ids) if t in eos_ids), None)
            gen = ids if stop is None else ids[:stop + 1]
            text = tok.decode(gen, skip_special_tokens=True).strip()
            n_new = len(gen) if stop is None else stop      # tokens excluding the EOS
            e = embedder.encode(text)
            re_ = ref_emb[r.reference]
            s = float(np.dot(e, re_) / ((np.linalg.norm(e) + 1e-12) * (np.linalg.norm(re_) + 1e-12)))
            rec = dict(idx=r.idx, task_id=r.task_id, task_type=r.task_type, question=r.question,
                       reference=r.reference, family=r.family, variant=r.variant,
                       template_id=r.template_id, prompt_tokens=r.prompt_tokens,
                       output=text, n_new_tokens=n_new, ended_eos=stop is not None,
                       hit_cap=(stop is None and len(gen) >= args.max_new_tokens),
                       semantic_similarity=s,
                       rouge_l=float(rouge.score(r.reference, text)["rougeL"].fmeasure))
            rec.update(score_row(r.task_type, r.question, r.reference, text))
            buf.append(rec)
        n_done_run += len(batch)
        if len(buf) >= args.checkpoint_every:
            flush()
        el = time.time() - t0
        print(f"  {n_done_run}/{len(rows)}  {n_done_run/el:.2f} inst/s  elapsed {el:.0f}s", flush=True)
    flush()
    el = time.time() - t0
    print(f"DONE {n_done_run} instances in {el:.0f}s ({n_done_run/max(el,1e-9):.2f} inst/s); "
          f"peak GPU mem {torch.cuda.max_memory_allocated()/1e9:.2f} GB" if torch.cuda.is_available() else "", flush=True)

    # final tidy: sort rows into notebook order
    if out_csv.exists():
        df = pd.read_csv(out_csv).drop_duplicates("idx", keep="last").sort_values("idx")
        df.to_csv(out_csv, index=False)


if __name__ == "__main__":
    main()
