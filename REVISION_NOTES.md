# Revision notes and pre-submission review (27 Oct 2026)

Prepared by Abdul Manan Khan for Vishwanath Bijalwan and Deepinder Singh.
Paper: updated on Overleaf (History panel shows every change). Code: this branch, `revision/abdul-2026-10`.

## Summary

**Reviewer verdict: major revision before submission.** The data and analysis reproduce exactly, and the paper now claims only what the evidence supports. Its strongest result: the relational geometry of prompt families carries across models (Mantel r̄ = 0.38; 0.84 among instruction-tuned models), while rankings of the best prompt family do not (Kendall τ̄ = 0.05). Two weaknesses remain: the quality score (embedding similarity) rewards short answers rather than correct ones, and Φ is called a Fisher quantity although it is not one.

## What was changed, why, and where

| Change | Why | Where |
|---|---|---|
| New headline: geometry conserved, rankings not; "concise is best" described as a property of the score | Only finding that survives every check; concise advantage disappears under correctness scoring | Abstract, contributions, §6, §8 |
| 256-token rerun of the three small models with exact-match scoring | 64-token cap truncated 67–88% of cot/structured outputs; at 256 tokens cot = concise (70% vs 70%) and concise ranks last by correctness | Limitation (vi), §5.1; `experiments/rerun_256/` |
| Precision disclosed (DistilGPT-2 fp32); bf16 Llama-Instruct JSD matrix now primary, NF4 values also reported | Paper said all bf16; the JSD matrix came from a 4-bit run | §4, Table 1, §5.4, §5.6, Figs 3–5; `data/jsd_matrices/`, `analysis/cross_model_agreement.py` |
| 6 of 7 slopes significant (was "all 7"); equivalence test; interaction tests; robustness table | Overstated; Llama weakening depends on model specification | §5.2, Tables 4–5 |
| Fisher-free result restated within its evidence | PQS⁻ᶠ ranks families exactly as s in two instruct models; "all three negative" holds in 21% of weightings | §5.3, Table 6, §5.5 |
| Methods corrected to match the code (σ²_s, entropy, prompt format, JSD diagonal, Mantel) | Text described computations the code does not do | §3, §4 |
| Figs 6–7 (were 8–9) now plot Φ vs s | They plotted PQS but quoted ρ(Φ, s) | `figures/scripts/fisher_vs_pqs.py` |
| Pilot study → Appendix A; effective dimension and curvature index removed | Not part of the argument; never analysed | Appendix A, §3 |
| Contradictions removed, bibliography fixed, Fig 4 fitted to page | Consistency and clean build | Throughout |
| Repo: checksums fixed, licences (proposed), CITATION.cff, pinned versions, provenance notes | 5 checksums failed; no licence or pinning | `data/SHA256SUMS`, `LICENSE*`, `requirements-lock.txt`, `experiments/ENVIRONMENT.md` |

## Reviewer comments: major

- **M1. The quality score does not measure correctness.** Rerun the 7B/8B models at 256 tokens (`experiments/rerun_256/run_large_a100.sh`, about $3–6 on an A100, needs a Hugging Face token for Llama), regenerate the entropy samples at 256 tokens, and report exact-match accuracy alongside s. *Owner: Deepinder; Manan updates text.*
- **M2. Instruction tuning is confounded with input format.** Base models get plain text, instruct models a chat template. Run the 0.5B instruct model with the plain format (and ideally base with the chat template). *Owner: Deepinder.*
- **M3. Φ mostly reflects the task, not the prompt wording.** Prompt length does not drive Φ (|ρ| ≤ 0.26), but 45–91% of its variance is explained by task and only 6–37% by template. Report this and make the task + template mixed model primary. *Owner: Manan.*
- **M4. The headline compares two different statistics** (Kendall on 8 scalars vs Mantel on 28 distances). Add Mantel on |PQS_i − PQS_j| and |s_i − s_j| matrices. Already checked: prompt-text similarity does not explain the JSD geometry (Mantel r −0.50 to +0.10, all n.s.); add this to the paper. *Owner: Manan.*
- **M5. "Fisher"/"information geometry" overstates the method.** Either rename Φ (e.g. "trajectory dispersion") and retitle, or compute a genuine Fisher quantity (next-token Fisher information pulled back to input embeddings) on the 0.5B models. *Owner: all authors decide; Vishwanath for title.*
- **M6. Two runs have no stored record** (Qwen2.5-7B-Instruct, Llama-3.1-8B base). Add logs or rerun. *Owner: Deepinder.*
- **M7. Length (33 pp) and language.** Halve Related Work, merge §5.3 into §5.2, move robustness tables to an appendix, full language pass. *Owner: Manan; Vishwanath reviews.*

## Reviewer comments: minor

1. Disclose `repetition_penalty = 1.1` for Qwen-Instruct (from its generation config). *Manan; Deepinder to confirm.*
2. JSD geometry uses only the canonical wording and first token; report stability with a paraphrase. *Deepinder.*
3. Repeat s with a second embedding model. *Manan.*
4. Family-level tests (n = 8) are uncorrected; say so or apply Holm. *Manan.*
5. Data availability says the repo is restricted, but it is public. *Deepinder, Vishwanath.*
6. Admin: equal-contribution mark, who paid for the A100, the "summer internship" sentence. *Vishwanath.*
7. Code: syntax error on the σ²_s line in the instruct notebook; figure notebooks need a local data path instead of Colab uploads. *Deepinder.*
8. Licences (MIT code, CC-BY-4.0 data) are a proposal; all authors to agree. *All.*

## Where to submit

1. **TMLR (Transactions on Machine Learning Research)** — recommended. Judges whether claims are supported rather than novelty or scale; rolling submission, open review, no fees. Needs the TMLR template; do M1–M5 first.
2. **Neural Computing and Applications (Springer)** — if we keep the sn-jnl template and want to submit sooner (after at least M1 and M7). *Machine Learning* (Springer) is the stronger Springer option, after M1–M5.
3. **Findings of ACL / EMNLP (ACL Rolling Review)** — closest topical fit, but needs cutting to 8 pages; BlackboxNLP workshop is a faster interpretability option.

Avoid Springer's *Information Geometry* journal unless M5 is done with a genuine Fisher quantity.

## Actions by person

- **Vishwanath:** review Overleaf changes; decide title/"Fisher" naming (M5) and venue; admin items (minor 6); author order and contributions.
- **Deepinder:** review and merge this branch; A100 rerun (M1) and the two unrecorded runs (M6); format control (M2); minor 1, 2, 5, 7.
- **Manan:** M3, M4, minor 3 and 4, then shortening and language pass (M7).
