# GPU experiment environments (as recorded)

The versions below are collected from the stored outputs of the notebooks in this folder
(the environment-check cell prints Python, torch, CUDA build and GPU) and from
`data/surface_vs_behaviour/provenance_manifest.json`. Anything not printed by a notebook is
marked *not recorded*.

| Run (notebook) | Checkpoint | Python | torch (CUDA build) | transformers | GPU | Precision |
|---|---|---|---|---|---|---|
| `main_experiment/distilgpt2.ipynb` | `distilgpt2` | 3.12.3 | 2.12.0+cu130 (13.0) | not recorded | RTX 5070 12 GB | fp32 ("Loaded fp32 (full precision)") |
| `main_experiment/qwen2_5_0_5b.ipynb` | `Qwen/Qwen2.5-0.5B` | 3.12.3 | 2.12.0+cu130 (13.0) | not recorded | RTX 5070 12 GB | bf16 |
| `main_experiment/qwen2_5_0_5b_instruct.ipynb` | `Qwen/Qwen2.5-0.5B-Instruct` | 3.12.3 | 2.12.0+cu130 (13.0) | not recorded | RTX 5070 12 GB | bf16 |
| `main_experiment/qwen2_5_7b.ipynb` | `Qwen/Qwen2.5-7B` | 3.10.20 | 2.11.0+cu130 (13.0) | not recorded | A100-PCIE-40GB | bf16 |
| `main_experiment/llama_3_1_8b_instruct.ipynb` | `meta-llama/Llama-3.1-8B-Instruct` | 3.10.20 | 2.11.0+cu130 (13.0) | not recorded | A100-PCIE-40GB | bf16 |
| `main_experiment/llama_3_1_8b_instruct_rtx5070_nf4.ipynb` | `meta-llama/Llama-3.1-8B-Instruct` (source of the supplementary NF4 JSD matrix only) | 3.12.3 | 2.12.0+cu130 (13.0) | not recorded | RTX 5070 12 GB | 4-bit NF4 (bitsandbytes; version not recorded) |
| `surface_vs_behaviour/surface_vs_behaviour_qwen2_5_7b_instruct.ipynb` | `Qwen/Qwen2.5-7B-Instruct` (16 prompts) | 3.10.20 | 2.11.0+cu130 | 5.12.1 | A100-PCIE-40GB | bf16 |

Appendix A run also records sentence-transformers 5.6.0, scipy 1.15.3, scikit-learn 1.7.2.

**No stored run.** According to the main README, the data for `Qwen/Qwen2.5-7B-Instruct` and `meta-llama/Llama-3.1-8B`
(base) in `data/` were produced by the `qwen2_5_7b.ipynb` and `llama_3_1_8b_instruct.ipynb`
notebooks with their `MODEL_NAME` switched, but those executions were not saved: the
notebooks store only the Qwen2.5-7B and Llama-3.1-8B-Instruct runs. Their environments are
presumably the same A100 setup, but this is not recorded.

**Not recorded anywhere:** the transformers, accelerate, bitsandbytes, sentence-transformers
and rouge-score versions of the main-experiment runs, the NVIDIA driver version, and the
cuDNN version. Greedy generations, and hence every behavioural descriptor, can change with
these. For any new or repeated run, record them, e.g. by adding to the environment-check
cell:

```python
import platform, torch, transformers, accelerate, sentence_transformers, subprocess
print(platform.python_version(), torch.__version__, torch.version.cuda,
      torch.backends.cudnn.version(), transformers.__version__, accelerate.__version__,
      sentence_transformers.__version__)
print(subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout)
```

and save `pip freeze > pip_freeze_<model>.txt` next to the run's outputs.
