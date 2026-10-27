"""Run every CPU analysis in order and write the tables to analysis/results/.

    python analysis/run_all.py           # full run, about 10 minutes on a laptop CPU
    python analysis/run_all.py --quick   # 5,000-draw robustness only, 9,999-permutation
                                         # Mantel test for Appendix A (about 4 minutes)

Appendix A needs sentence-transformers and a one-off download of all-MiniLM-L6-v2.
If that package is missing, the script falls back to the saved 2-decimal matrices and
says so.
"""

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(script, *args):
    print(f"\n>>> python analysis/{script} {' '.join(args)}".rstrip(), flush=True)
    subprocess.run([sys.executable, str(HERE / script), *args], check=True, cwd=HERE)


def main(quick: bool):
    run("fisher_fidelity.py")
    run("mixed_effects.py")
    run("fisher_free_pqs.py")
    run("cross_model_agreement.py")
    run("pqs_dispersion.py")
    run("pqs_weighting_ablation.py")
    run("pqs_f_weighting_robustness.py", *(["--quick"] if quick else []))
    sv_args = ["--n-perm", "9999"] if quick else []
    if importlib.util.find_spec("sentence_transformers") is None:
        print("\nsentence-transformers not installed: Appendix A uses the saved 2-decimal "
              "matrices (values close to, but not exactly, the paper's).")
        sv_args.append("--from-matrices")
    run("surface_vs_behaviour.py", *sv_args)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
