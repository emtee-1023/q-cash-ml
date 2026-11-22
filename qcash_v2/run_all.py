"""End-to-end orchestrator for the Q-Cash v2 rebuild.

Runs the pipeline in order:

    1. generator   -> data/*.csv
    2. features    -> data/supervised.csv
    3. train       -> outputs/model.joblib, outputs/metrics.json
    4. simulate    -> outputs/simulation.csv, outputs/simulation_summary.json
    5. report      -> reports/*.md, plots/*.png
    6. tests       -> pipeline checks

Run from the repo root:  ../.venv/bin/python run_all.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def run(script: str) -> int:
    print("\n" + "=" * 70)
    print(f"RUNNING {script}")
    print("=" * 70)
    return subprocess.run([PY, script], cwd=str(ROOT)).returncode


def main() -> int:
    steps = [
        "generator.py",
        "features.py",
        "train.py",
        "simulate.py",
        "report.py",
        "make_samples.py",
        "tests/test_pipeline.py",
    ]
    for step in steps:
        code = run(step)
        if code != 0:
            print(f"\nStep {step} failed with exit code {code}.")
            return code
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE — see reports/RETRAINING_SUMMARY.md")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
