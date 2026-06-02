from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from app.src.gates import evaluate_gate
from app.src.run_registry import load_all_runs, load_run


def main() -> None:
    subprocess.run([sys.executable, str(ROOT_DIR / "scripts" / "bootstrap_demo_runs.py")], check=True)
    runs = load_all_runs()
    assert len(runs) >= 4, "expected at least 4 runs"

    baseline = load_run("baseline-qwen25-base")
    candidate = load_run("candidate-qwen25-qlora-v1")
    failed_candidate = load_run("candidate-qlora-regressed")

    pass_decision = evaluate_gate(baseline, candidate)
    fail_decision = evaluate_gate(baseline, failed_candidate)

    assert pass_decision.status == "pass", "expected tuned candidate to pass"
    assert fail_decision.status == "fail", "expected regressed candidate to fail"

    subprocess.run(
        [
            sys.executable,
            str(ROOT_DIR / "scripts" / "export_run_report.py"),
            "--baseline",
            baseline.run_id,
            "--candidate",
            candidate.run_id,
        ],
        check=True,
    )
    print("smoke test passed")


if __name__ == "__main__":
    main()
