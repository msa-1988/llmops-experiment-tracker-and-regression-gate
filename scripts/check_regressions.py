from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from app.src.gates import evaluate_gate, save_gate_decision
from app.src.run_registry import load_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = load_run(args.baseline)
    candidate = load_run(args.candidate)
    decision = evaluate_gate(baseline, candidate)
    save_gate_decision(decision)
    print(json.dumps(decision.to_dict(), indent=2))
    if decision.status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
