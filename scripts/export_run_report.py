from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from app.src.gates import evaluate_gate
from app.src.reporting import render_markdown_report, save_markdown_report
from app.src.run_registry import load_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument(
        "--output",
        default=str(ROOT_DIR / "artifacts" / "run_report.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = load_run(args.baseline)
    candidate = load_run(args.candidate)
    decision = evaluate_gate(baseline, candidate)
    report = render_markdown_report(baseline, candidate, decision)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_markdown_report(report, output_path)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
