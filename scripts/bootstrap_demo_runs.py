from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from app.src.run_registry import RunRecord, save_run


SOURCE_METRICS = (
    ROOT_DIR.parent / "03 - Fine-Tune and Optimize a Small Domain LLM" / "artifacts" / "evaluation_metrics.json"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_seed_runs() -> list[RunRecord]:
    metrics = json.loads(SOURCE_METRICS.read_text(encoding="utf-8"))
    base = metrics["base"]
    tuned = metrics["tuned"]

    return [
        RunRecord(
            run_id="baseline-qwen25-base",
            model_name="Qwen2.5-0.5B-Instruct",
            stage="production",
            source_project="03 - Fine-Tune and Optimize a Small Domain LLM",
            created_at=utc_now(),
            notes="Production reference seeded from the base summarization model.",
            stack=["PyTorch", "Transformers", "ROUGE-L", "promotion-baseline"],
            metrics=base,
            tags={"family": "scientific-tldr", "runtime": "base"},
        ),
        RunRecord(
            run_id="candidate-qwen25-qlora-v1",
            model_name="Qwen2.5-0.5B-Instruct + QLoRA adapter",
            stage="staging",
            source_project="03 - Fine-Tune and Optimize a Small Domain LLM",
            created_at=utc_now(),
            notes="First staging candidate seeded from the real fine-tuned run.",
            stack=["PyTorch", "QLoRA", "PEFT", "evaluation-gated"],
            metrics=tuned,
            tags={"family": "scientific-tldr", "runtime": "adapter-v1"},
        ),
        RunRecord(
            run_id="candidate-qlora-optimized",
            model_name="Qwen2.5-0.5B-Instruct + optimized adapter",
            stage="staging",
            source_project="04 - LLMOps Experiment Tracker and Regression Gate",
            created_at=utc_now(),
            notes="Synthetic optimized candidate with slightly better quality and tighter latency.",
            stack=["PyTorch", "QLoRA", "quantized-inference", "latency-tuning"],
            metrics={
                "rougeL_mean": 0.3514,
                "latency_mean_seconds": 0.7020,
                "generated_tokens_mean": 24.90,
                "prompt_tokens_mean": tuned["prompt_tokens_mean"],
            },
            tags={"family": "scientific-tldr", "runtime": "adapter-v2"},
        ),
        RunRecord(
            run_id="candidate-qlora-regressed",
            model_name="Qwen2.5-0.5B-Instruct + unstable adapter",
            stage="staging",
            source_project="04 - LLMOps Experiment Tracker and Regression Gate",
            created_at=utc_now(),
            notes="Synthetic negative-control run used to demonstrate gate failures.",
            stack=["PyTorch", "QLoRA", "misconfigured-runtime"],
            metrics={
                "rougeL_mean": 0.2810,
                "latency_mean_seconds": 0.8910,
                "generated_tokens_mean": 42.10,
                "prompt_tokens_mean": tuned["prompt_tokens_mean"],
            },
            tags={"family": "scientific-tldr", "runtime": "adapter-regressed"},
        ),
    ]


def main() -> None:
    for run in build_seed_runs():
        save_run(run)
        print(f"registered {run.run_id}")


if __name__ == "__main__":
    main()
