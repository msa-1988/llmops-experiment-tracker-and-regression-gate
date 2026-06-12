from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from app.src.gates import evaluate_gate, save_gate_decision
from app.src.run_registry import RunRecord, load_run, save_run


@dataclass(frozen=True)
class EvalExample:
    """One summarization-style evaluation item.

    In a production LLMOps system this would come from a held-out dataset and
    include prompt text, reference answer, task metadata, and maybe safety tags.
    Here it stays synthetic and deterministic so the multi-GPU mechanics are
    easy to rerun on Kaggle without external model downloads.
    """

    prompt: str
    reference_summary: str
    difficulty: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distributed LLM evaluation practice run for the LLMOps gate.")
    parser.add_argument("--baseline", default="baseline-qwen25-base")
    parser.add_argument("--candidate-run-id", default="candidate-kaggle-ddp-eval")
    parser.add_argument("--candidate-name", default="Qwen2.5-0.5B + Kaggle DDP evaluator")
    parser.add_argument("--num-examples", type=int, default=512)
    parser.add_argument("--profile", choices=["optimized", "balanced", "regressed"], default="optimized")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--matmul-size", type=int, default=384)
    return parser.parse_args()


def setup_distributed(requested_device: str) -> tuple[bool, int, int, int]:
    """Initialize torch.distributed when launched with torchrun.

    `torchrun --nproc_per_node=2 ...` sets WORLD_SIZE, RANK, and LOCAL_RANK.
    This function reads those variables and chooses NCCL for CUDA or Gloo for
    CPU. The rest of the code can then use the same reduction path locally,
    on Kaggle T4x2, or in CI-style CPU sanity checks.
    """

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if distributed:
        # NCCL is the standard backend for multi-GPU CUDA runs. For a local CPU
        # smoke test we force Gloo, because NCCL cannot reduce CPU tensors.
        use_cuda = requested_device != "cpu" and torch.cuda.is_available()
        backend = "nccl" if use_cuda else "gloo"
        dist.init_process_group(backend=backend)
        if use_cuda:
            torch.cuda.set_device(local_rank)

    return distributed, rank, local_rank, world_size


def select_device(requested: str, local_rank: int) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is not available.")
    if requested in {"auto", "cuda"} and torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def make_eval_set(num_examples: int) -> list[EvalExample]:
    """Create a deterministic evaluation set with varied difficulty.

    The examples mimic scientific-summary prompts. Difficulty drives the
    simulated metric distribution, which makes the aggregate scores feel like a
    real evaluation benchmark while keeping the exercise lightweight.
    """

    topics = [
        "battery degradation",
        "traffic prediction",
        "vegetation risk",
        "sensor fusion",
        "graph attention",
        "LLM regression testing",
        "energy forecasting",
        "autonomous mobility",
    ]
    examples: list[EvalExample] = []
    for idx in range(num_examples):
        topic = topics[idx % len(topics)]
        difficulty = 0.25 + 0.70 * ((idx * 37) % 101) / 100.0
        examples.append(
            EvalExample(
                prompt=f"Summarize the key experimental contribution for paper {idx} about {topic}.",
                reference_summary=f"The paper studies {topic} and reports reproducible evaluation metrics.",
                difficulty=difficulty,
            )
        )
    return examples


def stable_noise(text: str, scale: float) -> float:
    """Map text to deterministic pseudo-random noise.

    Python's built-in hash is salted per process, so distributed workers would
    disagree. SHA-256 gives stable noise across Kaggle workers and local runs.
    """

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return (value - 0.5) * scale


def profile_offsets(profile: str) -> dict[str, float]:
    if profile == "optimized":
        return {"quality": 0.356, "latency": 0.66, "tokens": 24.0}
    if profile == "balanced":
        return {"quality": 0.335, "latency": 0.74, "tokens": 28.0}
    return {"quality": 0.286, "latency": 0.91, "tokens": 43.0}


def simulate_model_eval(example: EvalExample, profile: str, device: torch.device, matmul_size: int) -> dict[str, float]:
    """Simulate one LLM evaluation item and do a small GPU operation.

    The tensor operation is intentionally simple: it proves each worker is using
    its assigned CUDA device without making the notebook depend on a large LLM.
    The reported metrics are deterministic functions of the example difficulty
    and the selected candidate profile.
    """

    if device.type == "cuda":
        # A tiny GPU workload makes nvidia-smi show activity during the run.
        # It is not used as a metric; it is only a lightweight stand-in for
        # model forward passes during LLM evaluation.
        a = torch.ones((matmul_size, matmul_size), device=device)
        b = torch.eye(matmul_size, device=device)
        _ = (a @ b).sum().item()

    offsets = profile_offsets(profile)
    prompt_key = f"{profile}:{example.prompt}"
    quality = offsets["quality"] - 0.035 * example.difficulty + stable_noise(prompt_key, 0.018)
    latency = offsets["latency"] + 0.10 * example.difficulty + stable_noise(prompt_key + ":latency", 0.035)
    tokens = offsets["tokens"] + 8.0 * example.difficulty + stable_noise(prompt_key + ":tokens", 2.0)

    return {
        "rougeL": max(0.0, min(1.0, quality)),
        "latency": max(0.01, latency),
        "generated_tokens": max(1.0, tokens),
    }


def reduce_sums(local_sums: torch.Tensor, distributed: bool) -> torch.Tensor:
    """Sum local worker statistics into rank 0-visible global totals."""

    if distributed:
        dist.all_reduce(local_sums, op=dist.ReduceOp.SUM)
    return local_sums


def build_run_record(
    args: argparse.Namespace,
    metrics: dict[str, float],
    rank: int,
    world_size: int,
    device: torch.device,
) -> RunRecord:
    return RunRecord(
        run_id=args.candidate_run_id,
        model_name=args.candidate_name,
        stage="staging",
        source_project="04 - LLMOps Experiment Tracker and Regression Gate",
        created_at=datetime.now(timezone.utc).isoformat(),
        notes=(
            "Distributed Kaggle-style evaluation run. Evaluation examples were sharded "
            "across workers, metrics were reduced with torch.distributed, and the final "
            "candidate was checked against the production baseline."
        ),
        stack=["PyTorch Distributed", "Kaggle T4x2", "LLMOps", "regression-gate"],
        metrics=metrics,
        tags={
            "profile": args.profile,
            "rank_saved_by": rank,
            "world_size": world_size,
            "device": str(device),
            "num_examples": args.num_examples,
        },
    )


def main() -> None:
    args = parse_args()
    distributed, rank, local_rank, world_size = setup_distributed(args.device)
    device = select_device(args.device, local_rank)

    random.seed(13 + rank)
    examples = make_eval_set(args.num_examples)

    # Strided sharding is enough for an evaluation workload: rank 0 gets
    # examples 0, world_size, 2*world_size..., rank 1 gets 1, 1+world_size...
    local_examples = examples[rank::world_size]
    local = torch.zeros(4, dtype=torch.float64, device=device)

    for example in local_examples:
        row = simulate_model_eval(example, args.profile, device, args.matmul_size)
        local[0] += row["rougeL"]
        local[1] += row["latency"]
        local[2] += row["generated_tokens"]
        local[3] += 1

    totals = reduce_sums(local, distributed)
    count = max(float(totals[3].item()), 1.0)
    metrics = {
        "rougeL_mean": float((totals[0] / count).item()),
        "latency_mean_seconds": float((totals[1] / count).item()),
        "generated_tokens_mean": float((totals[2] / count).item()),
        "prompt_tokens_mean": 118.0,
    }

    if rank == 0:
        run = build_run_record(args, metrics, rank, world_size, device)
        save_run(run)
        baseline = load_run(args.baseline)
        decision = evaluate_gate(baseline, run)
        save_gate_decision(decision)
        print(json.dumps({"run": run.to_dict(), "gate_decision": decision.to_dict()}, indent=2))

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
