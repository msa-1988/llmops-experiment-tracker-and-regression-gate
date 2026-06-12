#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

python - <<PY
import torch

required = int("${NPROC_PER_NODE}")
available = torch.cuda.device_count()
if available < required:
    raise SystemExit(
        f"This Kaggle practice run needs {required} CUDA GPUs, but only {available} were detected. "
        "Select Accelerator: GPU T4 x2, or set NPROC_PER_NODE=1 for a single-GPU dry run."
    )
print(f"CUDA GPUs detected: {available}")
PY

python scripts/bootstrap_demo_runs.py

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" scripts/distributed_llm_eval.py \
  --baseline baseline-qwen25-base \
  --candidate-run-id candidate-kaggle-ddp-eval \
  --candidate-name "Qwen2.5-0.5B + Kaggle DDP evaluation" \
  --num-examples 512 \
  --profile optimized \
  --device cuda

python scripts/export_run_report.py \
  --baseline baseline-qwen25-base \
  --candidate candidate-kaggle-ddp-eval \
  --output artifacts/kaggle_ddp_run_report.md
