from __future__ import annotations

from pathlib import Path

from .gates import GateDecision
from .run_registry import RunRecord


def render_markdown_report(baseline: RunRecord, candidate: RunRecord, decision: GateDecision) -> str:
    return f"""# LLMOps Promotion Report

## Summary

- Baseline run: `{baseline.run_id}`
- Candidate run: `{candidate.run_id}`
- Decision: `{decision.status.upper()}`

{decision.summary}

## Baseline Metrics

- ROUGE-L: `{baseline.metrics['rougeL_mean']:.4f}`
- Mean latency (s): `{baseline.metrics['latency_mean_seconds']:.4f}`
- Mean generated tokens: `{baseline.metrics['generated_tokens_mean']:.2f}`

## Candidate Metrics

- ROUGE-L: `{candidate.metrics['rougeL_mean']:.4f}`
- Mean latency (s): `{candidate.metrics['latency_mean_seconds']:.4f}`
- Mean generated tokens: `{candidate.metrics['generated_tokens_mean']:.2f}`

## Gate Checks

- Stage check: `{decision.checks['stage']['passed']}`
- Quality check: `{decision.checks['quality']['passed']}`
- Latency check: `{decision.checks['latency']['passed']}`
- Token budget check: `{decision.checks['tokens']['passed']}`

## Interpretation

This report is designed for a deployment review workflow where LLM candidates must prove they improve or at least preserve quality without violating latency or output-budget thresholds. It gives a compact, reproducible promotion decision that can be attached to CI or release approval steps.
"""


def save_markdown_report(text: str, destination: Path) -> Path:
    destination.write_text(text, encoding="utf-8")
    return destination
