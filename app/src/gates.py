from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .config import GATE_CONFIG_PATH, GATE_DECISIONS_DIR
from .run_registry import RunRecord


@dataclass
class GateDecision:
    baseline_run_id: str
    candidate_run_id: str
    status: str
    created_at: str
    checks: dict[str, dict[str, Any]]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_run_id": self.baseline_run_id,
            "candidate_run_id": self.candidate_run_id,
            "status": self.status,
            "created_at": self.created_at,
            "checks": self.checks,
            "summary": self.summary,
        }


def load_gate_config() -> dict[str, Any]:
    return yaml.safe_load(GATE_CONFIG_PATH.read_text(encoding="utf-8"))


def evaluate_gate(baseline: RunRecord, candidate: RunRecord) -> GateDecision:
    config = load_gate_config()
    thresholds = config["thresholds"]

    quality_delta = candidate.metrics["rougeL_mean"] - baseline.metrics["rougeL_mean"]
    latency_delta = candidate.metrics["latency_mean_seconds"] - baseline.metrics["latency_mean_seconds"]
    token_delta = candidate.metrics["generated_tokens_mean"] - baseline.metrics["generated_tokens_mean"]

    checks = {
        "stage": {
            "passed": candidate.stage == config["required_stage_for_promotion"],
            "observed": candidate.stage,
            "expected": config["required_stage_for_promotion"],
        },
        "quality": {
            "passed": quality_delta >= -thresholds["max_quality_drop"],
            "observed_delta": round(quality_delta, 4),
            "minimum_allowed_delta": -thresholds["max_quality_drop"],
        },
        "latency": {
            "passed": latency_delta <= thresholds["max_latency_increase_seconds"],
            "observed_delta_seconds": round(latency_delta, 4),
            "maximum_allowed_delta_seconds": thresholds["max_latency_increase_seconds"],
        },
        "tokens": {
            "passed": token_delta <= thresholds["max_generated_token_increase"],
            "observed_delta": round(token_delta, 4),
            "maximum_allowed_delta": thresholds["max_generated_token_increase"],
        },
    }

    status = "pass" if all(item["passed"] for item in checks.values()) else "fail"
    summary = (
        f"Candidate {candidate.run_id} {'passes' if status == 'pass' else 'fails'} the promotion gate "
        f"against baseline {baseline.run_id}."
    )
    return GateDecision(
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        status=status,
        created_at=datetime.now(timezone.utc).isoformat(),
        checks=checks,
        summary=summary,
    )


def save_gate_decision(decision: GateDecision) -> Path:
    GATE_DECISIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = GATE_DECISIONS_DIR / f"{decision.candidate_run_id}_vs_{decision.baseline_run_id}.json"
    path.write_text(json.dumps(decision.to_dict(), indent=2), encoding="utf-8")
    return path
