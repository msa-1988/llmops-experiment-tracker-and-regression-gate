from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import RUNS_DIR


@dataclass
class RunRecord:
    run_id: str
    model_name: str
    stage: str
    source_project: str
    created_at: str
    notes: str
    stack: list[str]
    metrics: dict[str, float]
    tags: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RunRecord":
        return cls(
            run_id=payload["run_id"],
            model_name=payload["model_name"],
            stage=payload["stage"],
            source_project=payload["source_project"],
            created_at=payload["created_at"],
            notes=payload["notes"],
            stack=payload["stack"],
            metrics=payload["metrics"],
            tags=payload["tags"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "model_name": self.model_name,
            "stage": self.stage,
            "source_project": self.source_project,
            "created_at": self.created_at,
            "notes": self.notes,
            "stack": self.stack,
            "metrics": self.metrics,
            "tags": self.tags,
        }


def ensure_run_dir() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)


def run_path(run_id: str) -> Path:
    return RUNS_DIR / f"{run_id}.json"


def save_run(run: RunRecord) -> Path:
    ensure_run_dir()
    path = run_path(run.run_id)
    path.write_text(json.dumps(run.to_dict(), indent=2), encoding="utf-8")
    return path


def load_run(run_id: str) -> RunRecord:
    payload = json.loads(run_path(run_id).read_text(encoding="utf-8"))
    return RunRecord.from_dict(payload)


def load_all_runs() -> list[RunRecord]:
    ensure_run_dir()
    records: list[RunRecord] = []
    for path in sorted(RUNS_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        records.append(RunRecord.from_dict(payload))
    return records


def runs_to_frame(runs: list[RunRecord]) -> pd.DataFrame:
    rows = []
    for run in runs:
        row = {
            "run_id": run.run_id,
            "model_name": run.model_name,
            "stage": run.stage,
            "source_project": run.source_project,
            "created_at": run.created_at,
            "notes": run.notes,
            "stack": ", ".join(run.stack),
        }
        row.update(run.metrics)
        row.update({f"tag_{k}": v for k, v in run.tags.items()})
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(by=["stage", "rougeL_mean"], ascending=[True, False])
