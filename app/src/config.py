from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
RUNS_DIR = ARTIFACTS_DIR / "runs"
GATE_DECISIONS_DIR = ARTIFACTS_DIR / "gate_decisions"
SCREENSHOTS_DIR = ROOT_DIR / "screenshots"
CONFIG_DIR = ROOT_DIR / "config"

GATE_CONFIG_PATH = CONFIG_DIR / "regression_gate.yaml"
PROJECT_METADATA_PATH = CONFIG_DIR / "project_metadata.yaml"

