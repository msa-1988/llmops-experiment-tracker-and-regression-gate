from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from app.src.config import SCREENSHOTS_DIR
from app.src.gates import evaluate_gate
from app.src.run_registry import load_all_runs, load_run, runs_to_frame


plt.rcParams["figure.facecolor"] = "#F5F8F7"
plt.rcParams["axes.facecolor"] = "#FFFFFF"
plt.rcParams["savefig.facecolor"] = "#F5F8F7"


def save_leaderboard(frame: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    sorted_frame = frame.sort_values("rougeL_mean", ascending=True)
    colors = ["#0F766E" if stage == "staging" else "#16324F" for stage in sorted_frame["stage"]]
    ax.barh(sorted_frame["run_id"], sorted_frame["rougeL_mean"], color=colors)
    ax.set_title("Tracked runs by ROUGE-L")
    ax.set_xlabel("ROUGE-L")
    ax.grid(axis="x", alpha=0.15)
    fig.tight_layout()
    fig.savefig(SCREENSHOTS_DIR / "run-leaderboard.png", dpi=180)
    plt.close(fig)


def save_frontier(frame: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    color_map = {"production": "#16324F", "staging": "#0F766E"}
    for _, row in frame.iterrows():
        ax.scatter(
            row["latency_mean_seconds"],
            row["rougeL_mean"],
            s=120,
            color=color_map.get(row["stage"], "#5B6877"),
        )
        ax.text(row["latency_mean_seconds"] + 0.002, row["rougeL_mean"] + 0.001, row["run_id"], fontsize=8)
    ax.set_title("Quality vs latency frontier")
    ax.set_xlabel("Mean latency (seconds)")
    ax.set_ylabel("ROUGE-L")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(SCREENSHOTS_DIR / "quality-latency-frontier.png", dpi=180)
    plt.close(fig)


def save_promotion_pipeline() -> None:
    baseline = load_run("baseline-qwen25-base")
    candidate = load_run("candidate-qwen25-qlora-v1")
    decision = evaluate_gate(baseline, candidate)

    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.axis("off")
    boxes = [
        (0.05, "Register\nrun metrics", "#EAF2F0"),
        (0.32, "Compare against\nproduction baseline", "#EAF2F0"),
        (0.60, "Apply regression\nthresholds", "#EAF2F0"),
        (0.82, f"Promotion\n{decision.status.upper()}", "#D1FADF" if decision.status == "pass" else "#FEE4E2"),
    ]
    for x, label, color in boxes:
        rect = plt.Rectangle((x, 0.30), 0.16, 0.36, facecolor=color, edgecolor="#16324F", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + 0.08, 0.48, label, ha="center", va="center", fontsize=11, color="#16324F", weight="bold")
    for x0, x1 in [(0.21, 0.32), (0.48, 0.60), (0.76, 0.82)]:
        ax.annotate("", xy=(x1, 0.48), xytext=(x0, 0.48), arrowprops=dict(arrowstyle="->", color="#0F766E", lw=2))
    ax.text(0.60, 0.12, "Default gate: quality drop <= 0.005, latency increase <= 0.20s, token increase <= 5",
            ha="center", va="center", fontsize=10, color="#5B6877")
    fig.tight_layout()
    fig.savefig(SCREENSHOTS_DIR / "promotion-pipeline.png", dpi=180)
    plt.close(fig)


def main() -> None:
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    frame = runs_to_frame(load_all_runs())
    save_leaderboard(frame)
    save_frontier(frame)
    save_promotion_pipeline()
    print("generated screenshots")


if __name__ == "__main__":
    main()
