from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

from app.src.gates import evaluate_gate
from app.src.run_registry import load_all_runs, load_run, runs_to_frame


st.set_page_config(
    page_title="LLMOps Experiment Tracker",
    page_icon="🧪",
    layout="wide",
)


def metric_card(label: str, value: str, help_text: str) -> None:
    st.markdown(
        f"""
        <div style="padding:1rem;border:1px solid #d8eae7;border-radius:16px;background:#f5f8f7;height:100%;">
          <div style="font-size:0.85rem;color:#5b6877;">{label}</div>
          <div style="font-size:1.7rem;font-weight:700;color:#16324F;margin-top:0.2rem;">{value}</div>
          <div style="font-size:0.8rem;color:#5b6877;margin-top:0.25rem;">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.title("LLMOps Experiment Tracker and Regression Gate")
st.caption("Evaluation-first MLOps workflow for LLM fine-tuning promotion decisions.")

runs = load_all_runs()
frame = runs_to_frame(runs)

if frame.empty:
    st.warning("No runs found. Bootstrap demo data with `python scripts/bootstrap_demo_runs.py`.")
    st.stop()

production_count = int((frame["stage"] == "production").sum())
staging_count = int((frame["stage"] == "staging").sum())
top_rouge = frame["rougeL_mean"].max()

col1, col2, col3 = st.columns(3)
with col1:
    metric_card("Registered runs", str(len(frame)), "Baseline and candidate experiments tracked locally.")
with col2:
    metric_card("Staging candidates", str(staging_count), "Runs eligible for promotion checks.")
with col3:
    metric_card("Best ROUGE-L", f"{top_rouge:.4f}", "Current top quality score across all tracked runs.")

st.markdown("### Run Leaderboard")
display_frame = frame[
    [
        "run_id",
        "model_name",
        "stage",
        "rougeL_mean",
        "latency_mean_seconds",
        "generated_tokens_mean",
        "source_project",
    ]
].copy()
st.dataframe(display_frame, width="stretch", hide_index=True)

st.markdown("### Promotion Gate Review")
production_runs = frame[frame["stage"] == "production"]["run_id"].tolist()
candidate_runs = frame[frame["stage"] == "staging"]["run_id"].tolist()

left, right = st.columns([1, 1])
with left:
    baseline_id = st.selectbox("Baseline run", production_runs, index=0)
with right:
    candidate_id = st.selectbox("Candidate run", candidate_runs, index=0)

baseline = load_run(baseline_id)
candidate = load_run(candidate_id)
decision = evaluate_gate(baseline, candidate)

status_color = "#0F766E" if decision.status == "pass" else "#B42318"
st.markdown(
    f"""
    <div style="padding:1rem;border-radius:18px;background:#ffffff;border:1px solid #d8eae7;">
      <div style="font-size:0.85rem;color:#5b6877;">Promotion decision</div>
      <div style="font-size:2rem;font-weight:800;color:{status_color};margin-top:0.2rem;">{decision.status.upper()}</div>
      <div style="font-size:0.95rem;color:#16324F;margin-top:0.35rem;">{decision.summary}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

checks = pd.DataFrame(
    [
        {
            "check": name,
            "passed": values["passed"],
            "details": ", ".join(f"{k}={v}" for k, v in values.items() if k != "passed"),
        }
        for name, values in decision.checks.items()
    ]
)

tabs = st.tabs(["Gate Checks", "Metric Comparison", "Run Notes"])

with tabs[0]:
    st.dataframe(checks, width="stretch", hide_index=True)

with tabs[1]:
    comparison = pd.DataFrame(
        [
            {"metric": "ROUGE-L", "baseline": baseline.metrics["rougeL_mean"], "candidate": candidate.metrics["rougeL_mean"]},
            {
                "metric": "Mean latency (s)",
                "baseline": baseline.metrics["latency_mean_seconds"],
                "candidate": candidate.metrics["latency_mean_seconds"],
            },
            {
                "metric": "Generated tokens",
                "baseline": baseline.metrics["generated_tokens_mean"],
                "candidate": candidate.metrics["generated_tokens_mean"],
            },
        ]
    )
    st.dataframe(comparison, width="stretch", hide_index=True)

with tabs[2]:
    st.markdown(f"**Baseline notes**: {baseline.notes}")
    st.markdown(f"**Candidate notes**: {candidate.notes}")
    st.markdown(f"**Candidate stack**: {', '.join(candidate.stack)}")
