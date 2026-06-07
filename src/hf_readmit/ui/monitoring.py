"""Streamlit Monitoring View.

Shows live API metrics, the frozen model/RAG eval numbers from Units 3-4, and
the latest adversarial eval report (if present). Talks to the API over HTTP.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("HF_API_URL", "http://localhost:8000")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
LATEST_EVAL_PATH = PROJECT_ROOT / "evals" / "results" / "latest.json"

# Frozen results from Units 3 (model) and 4 (RAG). Keep in sync with README.
MODEL_METRICS = [
    {"metric": "AUROC", "value": 0.563},
    {"metric": "AUPRC", "value": 0.131},
    {"metric": "Brier score", "value": 0.094},
    {"metric": "Positive label rate", "value": 0.106},
]
RAG_METRICS = [
    {"metric": "Recall@5", "value": 0.90},
    {"metric": "Precision@5", "value": 0.76},
    {"metric": "MRR", "value": 0.942},
]


def _fetch_metrics() -> dict | None:
    """GET the API /metrics endpoint."""
    try:
        resp = requests.get(f"{API_URL}/metrics", timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        st.error(f"Failed to reach the metrics API at {API_URL}: {exc}")
        return None


def render() -> None:
    """Render the Monitoring page."""
    st.title("System Monitoring")

    live = _fetch_metrics() or {}

    # Row 1: live metric cards
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Requests", live.get("total_requests", 0))
    c2.metric("Avg Processing Time", f"{live.get('avg_processing_time_seconds', 0.0):.2f}s")
    c3.metric("Active Flags", len(live.get("flag_counts", {})))

    # Row 2: model metrics
    st.subheader("Model metrics (Unit 3 — SynPUF)")
    st.table(pd.DataFrame(MODEL_METRICS))

    # Row 3: RAG eval metrics
    st.subheader("RAG retrieval metrics (Unit 4)")
    st.table(pd.DataFrame(RAG_METRICS))

    # Row 4: adversarial eval results
    st.subheader("Adversarial eval results")
    if LATEST_EVAL_PATH.exists():
        report = json.loads(LATEST_EVAL_PATH.read_text())
        m1, m2, m3 = st.columns(3)
        m1.metric("Total scenarios", report.get("total_scenarios", 0))
        m2.metric("Passed", report.get("passed", 0))
        m3.metric("Failed", report.get("failed", 0))
        st.caption(f"Run at {report.get('timestamp', 'unknown')}")
        results = report.get("results", [])
        if results:
            st.dataframe(pd.DataFrame(results), use_container_width=True)
    else:
        st.info("Run eval harness to populate")

    # Row 5: recent agent runs
    st.subheader("Recent agent runs")
    recent = live.get("recent_runs", [])
    if recent:
        st.dataframe(pd.DataFrame(recent), use_container_width=True)
    else:
        st.write("No agent runs recorded yet this session.")
