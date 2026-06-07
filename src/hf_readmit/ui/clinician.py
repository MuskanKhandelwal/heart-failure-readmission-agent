"""Streamlit Clinician View.

Renders a discharge-planning assessment for a selected patient. This page talks
to the FastAPI ``/assess`` endpoint over HTTP (via ``requests``) and never
imports the agent package directly, so the UI stays decoupled from the model.
"""

from __future__ import annotations

import json
import os

import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.getenv("HF_API_URL", "http://localhost:8000")

# Sample request payloads (kept here so the UI does not import agent code).
# A ``medications`` list is included so drug-interaction warnings can render.
SAMPLE_PAYLOADS: dict[str, dict] = {
    "TEST001": {
        "age_at_admit": 88,
        "sex": 1,
        "length_of_stay": 18,
        "sp_chf": 1,
        "sp_ckd": 1,
        "sp_diabetes": 1,
        "sp_ischemic_hd": 1,
        "sp_copd": 1,
        "sp_depression": 1,
        "sp_alzheimer": 1,
        "sp_stroke": 1,
        "prior_ip_admits_6mo": 6,
        "prior_ip_admits_12mo": 12,
        "inpatient_reimbursement": 50000.0,
        "drg_code": "291",
        "hf_primary": 1,
        "medications": ["lisinopril", "spironolactone", "metoprolol", "furosemide", "ibuprofen"],
    },
    "TEST002": {
        "age_at_admit": 58,
        "sex": 0,
        "length_of_stay": 2,
        "sp_chf": 1,
        "sp_ckd": 0,
        "sp_diabetes": 0,
        "sp_ischemic_hd": 0,
        "sp_copd": 0,
        "sp_depression": 0,
        "sp_alzheimer": 0,
        "sp_stroke": 0,
        "prior_ip_admits_6mo": 0,
        "prior_ip_admits_12mo": 0,
        "inpatient_reimbursement": 2500.0,
        "drg_code": "293",
        "hf_primary": 1,
        "medications": ["metoprolol", "furosemide"],
    },
}

_CATEGORY_COLOR = {"low": "#2e7d32", "medium": "#f9a825", "high": "#c62828"}


def _risk_gauge(score: float) -> go.Figure:
    """Plotly indicator gauge on a 0-1 scale with green/yellow/red bands."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"valueformat": ".3f"},
            title={"text": "30-day readmission risk"},
            gauge={
                "axis": {"range": [0, 1]},
                "bar": {"color": "#37474f"},
                "steps": [
                    {"range": [0, 0.2], "color": "#a5d6a7"},
                    {"range": [0.2, 0.5], "color": "#fff59d"},
                    {"range": [0.5, 1.0], "color": "#ef9a9a"},
                ],
            },
        )
    )
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def _shap_chart(drivers: list[dict]) -> go.Figure:
    """Horizontal SHAP bar chart: red increases risk, blue decreases risk."""
    drivers = list(reversed(drivers))  # largest at top
    features = [d.get("feature", "") for d in drivers]
    values = [float(d.get("shap_value", 0.0)) for d in drivers]
    colors = ["#c62828" if v > 0 else "#1565c0" for v in values]
    fig = go.Figure(go.Bar(x=values, y=features, orientation="h", marker_color=colors))
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=30, b=10),
        title="Top SHAP drivers (red = increases risk, blue = decreases risk)",
        xaxis_title="SHAP value",
    )
    return fig


def _call_assess(patient_id: str, patient_input: dict) -> dict | None:
    """POST to the API /assess endpoint; surface errors in the UI."""
    try:
        resp = requests.post(
            f"{API_URL}/assess",
            json={"patient_id": patient_id, "patient_input": patient_input},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        st.error(f"Failed to reach the assessment API at {API_URL}: {exc}")
        return None


def render() -> None:
    """Render the Clinician View page."""
    st.title("Heart Failure Discharge Planning Agent")

    with st.sidebar:
        st.header("Patient")
        choice = st.selectbox("Select patient", ["TEST001", "TEST002", "Custom JSON"])
        if choice == "Custom JSON":
            default = json.dumps({"patient_id": "CUSTOM01", "patient_input": SAMPLE_PAYLOADS["TEST001"]}, indent=2)
            raw = st.text_area("Patient request JSON", value=default, height=320)
        run = st.button("Assess patient", type="primary")

    if not run:
        st.info("Select a patient in the sidebar and click **Assess patient**.")
        return

    if choice == "Custom JSON":
        try:
            payload = json.loads(raw)
            patient_id = payload["patient_id"]
            patient_input = payload["patient_input"]
        except (json.JSONDecodeError, KeyError) as exc:
            st.error(f"Invalid custom JSON (need patient_id + patient_input): {exc}")
            return
    else:
        patient_id = choice
        patient_input = SAMPLE_PAYLOADS[choice]

    with st.spinner("Running discharge-planning agent..."):
        result = _call_assess(patient_id, patient_input)
    if result is None:
        return

    summary = result.get("discharge_summary") or {}

    # Row 1: gauge + category badge + patient summary
    c1, c2 = st.columns([1, 2])
    with c1:
        st.plotly_chart(_risk_gauge(result.get("risk_score", 0.0)), use_container_width=True)
        category = result.get("risk_category", "unknown")
        color = _CATEGORY_COLOR.get(category, "#616161")
        st.markdown(
            f"<div style='text-align:center'><span style='background:{color};color:white;"
            f"padding:6px 16px;border-radius:14px;font-weight:700;text-transform:uppercase'>"
            f"{category} risk</span></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.subheader("Patient summary")
        st.write(summary.get("patient_summary", "—"))
        st.caption(f"Nodes visited: {' → '.join(result.get('nodes_visited', []))}  "
                   f"| {result.get('processing_time_seconds', 0):.2f}s")

    # Row 2: SHAP chart
    drivers = result.get("top_drivers") or []
    if drivers:
        st.plotly_chart(_shap_chart(drivers), use_container_width=True)

    # Row 3: intervention cards
    st.subheader("Proposed interventions")
    citations = {c.get("chunk_id"): c for c in summary.get("citations", [])}
    interventions = result.get("interventions") or []
    if not interventions:
        st.write("No grounded interventions were proposed.")
    for iv in interventions:
        with st.expander(f"[{iv.get('evidence_level', '—')}] {iv.get('description', 'Intervention')}"):
            st.markdown(f"**Rationale:** {iv.get('rationale', '—')}")
            st.markdown(f"**Evidence level:** {iv.get('evidence_level', '—')}")
            cite = citations.get(iv.get("citation_chunk_id"))
            if cite:
                st.markdown(
                    f"**Citation:** {cite.get('source', '—')} — {cite.get('section', '—')} "
                    f"(`{iv.get('citation_chunk_id')}`)"
                )
            else:
                st.markdown(f"**Citation:** `{iv.get('citation_chunk_id', '—')}`")

    # Row 4: drug-interaction warnings
    interaction_flags = [f for f in (result.get("flags") or []) if f.startswith("drug_interaction:")]
    if interaction_flags:
        st.subheader("⚠️ Drug interaction warnings")
        for flag in interaction_flags:
            parts = flag.split(":", 2)
            severity = parts[1] if len(parts) > 1 else "unknown"
            detail = parts[2] if len(parts) > 2 else ""
            st.warning(f"**{severity.upper()}** — {detail.replace('+', ' + ')}")

    # Row 5: full discharge summary
    with st.expander("Full discharge summary (JSON)"):
        st.json(summary)

    st.divider()
    st.caption("Powered by GPT-4o + AHA/ACC 2022 HF Guidelines")
