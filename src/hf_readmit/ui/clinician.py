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
    """Plotly indicator gauge on a 0-1 scale with green/yellow/red bands.

    Bands aligned to population-relative tertiles:
    - Low (green): < 0.092
    - Medium (yellow): 0.092-0.148
    - High (red): > 0.148
    """
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
                    {"range": [0, 0.092], "color": "#a5d6a7"},
                    {"range": [0.092, 0.148], "color": "#fff59d"},
                    {"range": [0.148, 1.0], "color": "#ef9a9a"},
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

    # Patient header bar
    patient_age = patient_input.get('age_at_admit', '—')
    patient_sex = 'Male' if patient_input.get('sex') == 1 else 'Female'
    patient_los = patient_input.get('length_of_stay', '—')
    patient_drg = patient_input.get('drg_code', '—')
    hf_primary = 'Yes' if patient_input.get('hf_primary') else 'No'

    st.markdown(f"""
<div style='background:#1e3a5f;color:white;padding:12px 20px;
border-radius:8px;margin-bottom:16px;display:flex;
justify-content:space-between;align-items:center'>
    <span style='font-size:16px;font-weight:700'>{patient_id}</span>
    <span><b>Age:</b> {patient_age}</span>
    <span><b>Sex:</b> {patient_sex}</span>
    <span><b>Length of Stay:</b> {patient_los} days</span>
    <span><b>Diagnosis Code:</b> {patient_drg}</span>
    <span><b>Heart Failure Primary Diagnosis:</b> {hf_primary}</span>
</div>
""", unsafe_allow_html=True)

    # Summary metrics row
    m1, m2, m3 = st.columns(3)
    risk_score_val = result.get("risk_score", 0.0)
    interventions_count = len(result.get("interventions") or [])
    drug_warning_count = len([
        f for f in (result.get("flags") or [])
        if f.startswith("drug_interaction:")
    ])
    m1.metric(
        label="30-day Readmission Risk",
        value=f"{risk_score_val:.1%}",
        delta=f"{(risk_score_val - 0.106):+.1%} vs 10.6% cohort avg",
        delta_color="inverse"
    )
    m2.metric(
        label="Grounded Interventions",
        value=interventions_count,
        delta="citations verified" if interventions_count > 0 else "none proposed"
    )
    m3.metric(
        label="Drug Warnings",
        value=drug_warning_count,
        delta="interactions flagged" if drug_warning_count > 0 else "none detected",
        delta_color="off" if drug_warning_count == 0 else "inverse"
    )
    st.divider()

    # Row 1: gauge + category badge + patient summary
    c1, c2 = st.columns([1, 2])
    with c1:
        risk_score = result.get("risk_score", 0.0)
        st.plotly_chart(_risk_gauge(risk_score), use_container_width=True)
        risk_category = result.get("risk_category", "unknown")
        color = _CATEGORY_COLOR.get(risk_category, "#616161")
        st.markdown(
            f"<div style='text-align:center'><span style='background:{color};color:white;"
            f"padding:6px 16px;border-radius:14px;font-weight:700;text-transform:uppercase'>"
            f"{risk_category} risk</span></div>",
            unsafe_allow_html=True,
        )
        if risk_category in ["medium", "high"]:
            st.warning(f"⚠️ Patient in {risk_category} risk tertile — recommend full discharge planning review")
    with c2:
        st.subheader("Patient summary")
        st.write(summary.get("patient_summary", "—"))
        # Agent execution timeline
        nodes_visited = result.get('nodes_visited', [])
        if nodes_visited:
            node_icons = {
                'assess_risk': '🎯',
                'retrieve_guidelines': '📚',
                'propose_plan': '💡',
                'safety_check': '🛡️',
                'format_discharge_summary': '📄'
            }
            timeline_parts = []
            for node in nodes_visited:
                icon = node_icons.get(node, '⚙️')
                label = node.replace('_', ' ').title()
                timeline_parts.append(f"{icon} {label}")
            st.markdown(
                " **→** ".join(timeline_parts),
                help="Agent execution path for this patient"
            )
        st.caption(f"Processing time: {result.get('processing_time_seconds', 0):.2f}s")

    # Row 2: SHAP chart
    drivers = result.get("top_drivers") or []
    if drivers:
        st.plotly_chart(_shap_chart(drivers), use_container_width=True)

    # Row 3: intervention cards with visible citations
    st.subheader(f"📋 Proposed Interventions ({len(result.get('interventions') or [])} grounded)")
    citations = {c.get("chunk_id"): c for c in summary.get("citations", [])}
    interventions = result.get("interventions") or []

    if not interventions:
        st.info("No grounded interventions were proposed for this patient.")
    else:
        for iv in interventions:
            evidence = iv.get('evidence_level', '')
            border_color = "#2e7d32" if "Class I" in evidence else (
                "#f9a825" if "Class II" in evidence else "#616161"
            )
            cite = citations.get(iv.get("citation_chunk_id"))
            if cite:
                cite_text = (
                    f"📎 {cite.get('source','').replace('_',' ').replace('-',' ').title()}"
                    f" — {cite.get('section','')}"
                )
            else:
                cite_text = f"📎 {iv.get('citation_chunk_id', 'No citation')}"

            st.markdown(f"""
        <div style='border-left:4px solid {border_color};padding:12px 16px;
        margin-bottom:10px;background:#f8f9fa;border-radius:0 6px 6px 0'>
            <div style='display:flex;align-items:center;gap:10px;margin-bottom:4px'>
                <b style='font-size:15px'>{iv.get('description','Intervention')}</b>
                <span style='background:{border_color};color:white;padding:2px 10px;
                border-radius:10px;font-size:11px;white-space:nowrap'>{evidence}</span>
            </div>
            <div style='color:#555;font-size:13px;margin-bottom:6px'>
                {iv.get('rationale','—')}
            </div>
            <div style='color:#1565c0;font-size:12px'>{cite_text}</div>
        </div>
            """, unsafe_allow_html=True)

    # Row 4: drug-interaction warnings
    interaction_flags = [f for f in (result.get("flags") or []) if f.startswith("drug_interaction:")]
    if interaction_flags:
        st.subheader("⚠️ Potential Drug Interactions")
        st.caption("Medication combinations detected that may require review or adjustment")
        for flag in interaction_flags:
            parts = flag.split(":", 2)
            severity = parts[1] if len(parts) > 1 else "unknown"
            detail = parts[2] if len(parts) > 2 else ""
            st.warning(f"**{severity.upper()}** — {detail.replace('+', ' + ')}")

    # Row 5: full discharge summary
    with st.expander("Full discharge summary (JSON)"):
        st.caption("**Flags:** Safety/clinical alerts triggered during assessment (e.g., drug_interaction:high:ACE Inhibitor + ARB, missing_data:renal_function)")
        st.json(summary)

    st.divider()
    st.caption("Powered by GPT-4o + AHA/ACC 2022 HF Guidelines")
