"""Tests for the FastAPI discharge-planning service (Unit 6).

``run_agent`` and the heavy resource loaders are mocked so no real agent runs,
model loads, or network calls occur.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from hf_readmit.api.app import AssessResponse, app

# A representative final AgentState as returned by run_agent (full pipeline).
FAKE_STATE = {
    "patient_id": "TEST001",
    "risk_score": 0.42,
    "risk_category": "medium",
    "top_drivers": [
        {"feature": "prior_ip_admits_6mo", "shap_value": 1.14, "direction": "increases_risk"}
    ],
    "conditions": ["heart failure", "chronic kidney disease"],
    "retrieved_chunks": [
        {"chunk_id": "aha_hf_guideline_2022_p1_0", "text": "...", "source": "aha.pdf", "section": "page 1"}
    ],
    "proposed_interventions": [
        {
            "intervention_id": "hf-followup-7day",
            "description": "Schedule 7-day follow-up",
            "rationale": "Reduces readmission risk",
            "citation_chunk_id": "aha_hf_guideline_2022_p1_0",
            "evidence_level": "Class I",
        }
    ],
    "grounding_failures": None,
    "discharge_summary": {
        "patient_summary": "Test patient",
        "citations": [{"chunk_id": "aha_hf_guideline_2022_p1_0", "source": "aha.pdf", "section": "page 1"}],
    },
    "flags": ["drug_interaction:high:ace_inhibitor+k_sparing_diuretic"],
    "retry_count": 0,
    "error": None,
}

TEST001_PAYLOAD = {
    "patient_id": "TEST001",
    "patient_input": {
        "age_at_admit": 72,
        "sex": 1,
        "length_of_stay": 5,
        "sp_chf": 1,
        "sp_ckd": 1,
        "sp_diabetes": 0,
        "sp_ischemic_hd": 1,
        "sp_copd": 0,
        "sp_depression": 0,
        "sp_alzheimer": 0,
        "sp_stroke": 0,
        "prior_ip_admits_6mo": 2,
        "prior_ip_admits_12mo": 3,
        "inpatient_reimbursement": 12000.0,
        "drg_code": "291",
        "hf_primary": 1,
    },
}


@pytest.fixture(autouse=True)
def mock_agent_and_resources():
    """Mock the agent and resource loaders so tests are hermetic."""
    with patch("hf_readmit.api.app.run_agent", return_value=FAKE_STATE) as mock_run, patch(
        "hf_readmit.agent.tools._get_predictor", return_value=MagicMock()
    ), patch("hf_readmit.agent.tools._get_retriever", return_value=MagicMock()):
        yield mock_run


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "model_loaded" in body
    assert body["status"] == "ok"


def test_assess_endpoint(client):
    resp = client.post("/assess", json=TEST001_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    for field in (
        "patient_id",
        "risk_score",
        "risk_category",
        "top_drivers",
        "interventions",
        "discharge_summary",
        "flags",
        "nodes_visited",
        "processing_time_seconds",
    ):
        assert field in body
    assert body["patient_id"] == "TEST001"
    assert body["interventions"][0]["intervention_id"] == "hf-followup-7day"
    assert body["nodes_visited"][0] == "assess_risk"
    assert "format_discharge_summary" in body["nodes_visited"]


def test_assess_response_schema(client):
    resp = client.post("/assess", json=TEST001_PAYLOAD)
    assert resp.status_code == 200
    # Round-trips through the response model without validation error.
    model = AssessResponse(**resp.json())
    assert isinstance(model.risk_score, float)
    assert model.risk_category == "medium"


def test_metrics_endpoint(client):
    # Make one assessment so metrics are populated.
    client.post("/assess", json=TEST001_PAYLOAD)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_requests"] >= 1
    assert "avg_processing_time_seconds" in body
    assert "flag_counts" in body
    assert "recent_runs" in body
