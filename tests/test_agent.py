"""Tests for the LangGraph discharge-planning agent (Unit 5).

All LLM, predictor, and retriever access is mocked (see the autouse fixtures
below) so the suite makes no real OpenAI API calls and loads no model/index
files. Tests are fully hermetic and fast.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from hf_readmit.agent import graph as graph_module
from hf_readmit.agent.graph import build_graph
from hf_readmit.agent.state import AgentState
from hf_readmit.agent.tools import (
    check_drug_interactions,
    get_patient_conditions,
    get_patient_risk_score,
    search_guidelines,
)

# Deterministic fake LLM payloads. The nodes parse different shapes per task:
# propose_plan -> JSON array; format_summary -> JSON object; assess_risk -> text.
_FAKE_INTERVENTIONS = [
    {
        "intervention_id": "hf-followup-7day",
        "description": "Schedule 7-day follow-up",
        "rationale": "Reduces readmission risk",
        "citation_chunk_id": "aha_hf_guideline_2022_p1_0",
        "evidence_level": "Class I",
    }
]
_FAKE_SUMMARY = {
    "patient_summary": "Test patient",
    "risk_assessment": "medium",
    "medications": [],
    "follow_up_plan": ["7-day follow-up"],
    "red_flag_symptoms": ["weight gain"],
    "patient_education": ["Monitor weight daily"],
    "citations": [],
}


def _fake_llm_call(prompt, system, max_tokens=None, task=None):
    """Return a task-appropriate canned response (no network)."""
    if task == "propose_plan":
        return json.dumps(_FAKE_INTERVENTIONS)
    if task == "format_summary":
        return json.dumps(_FAKE_SUMMARY)
    return "Mock clinical interpretation of the patient's readmission risk."


@pytest.fixture(autouse=True)
def mock_llm_client():
    """Patch LLMClient.call so no real OpenAI requests are made in tests."""
    with patch(
        "hf_readmit.llm.client.LLMClient.call", side_effect=_fake_llm_call
    ) as mock_call:
        yield mock_call


@pytest.fixture(autouse=True)
def mock_heavy_resources():
    """Prevent loading the risk model pickle and BM25/Chroma index in tests."""
    with patch("hf_readmit.agent.tools._get_predictor", return_value=MagicMock()), patch(
        "hf_readmit.agent.tools._get_retriever", return_value=MagicMock()
    ):
        yield


def test_agent_state_schema() -> None:
    """AgentState declares every required field with the right annotation set."""
    expected = {
        "patient_id",
        "patient_input",
        "risk_score",
        "risk_category",
        "top_drivers",
        "conditions",
        "retrieved_chunks",
        "proposed_interventions",
        "grounding_failures",
        "discharge_summary",
        "flags",
        "retry_count",
        "error",
    }
    assert set(AgentState.__annotations__) == expected


def test_tools_importable() -> None:
    """All four tools import and expose the LangChain tool interface."""
    for tool in (
        get_patient_risk_score,
        get_patient_conditions,
        search_guidelines,
        check_drug_interactions,
    ):
        assert hasattr(tool, "invoke")
        assert hasattr(tool, "name")


def test_drug_interaction_detection() -> None:
    """ACE inhibitor + ARB is flagged as a hyperkalemia interaction."""
    flagged = check_drug_interactions.invoke({"medications": ["lisinopril", "losartan"]})
    assert flagged, "expected at least one interaction"
    descriptions = " ".join(item["description"].lower() for item in flagged)
    assert "hyperkalemia" in descriptions
    assert any(item["severity"] == "high" for item in flagged)


def test_graph_compiles() -> None:
    """The StateGraph builds and compiles without error."""
    compiled = build_graph().compile()
    assert compiled is not None


class _StubTool:
    """Minimal stand-in exposing the tool ``.invoke`` interface used by nodes."""

    def __init__(self, fn):
        self._fn = fn

    def invoke(self, args: dict):
        return self._fn(**args)


def test_low_risk_patient_skips_pipeline(monkeypatch) -> None:
    """A low-risk patient with no medications ends after assess_risk.

    The abbreviated-plan short-circuit applies only when risk is low AND there are
    no medications to reconcile/check (so drug-interaction checks are never
    skipped for medicated patients).
    """
    # Force a low risk score regardless of model output.
    monkeypatch.setattr(
        graph_module,
        "get_patient_risk_score",
        _StubTool(
            lambda patient_id, patient_features: {
                "probability": 0.1,
                "risk_category": "low",
                "top_drivers": [],
            }
        ),
    )

    # Fail loudly if retrieval is ever reached for a low-risk patient.
    def _boom(**kwargs):  # pragma: no cover - should not be called
        raise AssertionError("retrieve_guidelines must not run for low-risk patients")

    monkeypatch.setattr(graph_module, "search_guidelines", _StubTool(_boom))

    patient_input = {
        "patient_id": "LOWRISK",
        "age_at_admit": 55,
        "length_of_stay": 2,
        "prior_ip_admits_6mo": 0,
        "prior_ip_admits_12mo": 0,
        "inpatient_reimbursement": 2000.0,
        "sp_chf": 1,
        "medications": [],
    }
    initial: AgentState = {
        "patient_id": "LOWRISK",
        "patient_input": patient_input,
        "flags": [],
        "retry_count": 0,
    }

    compiled = build_graph().compile()
    final_state = compiled.invoke(initial)

    assert final_state["risk_category"] == "low"
    assert final_state.get("retrieved_chunks") is None
    assert "low_risk_minimal_plan" in final_state.get("flags", [])
    assert final_state.get("discharge_summary") is not None
