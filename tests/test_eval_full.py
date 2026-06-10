"""Tests for the Unit-7 eval harness.

All agent and LLM calls are mocked; no real agent runs or API calls occur.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from hf_readmit.eval import agent_eval, ragas_eval
from hf_readmit.eval.scenarios import load_scenarios
from hf_readmit.eval.schemas import EvalReport, ScenarioCategory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ADVERSARIAL_PATH = PROJECT_ROOT / "evals" / "scenarios" / "adversarial_scenarios.yaml"
SEED_PATH = PROJECT_ROOT / "evals" / "scenarios" / "seed_scenarios.yaml"

# A mocked final AgentState that proposes two interventions with matching chunks.
FAKE_STATE_WITH_INTERVENTIONS = {
    "patient_id": "X",
    "risk_score": 0.42,
    "risk_category": "medium",
    "conditions": ["heart failure"],
    "retrieved_chunks": [
        {"chunk_id": "c1", "text": "Early follow-up within 7 days is recommended."},
        {"chunk_id": "c2", "text": "Medication reconciliation at discharge is recommended."},
    ],
    "proposed_interventions": [
        {"description": "7-day follow-up", "rationale": "reduce readmission", "citation_chunk_id": "c1", "evidence_level": "Class I"},
        {"description": "Med reconciliation", "rationale": "safety", "citation_chunk_id": "c2", "evidence_level": "Class I"},
    ],
    "discharge_summary": {"citations": []},
    "flags": ["drug_interaction:high:ace_inhibitor+arb"],
    "error": None,
}

FAKE_STATE_NO_INTERVENTIONS = {
    "patient_id": "X",
    "risk_score": 0.1,
    "risk_category": "low",
    "conditions": [],
    "retrieved_chunks": None,
    "proposed_interventions": None,
    "discharge_summary": {"patient_summary": "low risk"},
    "flags": ["low_risk_minimal_plan"],
    "error": None,
}


def test_adversarial_scenarios_load():
    """The full adversarial file has 25 scenarios covering all 7 categories."""
    scenarios = load_scenarios(ADVERSARIAL_PATH)
    assert len(scenarios) == 25
    categories = {s.category for s in scenarios}
    assert categories == set(ScenarioCategory)


def test_agent_eval_runs(tmp_path):
    """run_agent_eval produces a valid EvalReport with mocked agent + no grounding."""
    out = tmp_path / "seed_eval.json"
    with patch.object(agent_eval, "run_agent", return_value=FAKE_STATE_NO_INTERVENTIONS):
        report = agent_eval.run_agent_eval(SEED_PATH, out)

    assert isinstance(report, EvalReport)
    assert report.total_scenarios == 3
    assert len(report.results) == 3
    assert report.passed + report.failed == 3
    assert out.exists()


def test_grounding_judge_called(tmp_path):
    """The LLM-as-judge is invoked once per intervention across all scenarios."""
    out = tmp_path / "seed_eval.json"
    with patch.object(agent_eval, "run_agent", return_value=FAKE_STATE_WITH_INTERVENTIONS), patch.object(
        agent_eval.LLMClient, "offline", new_callable=PropertyMock, return_value=False
    ), patch.object(
        agent_eval.LLMClient, "call", return_value="YES, the excerpt supports it."
    ) as mock_call:
        agent_eval.run_agent_eval(SEED_PATH, out)

    # 3 seed scenarios x 2 interventions each = 6 judge calls.
    assert mock_call.call_count == 6


def test_ragas_eval_structure(tmp_path):
    """run_ragas_eval returns a dict with the four required metric keys."""
    out = tmp_path / "ragas.json"
    fake_scores = {
        "context_precision": 0.85,
        "context_recall": 0.80,
        "faithfulness": 0.90,
        "answer_relevancy": 0.88,
    }
    fake_retriever = MagicMock()
    fake_retriever.retrieve.return_value = [{"text": "ctx"}]
    with patch("hf_readmit.agent.tools._get_retriever", return_value=fake_retriever), patch.object(
        ragas_eval, "_evaluate_ragas", return_value=fake_scores
    ):
        result = ragas_eval.run_ragas_eval(out)

    for key in ("context_precision", "context_recall", "faithfulness", "answer_relevancy"):
        assert key in result
    assert out.exists()
