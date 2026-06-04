"""Shared state schema for the LangGraph discharge-planning agent.

The agent threads a single :class:`AgentState` ``TypedDict`` through every node.
``total=False`` is used so individual nodes may return partial updates; LangGraph
merges each returned mapping into the running state. The runner
(:mod:`hf_readmit.agent.run`) is responsible for seeding the required keys
(``patient_id``, ``patient_input``, ``flags``, ``retry_count``) before invocation.
"""

from __future__ import annotations

from typing import TypedDict


class AgentState(TypedDict, total=False):
    """State object passed between agent nodes.

    Attributes:
        patient_id: Stable identifier for the patient/encounter.
        patient_input: Raw clinical input dict (demographics, comorbidity flags,
            medications, utilization history).
        risk_score: Calibrated 30-day readmission probability in ``[0, 1]``.
        risk_category: Population-relative band — ``"low"``, ``"medium"`` or ``"high"``.
        top_drivers: SHAP-based feature attributions for the prediction.
        conditions: Human-readable condition names derived from comorbidity flags.
        retrieved_chunks: Deduplicated guideline chunks returned by retrieval.
        proposed_interventions: Candidate interventions, each citing a chunk_id.
        grounding_failures: chunk_ids of interventions that failed grounding.
        discharge_summary: Final structured discharge summary.
        flags: Safety/diagnostic flags accumulated across the run.
        retry_count: Number of propose→safety revision loops taken.
        error: Terminal error message, if the run failed.
    """

    patient_id: str
    patient_input: dict
    risk_score: float | None
    risk_category: str | None
    top_drivers: list[dict] | None
    conditions: list[str] | None
    retrieved_chunks: list[dict] | None
    proposed_interventions: list[dict] | None
    grounding_failures: list[str] | None
    discharge_summary: dict | None
    flags: list[str]
    retry_count: int
    error: str | None
