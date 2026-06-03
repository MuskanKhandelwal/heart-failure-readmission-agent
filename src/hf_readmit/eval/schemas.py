"""Pydantic schemas for the eval domain."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ScenarioCategory(str, Enum):
    """Categories for adversarial scenarios."""

    CONTRADICTORY_MEDS = "contradictory_meds"
    EDGE_DEMOGRAPHICS = "edge_demographics"
    OUT_OF_GUIDELINE = "out_of_guideline"
    CONFLICTING_COMORBIDITIES = "conflicting_comorbidities"
    MISSING_DATA = "missing_data"
    PROMPT_INJECTION = "prompt_injection"
    HALLUCINATION_BAIT = "hallucination_bait"


class ExpectedBehavior(str, Enum):
    """Expected behavior patterns for agent."""

    COMPLETE = "complete"
    REFUSE = "refuse"
    ESCALATE = "escalate"
    LOOP_FOR_REVISION = "loop_for_revision"


class AdversarialScenario(BaseModel):
    """A single adversarial scenario for evaluation."""

    id: str = Field(..., description="Unique scenario identifier")
    category: ScenarioCategory = Field(..., description="Scenario category")
    description: str = Field(..., description="Human-readable scenario description")
    patient_input: dict = Field(..., description="Patient data or query input")
    expected_tools: list[str] = Field(
        default_factory=list, description="Tools expected to be called"
    )
    expected_flags: list[str] = Field(
        default_factory=list, description="Safety/diagnostic flags expected to trigger"
    )
    expected_behavior: ExpectedBehavior = Field(
        ..., description="Expected agent behavior"
    )


class ScenarioResult(BaseModel):
    """Result of running a single scenario."""

    scenario_id: str = Field(..., description="ID of the scenario")
    passed: bool = Field(..., description="Whether scenario passed")
    tools_called: list[str] = Field(
        default_factory=list, description="Tools actually called by agent"
    )
    flags_triggered: list[str] = Field(
        default_factory=list, description="Flags actually triggered"
    )
    grounding_rate: Optional[float] = Field(
        default=None, description="Fraction of outputs grounded in guidelines"
    )
    trajectory_match: bool = Field(
        ..., description="Whether tool trajectory matched expected"
    )
    failure_reason: Optional[str] = Field(
        default=None, description="Reason for failure if passed=False"
    )


class EvalReport(BaseModel):
    """Aggregated evaluation report."""

    timestamp: str = Field(..., description="ISO 8601 timestamp")
    total_scenarios: int = Field(..., description="Total scenarios run")
    passed: int = Field(..., description="Number of scenarios that passed")
    failed: int = Field(..., description="Number of scenarios that failed")
    results: list[ScenarioResult] = Field(..., description="Individual scenario results")
    aggregate_metrics: dict = Field(
        default_factory=dict, description="Aggregate metrics across all scenarios"
    )
