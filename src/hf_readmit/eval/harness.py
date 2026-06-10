"""Eval harness for running scenarios and generating reports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from hf_readmit.eval.schemas import (
    AdversarialScenario,
    EvalReport,
    ScenarioResult,
)


def run_scenario(
    scenario: AdversarialScenario,
    agent_fn: Callable[[dict], dict],
) -> ScenarioResult:
    """Run a single scenario against an agent function.

    Args:
        scenario: The adversarial scenario to evaluate.
        agent_fn: Agent function that takes patient_input dict and returns a dict.

    Returns:
        ScenarioResult with pass/fail and extracted tool/flag data.
    """
    # Call the agent
    agent_response = agent_fn(scenario.patient_input)

    # Extract what the agent actually did
    tools_called = agent_response.get("tools_called", [])
    flags_triggered = agent_response.get("flags_triggered", [])

    # Placeholder metric: trajectory_match compares tool sets for equality
    expected_tools_set = set(scenario.expected_tools)
    actual_tools_set = set(tools_called)
    trajectory_match = expected_tools_set == actual_tools_set

    # Placeholder metric: grounding_rate returns None for now
    grounding_rate = None

    # Pass condition: trajectory_match AND all expected flags are in flags_triggered
    expected_flags_set = set(scenario.expected_flags)
    flags_satisfied = expected_flags_set.issubset(set(flags_triggered))
    passed = trajectory_match and flags_satisfied

    failure_reason = None
    if not passed:
        reasons = []
        if not trajectory_match:
            reasons.append(
                f"tool mismatch (expected {expected_tools_set}, got {actual_tools_set})"
            )
        if not flags_satisfied:
            missing_flags = expected_flags_set - set(flags_triggered)
            reasons.append(f"missing flags {missing_flags}")
        failure_reason = "; ".join(reasons)

    return ScenarioResult(
        scenario_id=scenario.id,
        passed=passed,
        tools_called=tools_called,
        flags_triggered=flags_triggered,
        grounding_rate=grounding_rate,
        trajectory_match=trajectory_match,
        failure_reason=failure_reason,
    )


def run_eval_suite(
    scenarios: list[AdversarialScenario],
    agent_fn: Callable[[dict], dict],
) -> EvalReport:
    """Run a full suite of scenarios and generate an eval report.

    Args:
        scenarios: List of scenarios to evaluate.
        agent_fn: Agent function to evaluate.

    Returns:
        EvalReport with aggregate results and metrics.
    """
    results = []
    for scenario in scenarios:
        result = run_scenario(scenario, agent_fn)
        results.append(result)

    passed_count = sum(1 for r in results if r.passed)
    failed_count = len(results) - passed_count

    # Aggregate metrics (placeholder)
    aggregate_metrics = {
        "avg_trajectory_match": sum(
            1 for r in results if r.trajectory_match
        ) / len(results)
        if results
        else 0.0,
    }

    timestamp = datetime.now(timezone.utc).isoformat()

    return EvalReport(
        timestamp=timestamp,
        total_scenarios=len(results),
        passed=passed_count,
        failed=failed_count,
        results=results,
        aggregate_metrics=aggregate_metrics,
    )


def run_full_eval_suite() -> EvalReport:
    """Run the full Unit-7 eval suite: real agent eval (25 scenarios) + RAGAS.

    Runs the adversarial agent eval over all 25 scenarios and the RAGAS retrieval
    eval, merges the RAGAS metrics into the report's aggregate metrics, writes the
    combined report to ``evals/results/latest.json``, and prints a summary.

    WARNING: this makes many real LLM calls and is expensive (see cost notes in
    ``agent_eval`` and ``ragas_eval``). Not exercised by the test suite.

    Returns:
        The combined :class:`EvalReport`.
    """
    # Lazy imports: these pull in the agent graph and RAGAS, which are heavy.
    from hf_readmit.eval.agent_eval import run_agent_eval
    from hf_readmit.eval.ragas_eval import run_ragas_eval

    project_root = Path(__file__).resolve().parents[3]
    scenarios_path = project_root / "evals" / "scenarios" / "adversarial_scenarios.yaml"
    results_dir = project_root / "evals" / "results"
    latest_path = results_dir / "latest.json"
    ragas_path = results_dir / "ragas.json"

    print("Running adversarial agent eval (25 scenarios)...")
    report = run_agent_eval(scenarios_path, latest_path)

    print("\nRunning RAGAS retrieval eval (10 questions)...")
    ragas_scores = run_ragas_eval(ragas_path)
    report.aggregate_metrics["ragas"] = ragas_scores

    # Re-write latest.json with RAGAS metrics merged in.
    latest_path.write_text(json.dumps(report.model_dump(), indent=2))

    print("\nRAGAS metrics:")
    for key, value in ragas_scores.items():
        print(f"  {key}: {value}")
    print(f"\nCombined report written to {latest_path}")
    return report
