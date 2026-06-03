"""CLI runner for the eval harness."""

from __future__ import annotations

import json
from pathlib import Path

from hf_readmit.eval.harness import run_eval_suite
from hf_readmit.eval.placeholders import placeholder_agent
from hf_readmit.eval.scenarios import load_scenarios


def main() -> None:
    """Load scenarios, run eval suite, print summary, and write results."""
    # Paths
    project_root = Path(__file__).parent.parent.parent.parent
    scenario_path = project_root / "evals" / "scenarios" / "seed_scenarios.yaml"
    results_dir = project_root / "evals" / "results"

    # Load scenarios
    print(f"Loading scenarios from {scenario_path}...")
    scenarios = load_scenarios(scenario_path)
    print(f"✓ Loaded {len(scenarios)} scenarios\n")

    # Run eval suite
    print("Running eval suite...")
    report = run_eval_suite(scenarios, placeholder_agent)
    print(f"✓ Completed\n")

    # Print summary table
    print("=" * 70)
    print(f"{'Scenario ID':<35} {'Status':<10} {'Reason':<25}")
    print("=" * 70)
    for result in report.results:
        status = "✓ PASS" if result.passed else "✗ FAIL"
        reason = result.failure_reason or ""
        print(f"{result.scenario_id:<35} {status:<10} {reason:<25}")
    print("=" * 70)

    print(f"\nTotal: {report.total_scenarios} scenarios")
    print(f"Passed: {report.passed}")
    print(f"Failed: {report.failed}")
    print(f"Avg trajectory match: {report.aggregate_metrics.get('avg_trajectory_match', 0):.2%}\n")

    # Write results to JSON
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / "latest.json"
    with open(results_file, "w") as f:
        json.dump(report.model_dump(), f, indent=2)
    print(f"✓ Results written to {results_file}")


if __name__ == "__main__":
    main()
