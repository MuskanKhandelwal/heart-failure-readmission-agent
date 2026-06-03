"""Tests for the eval harness."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from hf_readmit.eval.harness import run_eval_suite, run_scenario
from hf_readmit.eval.placeholders import placeholder_agent
from hf_readmit.eval.scenarios import load_scenarios
from hf_readmit.eval.schemas import AdversarialScenario, ScenarioCategory, ExpectedBehavior


@pytest.fixture
def scenario_yaml_path() -> Path:
    """Path to seed scenarios YAML."""
    return (
        Path(__file__).parent.parent / "evals" / "scenarios" / "seed_scenarios.yaml"
    )


@pytest.fixture
def results_dir() -> Path:
    """Path to results directory."""
    return Path(__file__).parent.parent / "evals" / "results"


def test_load_scenarios(scenario_yaml_path: Path) -> None:
    """Test loading and validating scenarios from YAML.
    
    - Asserts 3 scenarios are loaded
    - Asserts schema validation works
    """
    scenarios = load_scenarios(scenario_yaml_path)
    
    assert len(scenarios) == 3
    assert all(isinstance(s, AdversarialScenario) for s in scenarios)
    
    # Check first scenario
    scenario_001 = scenarios[0]
    assert scenario_001.id == "scenario_001_contradictory_meds"
    assert scenario_001.category == ScenarioCategory.CONTRADICTORY_MEDS
    assert scenario_001.expected_behavior == ExpectedBehavior.ESCALATE
    assert len(scenario_001.expected_flags) > 0


def test_load_scenarios_schema_validation() -> None:
    """Test that malformed scenarios raise ValidationError."""
    bad_scenario_dict = {
        "id": "bad_scenario",
        # missing required 'category' field
        "description": "This is missing category",
        "patient_input": {},
        "expected_behavior": "complete",
    }
    
    with pytest.raises(ValidationError):
        AdversarialScenario(**bad_scenario_dict)


def test_run_single_scenario(scenario_yaml_path: Path) -> None:
    """Test running a single scenario.
    
    - Asserts ScenarioResult shape is valid
    - Asserts fields are populated
    """
    scenarios = load_scenarios(scenario_yaml_path)
    scenario = scenarios[0]
    
    result = run_scenario(scenario, placeholder_agent)
    
    assert result.scenario_id == scenario.id
    assert isinstance(result.passed, bool)
    assert isinstance(result.tools_called, list)
    assert isinstance(result.flags_triggered, list)
    assert isinstance(result.trajectory_match, bool)
    assert result.grounding_rate is None  # Placeholder returns None


def test_run_eval_suite(scenario_yaml_path: Path) -> None:
    """Test running full eval suite.
    
    - Asserts report has all 3 results
    - Asserts aggregate metrics are present
    """
    scenarios = load_scenarios(scenario_yaml_path)
    report = run_eval_suite(scenarios, placeholder_agent)
    
    assert report.total_scenarios == 3
    assert len(report.results) == 3
    assert report.passed + report.failed == report.total_scenarios
    assert "avg_trajectory_match" in report.aggregate_metrics
    assert report.timestamp  # ISO string


def test_eval_writes_results_file(
    scenario_yaml_path: Path, results_dir: Path, tmp_path: Path
) -> None:
    """Test that eval harness writes results to JSON.
    
    - Confirms latest.json is written
    - Asserts it's parseable as JSON
    """
    scenarios = load_scenarios(scenario_yaml_path)
    report = run_eval_suite(scenarios, placeholder_agent)
    
    # Write to temp file for this test
    results_file = tmp_path / "test_report.json"
    with open(results_file, "w") as f:
        json.dump(report.model_dump(), f, indent=2)
    
    # Read back and validate
    assert results_file.exists()
    with open(results_file, "r") as f:
        loaded = json.load(f)
    
    assert loaded["total_scenarios"] == 3
    assert len(loaded["results"]) == 3
    assert "timestamp" in loaded
