"""YAML scenario loader and validator."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from hf_readmit.eval.schemas import AdversarialScenario


def load_scenarios(path: Path) -> list[AdversarialScenario]:
    """Load and validate adversarial scenarios from a YAML file.

    Args:
        path: Path to YAML file containing scenario definitions.

    Returns:
        List of validated AdversarialScenario objects.

    Raises:
        FileNotFoundError: If the scenario file does not exist.
        ValidationError: If a scenario fails Pydantic validation.
        yaml.YAMLError: If the YAML is malformed.
    """
    if not path.exists():
        raise FileNotFoundError(f"Scenario file not found: {path}")

    with open(path, "r") as f:
        raw_data = yaml.safe_load(f)

    if not isinstance(raw_data, list):
        raise ValueError(f"Expected YAML list of scenarios, got {type(raw_data).__name__}")

    scenarios = []
    for idx, scenario_dict in enumerate(raw_data):
        try:
            scenario = AdversarialScenario(**scenario_dict)
            scenarios.append(scenario)
        except ValidationError as e:
            scenario_id = scenario_dict.get("id", f"<unknown at index {idx}>")
            raise ValidationError.from_exception_data(
                title="AdversarialScenario",
                line_errors=[
                    {
                        "type": "value_error",
                        "loc": ("scenarios", scenario_id),
                        "msg": f"Validation error in scenario {scenario_id}: {e}",
                        "input": scenario_dict,
                    }
                ],
            ) from e

    return scenarios
