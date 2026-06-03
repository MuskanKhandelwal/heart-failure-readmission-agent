"""Placeholder agent for eval harness."""

from __future__ import annotations


def placeholder_agent(patient_input: dict) -> dict:
    """Placeholder agent that returns a fixed response structure.

    This allows the eval harness to execute end-to-end before the real
    agent is implemented. The response structure matches what the
    harness expects to evaluate.

    Args:
        patient_input: Patient data dict (not used by placeholder).

    Returns:
        Dict with keys: tools_called, flags_triggered, interventions, citations.
    """
    return {
        "tools_called": [],
        "flags_triggered": [],
        "interventions": ["review_medications"],
        "citations": [],
    }
