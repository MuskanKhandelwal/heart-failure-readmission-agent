"""Agent runner and CLI entry point.

``run_agent`` builds and invokes the compiled graph, wiring a Langfuse callback
handler into the run when Langfuse credentials are configured. The module is also
runnable as ``python -m hf_readmit.agent.run --patient-id TEST001``.
"""

from __future__ import annotations

import argparse
import json
import logging

from hf_readmit.agent.graph import build_graph, get_llm_client
from hf_readmit.agent.samples import SAMPLE_PATIENTS, get_sample_patient
from hf_readmit.agent.state import AgentState
from hf_readmit.config import settings

logger = logging.getLogger(__name__)


def _initial_state(patient_input: dict) -> AgentState:
    """Seed a fresh :class:`AgentState` from raw patient input."""
    return {
        "patient_id": str(patient_input.get("patient_id", "")),
        "patient_input": patient_input,
        "risk_score": None,
        "risk_category": None,
        "top_drivers": None,
        "conditions": None,
        "retrieved_chunks": None,
        "proposed_interventions": None,
        "grounding_failures": None,
        "discharge_summary": None,
        "flags": [],
        "retry_count": 0,
        "error": None,
    }


def _langfuse_config() -> dict:
    """Build a RunnableConfig wiring the Langfuse callback handler if available.

    Returns an empty config when Langfuse credentials are absent, so the agent
    runs identically offline.
    """
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return {}
    try:
        from langfuse.callback import CallbackHandler

        handler = CallbackHandler(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        return {"callbacks": [handler]}
    except Exception as exc:  # pragma: no cover - tracing must not break runs
        logger.warning("Langfuse callback handler unavailable: %s", exc)
        return {}


def run_agent(patient_input: dict) -> AgentState:
    """Run the discharge-planning agent on a single patient.

    Args:
        patient_input: Raw clinical input dict (see ``samples.py`` for schema).

    Returns:
        The final :class:`AgentState` after the graph reaches END.
    """
    graph = build_graph().compile()
    config = _langfuse_config()
    try:
        final_state: AgentState = graph.invoke(_initial_state(patient_input), config=config)
    except Exception as exc:
        logger.exception("Agent run failed")
        state = _initial_state(patient_input)
        state["error"] = str(exc)
        return state
    finally:
        # Flush any buffered Langfuse events.
        if config:
            try:
                from langfuse.decorators import langfuse_context

                langfuse_context.flush()
            except Exception:  # pragma: no cover
                pass
    return final_state


def _print_trace(final_state: AgentState) -> None:
    """Pretty-print a human-readable trace of the completed run."""
    llm = get_llm_client()
    mode = "OFFLINE (deterministic fallback — no OPENAI_API_KEY)" if llm.offline else f"ONLINE (OpenAI {llm.model})"
    print("=" * 72)
    print(f"DISCHARGE-PLANNING AGENT RUN  |  LLM mode: {mode}")
    print("=" * 72)

    pid = final_state.get("patient_id")
    print(f"\nPatient: {pid}")
    print(f"Risk score: {final_state.get('risk_score'):.3f}" if isinstance(final_state.get("risk_score"), float) else "Risk score: n/a")
    print(f"Risk category: {final_state.get('risk_category')}")
    print(f"Conditions: {final_state.get('conditions')}")

    drivers = final_state.get("top_drivers") or []
    if drivers:
        print("Top risk drivers (SHAP):")
        for d in drivers[:5]:
            print(f"  - {d.get('feature')}: {d.get('shap_value'):+.4f} ({d.get('direction')})")

    chunks = final_state.get("retrieved_chunks") or []
    print(f"\nRetrieved guideline chunks: {len(chunks)}")
    for c in chunks[:5]:
        print(f"  - {c.get('chunk_id')}  [{c.get('source')}]")

    interventions = final_state.get("proposed_interventions") or []
    print(f"\nGrounded interventions: {len(interventions)}")
    for iv in interventions:
        print(f"  - [{iv.get('evidence_level')}] {iv.get('description')}  (cite: {iv.get('citation_chunk_id')})")

    print(f"\nRetry count: {final_state.get('retry_count')}")
    print(f"Flags: {final_state.get('flags')}")
    if final_state.get("grounding_failures"):
        print(f"Grounding failures: {final_state.get('grounding_failures')}")
    if final_state.get("error"):
        print(f"ERROR: {final_state.get('error')}")

    print("\n" + "-" * 72)
    print("DISCHARGE SUMMARY")
    print("-" * 72)
    print(json.dumps(final_state.get("discharge_summary") or {}, indent=2))


def main() -> None:
    """CLI: run the agent on a built-in sample patient."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Run the HF discharge-planning agent on a sample patient.")
    parser.add_argument(
        "--patient-id",
        default="TEST001",
        help=f"Sample patient id. Available: {sorted(SAMPLE_PATIENTS)}",
    )
    args = parser.parse_args()

    patient_input = get_sample_patient(args.patient_id)
    final_state = run_agent(patient_input)
    _print_trace(final_state)


if __name__ == "__main__":
    main()
