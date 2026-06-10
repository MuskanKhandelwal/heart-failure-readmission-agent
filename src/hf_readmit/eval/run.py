"""CLI runner for the full eval suite.

Run with ``python -m hf_readmit.eval.run``.

This executes the real agent eval over all 25 adversarial scenarios plus the
RAGAS retrieval eval and writes ``evals/results/latest.json``. It makes many real
LLM calls and is expensive (~$0.10-0.30 for the agent eval, plus RAGAS LLM-judge
calls). For a cheap smoke test, run the agent eval on the 3 seed scenarios:

    python -c "from pathlib import Path; from hf_readmit.eval.agent_eval import \
run_agent_eval; run_agent_eval(Path('evals/scenarios/seed_scenarios.yaml'), \
Path('evals/results/seed_eval.json'))"
"""

from __future__ import annotations

from hf_readmit.eval.harness import run_full_eval_suite


def main() -> None:
    """Run the full agent + RAGAS eval suite."""
    run_full_eval_suite()


if __name__ == "__main__":
    main()
