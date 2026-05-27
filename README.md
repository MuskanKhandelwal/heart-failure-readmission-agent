# Heart Failure Readmission Risk and Guideline-Grounded Discharge Planner

Minimal portfolio scaffold for a clinical AI engineering project. This repository will combine risk modeling, RAG retrieval over heart failure guidelines, a LangGraph-based discharge planning agent, and an adversarial evaluation harness.

## Project structure

- `src/hf_readmit/`: application package
- `docs/hipaa_design.md`: HIPAA-aware design notes
- `tests/`: pytest smoke tests and component tests

## Getting started

1. Create a `.env` file from `.env.example`
2. Install dependencies: `pip install -e .`
3. Run tests: `pytest`

## Notes

- Uses Python 3.11
- Uses `pydantic` for configuration and schemas
- Uses `python-dotenv` for environment loading
- `Langfuse` tracing and self-hosted docker deployment will be documented in `docs/hipaa_design.md`
