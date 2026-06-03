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

## Stack

- **Data:** CMS SynPUF 2008-2010 Sample 1 (synthetic Medicare claims)
- **Cohort:** 10,185 HF index admissions (ICD-9 428.x in any diagnosis position),
  10.6% 30-day readmission rate
- **Model:** XGBoost + isotonic calibration, SHAP explainability, MLflow tracking
- **PDF extraction:** pdfplumber with layout-aware extraction (Docling evaluated
  but ruled out due to CPU processing time on large PDFs — see limitations)
- **RAG:** Hybrid BM25 + dense (text-embedding-3-large) over 710 chunks from
  AHA/ACC 2022 HF Guideline, AHRQ readmission toolkit, SHM BOOST toolkit
- **Vector store:** Chroma (persistent), self-hosted path documented in
  docker/langfuse-selfhost/ for HIPAA-aware deployments
- **Agent:** LangGraph (coming in Unit 5)
- **Eval:** RAGAS + LLM-as-judge + adversarial scenarios (coming in Unit 7)
- **Tracing:** Langfuse Cloud (self-host config in docker/langfuse-selfhost/)
- **Serving:** FastAPI + Streamlit (coming in Unit 6)

## Evaluation results

| Metric | Value | Notes |
|--------|-------|-------|
| Readmission AUROC | 0.563 | Expected on synthetic SynPUF data; published range on real Medicare claims is 0.65-0.72 |
| Readmission AUPRC | 0.131 | Base rate 10.6%; marginal lift over random |
| Retrieval Recall@5 | 0.90 | Hybrid BM25+dense over 710 chunks from 3 guideline PDFs |
| Retrieval Precision@5 | 0.76 | |
| Retrieval MRR | 0.942 | Near-perfect source ranking |
| Citation grounding rate | TBD | Measured in Unit 7 |
| Adversarial pass rate | TBD | Measured in Unit 7 |
| Tool-call trajectory accuracy | TBD | Measured in Unit 7 |

## Known Limitations

- **SynPUF predictive performance:** AUROC 0.563 vs 0.65-0.72 on real Medicare
  claims (Kansagara et al., JAMA 2011). SynPUF chronic condition flags are
  synthetically generated and lack real predictive relationships.
- **PDF extraction:** pdfplumber extracts text only. Images, figures, and
  flowcharts from the AHA guideline are not captured. Docling would improve
  this on GPU infrastructure.
- **Beneficiary data:** Only 2009 beneficiary summary available; 2008/2010
  admissions use 2009 demographic snapshot (±1 year).
- **Drug interactions:** Hardcoded 5-pair lookup for v1. Production would use
  a real drug interaction API (e.g. OpenFDA, DrugBank).
- **RAG corpus:** 3 documents, no 2023 AHA focused update (SGLT2 inhibitor
  recommendations). Production corpus would include full guideline suite.

## Prior Art

- ClinNoteAgents (arxiv 2512.07081, AMIA 2026) — multi-agent HF readmission
  from clinical notes
- G.R.O.O.T (github.com/unrealdhanush/groot) — readmission prediction + RAG summaries
- Microsoft patient-discharge-planning — production discharge planning reference

This project differentiates via: (1) runnable agent eval harness with adversarial
scenarios, (2) explicit failure mode documentation, (3) HIPAA-aware observability
design with self-hosted Langfuse config.

## Notes

- Uses Python 3.11
- Uses `pydantic` for configuration and schemas
- Uses `python-dotenv` for environment loading
- `Langfuse` tracing and self-hosted docker deployment will be documented in `docs/hipaa_design.md`
