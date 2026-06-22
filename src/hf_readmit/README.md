# hf_readmit

This package contains the heart failure readmission prediction agent and its supporting infrastructure.

## Folder Structure

### agent
The core agent implementation using LangGraph. Contains the multi-step workflow that orchestrates risk assessment, guideline retrieval, intervention planning, and safety checks.
- `graph.py`: Main agent workflow definition
- `run.py`: Agent execution entry point
- `tools.py`: Tools used by the agent (risk model, retriever, LLM)
- `prompts.py`: LLM prompts for various agent steps
- `state.py`: Agent state schema definition
- `catalog.py`: Configuration for agent steps and outputs

### api
FastAPI web service exposing the agent through HTTP endpoints.
- `app.py`: FastAPI application with routes for assessment, health checks, and metrics

### ui
Streamlit web interface for clinicians and monitoring.
- `app.py`: Multi-page Streamlit entry point with navigation
- `clinician.py`: Patient assessment interface
- `monitoring.py`: Metrics and execution tracking dashboard

### models
Risk prediction model for identifying high-risk readmission patients.
- `predictor.py`: Risk prediction model and inference logic
- `train.py`: Model training code
- `explain.py`: Feature importance and model explanation
- `run_training.py`: Training pipeline script

### rag
Retrieval-augmented generation system for clinical guidelines.
- `retriever.py`: Vector and BM25 retrieval from clinical guidelines
- `ingest.py`: Document loading and embedding pipeline
- `bm25_index.py`: BM25 search index implementation
- `eval.py`: Retriever evaluation metrics

### data
Data loading and processing utilities.
- Patient data readers and preprocessing functions

### llm
LLM integration and fallback logic.
- LLM client initialization and configuration
- Offline fallback when API keys are unavailable

### eval
Evaluation harness for agent quality assessment.
- Adversarial and RAGAS evaluation frameworks
- Test data generation and metrics

### utils
General utility functions used across the package.
- Helper functions for logging, validation, and common operations
