"""FastAPI application exposing the heart-failure discharge-planning agent.

Endpoints:
    POST /assess   — run the agent on a patient and return the discharge plan.
    GET  /health   — liveness + whether the model/retriever are loaded.
    GET  /metrics  — in-memory agent execution stats (no database).

The app calls :func:`hf_readmit.agent.run.run_agent` rather than importing the
graph internals, and derives the visited-node list from the returned state.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from hf_readmit.agent.run import run_agent

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# In-memory metrics (reset on process restart; no persistence by design)
# --------------------------------------------------------------------------- #
METRICS: dict[str, Any] = {
    "total_requests": 0,
    "total_processing_time": 0.0,
    "flag_counts": {},
    "recent_runs": [],
}
_MAX_RECENT_RUNS = 20


def _record_metrics(patient_id: str, state: dict, elapsed: float) -> None:
    """Update the in-memory metrics store after an assessment."""
    METRICS["total_requests"] += 1
    METRICS["total_processing_time"] += elapsed
    for flag in state.get("flags") or []:
        METRICS["flag_counts"][flag] = METRICS["flag_counts"].get(flag, 0) + 1
    METRICS["recent_runs"].insert(
        0,
        {
            "patient_id": patient_id,
            "risk_score": state.get("risk_score"),
            "risk_category": state.get("risk_category"),
            "n_flags": len(state.get("flags") or []),
            "processing_time_seconds": round(elapsed, 4),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
    del METRICS["recent_runs"][_MAX_RECENT_RUNS:]


def _derive_nodes_visited(state: dict) -> list[str]:
    """Infer which graph nodes ran from the populated state fields.

    The agent does not record its own trace, so we reconstruct the path:
    ``assess_risk`` always runs; a low-risk patient short-circuits to END;
    otherwise retrieval → propose → safety → format all execute.
    """
    nodes = ["assess_risk"]
    if "low_risk_minimal_plan" in (state.get("flags") or []):
        return nodes
    if state.get("retrieved_chunks") is not None:
        nodes.append("retrieve_guidelines")
    if state.get("proposed_interventions") is not None:
        nodes.extend(["propose_plan", "safety_check"])
    if state.get("discharge_summary") is not None:
        nodes.append("format_discharge_summary")
    return nodes


# --------------------------------------------------------------------------- #
# Request / response models (Pydantic v2)
# --------------------------------------------------------------------------- #
class AssessRequest(BaseModel):
    """Request body for POST /assess."""

    patient_id: str = Field(..., description="Patient/encounter identifier")
    patient_input: dict[str, Any] = Field(
        ..., description="Raw clinical input (demographics, comorbidity flags, meds, utilization)"
    )


class AssessResponse(BaseModel):
    """Response body for POST /assess."""

    patient_id: str
    risk_score: float
    risk_category: str
    top_drivers: list[dict]
    interventions: list[dict]
    discharge_summary: dict
    flags: list[str]
    nodes_visited: list[str]
    processing_time_seconds: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    chroma_loaded: bool


# --------------------------------------------------------------------------- #
# App + lifespan (eager resource loading)
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Eagerly load the risk model and retriever so the first request is fast."""
    app.state.model_loaded = False
    app.state.chroma_loaded = False
    try:
        from hf_readmit.agent import tools

        tools._get_predictor()
        app.state.model_loaded = True
        retriever = tools._get_retriever()
        app.state.chroma_loaded = getattr(retriever, "chroma_store", None) is not None
    except Exception as exc:  # pragma: no cover - startup must not crash the server
        logger.warning("Eager resource load failed: %s", exc)
    yield


app = FastAPI(title="HF Discharge-Planning Agent", version="1.0.0", lifespan=lifespan)

# CORS for local Streamlit (any localhost port).
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe plus model/retriever load status."""
    return HealthResponse(
        status="ok",
        model_loaded=getattr(app.state, "model_loaded", False),
        chroma_loaded=getattr(app.state, "chroma_loaded", False),
    )


@app.post("/assess", response_model=AssessResponse)
def assess(request: AssessRequest) -> AssessResponse:
    """Run the discharge-planning agent on a patient and return the result."""
    patient_input = {**request.patient_input, "patient_id": request.patient_id}

    start = time.perf_counter()
    state = run_agent(patient_input)
    elapsed = time.perf_counter() - start

    _record_metrics(request.patient_id, state, elapsed)

    return AssessResponse(
        patient_id=state.get("patient_id") or request.patient_id,
        risk_score=float(state.get("risk_score") or 0.0),
        risk_category=state.get("risk_category") or "unknown",
        top_drivers=state.get("top_drivers") or [],
        interventions=state.get("proposed_interventions") or [],
        discharge_summary=state.get("discharge_summary") or {},
        flags=state.get("flags") or [],
        nodes_visited=_derive_nodes_visited(state),
        processing_time_seconds=round(elapsed, 4),
    )


@app.get("/metrics")
def metrics() -> dict:
    """Return in-memory agent execution statistics."""
    total = METRICS["total_requests"]
    avg = METRICS["total_processing_time"] / total if total else 0.0
    return {
        "total_requests": total,
        "avg_processing_time_seconds": round(avg, 4),
        "flag_counts": METRICS["flag_counts"],
        "recent_runs": METRICS["recent_runs"],
    }
