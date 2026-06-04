"""Tools available to the discharge-planning agent.

All four tools are decorated with ``langchain_core.tools.tool`` so they can be
invoked uniformly (``tool.invoke({...})``) and, in a future unit, bound to an
LLM for autonomous tool-calling. Heavy resources (the risk model and the BM25
index) are cached at module scope so repeated tool calls within a run do not
re-load multi-megabyte pickles.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

from hf_readmit.config import settings
from hf_readmit.models.predictor import ReadmissionPredictor
from hf_readmit.rag.bm25_index import BM25Index
from hf_readmit.rag.retriever import ChromaVectorStore, HybridRetriever, OpenAIEmbeddings

logger = logging.getLogger(__name__)

# Project root: src/hf_readmit/agent/tools.py -> parents[3] is the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = PROJECT_ROOT / "models" / "hf_readmit_model.pkl"
BM25_PATH = PROJECT_ROOT / "models" / "bm25_index.pkl"
CHROMA_PATH = PROJECT_ROOT / "chroma_db"

# Approximate 75th percentile of cohort inpatient reimbursement, used to recreate
# the training-time ``high_reimbursement`` feature without re-deriving the cohort
# quantile at inference time. Documented approximation (see README "Risks").
HIGH_REIMBURSEMENT_THRESHOLD = 8000.0

# Population-relative risk bands. The SynPUF model is weakly predictive
# (AUROC 0.563) so calibrated probabilities cluster near the ~10.6% base rate;
# the predictor's absolute bands (<0.2 low) would label every patient "low".
# These bands are tuned to the SynPUF probability distribution so the agent can
# triage relatively. On real Medicare claims you would use absolute clinical
# thresholds instead. See README "Risks".
RISK_BAND_HIGH = 0.11
RISK_BAND_MEDIUM = 0.085

# Comorbidity flag -> readable condition name.
CONDITION_NAMES: dict[str, str] = {
    "sp_chf": "heart failure",
    "sp_ckd": "chronic kidney disease",
    "sp_diabetes": "diabetes mellitus",
    "sp_ischemic_hd": "ischemic heart disease",
    "sp_copd": "chronic obstructive pulmonary disease",
    "sp_depression": "depression",
    "sp_alzheimer": "Alzheimer's disease / dementia",
    "sp_stroke": "stroke / transient ischemic attack",
}

# Medication name (lowercase) -> drug class used for interaction checks.
DRUG_CLASSES: dict[str, str] = {
    # ACE inhibitors
    "lisinopril": "ace_inhibitor", "enalapril": "ace_inhibitor", "ramipril": "ace_inhibitor",
    "captopril": "ace_inhibitor", "benazepril": "ace_inhibitor",
    # ARBs
    "losartan": "arb", "valsartan": "arb", "candesartan": "arb", "olmesartan": "arb",
    "irbesartan": "arb",
    # ARNI
    "sacubitril": "arni", "sacubitril/valsartan": "arni", "entresto": "arni",
    # Beta-blockers
    "metoprolol": "beta_blocker", "carvedilol": "beta_blocker", "bisoprolol": "beta_blocker",
    "atenolol": "beta_blocker",
    # Non-dihydropyridine calcium channel blockers
    "diltiazem": "non_dhp_ccb", "verapamil": "non_dhp_ccb",
    # Potassium-sparing diuretics
    "spironolactone": "k_sparing_diuretic", "eplerenone": "k_sparing_diuretic",
    "amiloride": "k_sparing_diuretic", "triamterene": "k_sparing_diuretic",
    # Loop / thiazide diuretics
    "furosemide": "diuretic", "torsemide": "diuretic", "bumetanide": "diuretic",
    "hydrochlorothiazide": "diuretic",
    # NSAIDs
    "ibuprofen": "nsaid", "naproxen": "nsaid", "ketorolac": "nsaid", "diclofenac": "nsaid",
    # Other HF-relevant agents
    "digoxin": "digoxin", "amiodarone": "amiodarone",
}

# Five common HF drug-interaction pairs: (class_a, class_b, severity, description).
INTERACTION_PAIRS: list[tuple[str, str, str, str]] = [
    ("ace_inhibitor", "arb",
     "high", "ACE inhibitor + ARB: increased hyperkalemia and renal injury risk"),
    ("ace_inhibitor", "k_sparing_diuretic",
     "high", "ACE inhibitor + potassium-sparing diuretic: hyperkalemia risk"),
    ("beta_blocker", "non_dhp_ccb",
     "high", "Beta-blocker + non-dihydropyridine CCB: bradycardia / heart block risk"),
    ("nsaid", "diuretic",
     "moderate", "NSAID + diuretic: reduced diuretic efficacy and renal risk"),
    ("digoxin", "amiodarone",
     "high", "Digoxin + amiodarone: increased digoxin levels / toxicity risk"),
]


# --------------------------------------------------------------------------- #
# Cached heavy resources
# --------------------------------------------------------------------------- #
_predictor: Optional[ReadmissionPredictor] = None
_retriever: Optional[HybridRetriever] = None


def _get_predictor() -> ReadmissionPredictor:
    """Load and cache the readmission predictor."""
    global _predictor
    if _predictor is None:
        _predictor = ReadmissionPredictor(MODEL_PATH)
    return _predictor


def _get_retriever() -> HybridRetriever:
    """Load and cache the guideline retriever.

    Uses hybrid BM25 + dense retrieval when an OpenAI key is configured;
    otherwise falls back to BM25-only retrieval, which needs no API key and is
    sufficient for grounding. See README "Risks".
    """
    global _retriever
    if _retriever is None:
        bm25 = BM25Index.load(BM25_PATH)
        chroma_store = None
        embedder = None
        if settings.openai_api_key:
            try:
                embedder = OpenAIEmbeddings(api_key=settings.openai_api_key)
                chroma_store = ChromaVectorStore(persist_path=CHROMA_PATH)
            except Exception as exc:  # pragma: no cover - optional dense path
                logger.warning("Falling back to BM25-only retrieval: %s", exc)
                embedder, chroma_store = None, None
        _retriever = HybridRetriever(bm25_index=bm25, chroma_store=chroma_store, embedder=embedder)
    return _retriever


def build_patient_feature_vector(patient_input: dict) -> dict:
    """Map raw clinical input to the exact 15 features the model expects.

    Mirrors the training-time feature engineering in
    :func:`hf_readmit.data.features.build_features` for a single patient.

    Args:
        patient_input: Raw clinical fields (age, LOS, utilization, ``sp_*`` flags).

    Returns:
        Dict keyed by the model's ``feature_names_in_`` columns.
    """
    age = float(patient_input.get("age_at_admit", 0) or 0)
    los = float(patient_input.get("length_of_stay", 0) or 0)
    reimb = float(patient_input.get("inpatient_reimbursement", 0) or 0)

    flags = {name: int(patient_input.get(name, 0) or 0) for name in CONDITION_NAMES}
    comorbidity_count = sum(flags.values())

    features = {
        "age_at_admit": age,
        "length_of_stay": los,
        "los_log": math.log1p(los),
        "high_reimbursement": int(reimb > HIGH_REIMBURSEMENT_THRESHOLD),
        "prior_ip_admits_6mo": int(patient_input.get("prior_ip_admits_6mo", 0) or 0),
        "prior_ip_admits_12mo": int(patient_input.get("prior_ip_admits_12mo", 0) or 0),
        "inpatient_reimbursement": reimb,
        "comorbidity_count": comorbidity_count,
        "multiple_comorbidities": int(comorbidity_count >= 3),
        "ckd_hf_interaction": flags["sp_ckd"] * flags["sp_chf"],
        "diabetes_hf_interaction": flags["sp_diabetes"] * flags["sp_chf"],
        "age_group_lt65": int(age < 65),
        "age_group_65_74": int(65 <= age < 75),
        "age_group_75_84": int(75 <= age < 85),
        "age_group_85_plus": int(age >= 85),
    }
    return features


def categorize_risk(probability: float) -> str:
    """Map a calibrated probability to a population-relative risk band."""
    if probability >= RISK_BAND_HIGH:
        return "high"
    if probability >= RISK_BAND_MEDIUM:
        return "medium"
    return "low"


@tool
def get_patient_risk_score(patient_id: str, patient_features: dict) -> dict:
    """Predict 30-day readmission risk for a patient.

    Loads the calibrated readmission model, scores the supplied model-ready
    features, and returns the probability, a population-relative risk category,
    and the SHAP-based top drivers.

    Args:
        patient_id: Patient/encounter identifier.
        patient_features: Model-ready feature dict (15 model columns).

    Returns:
        Dict with ``probability``, ``risk_category`` and ``top_drivers``.
    """
    predictor = _get_predictor()
    features = dict(patient_features)
    features["patient_id"] = patient_id
    result = predictor.predict(features)
    return {
        "probability": result.probability,
        "risk_category": categorize_risk(result.probability),
        "top_drivers": result.top_drivers,
    }


@tool
def get_patient_conditions(patient_input: dict) -> list[str]:
    """Extract readable condition names from a patient's comorbidity flags.

    Args:
        patient_input: Raw clinical input containing ``sp_*`` comorbidity flags.

    Returns:
        Ordered list of readable condition names for flags set to 1.
    """
    conditions: list[str] = []
    for flag, name in CONDITION_NAMES.items():
        if int(patient_input.get(flag, 0) or 0) == 1:
            conditions.append(name)
    return conditions


@tool
def search_guidelines(query: str, condition: str, k: int = 5) -> list[dict]:
    """Retrieve top-k guideline chunks relevant to a condition.

    Args:
        query: Free-text retrieval query.
        condition: Condition the query relates to (used to enrich the query).
        k: Number of chunks to return.

    Returns:
        List of dicts with ``chunk_id``, ``text``, ``source``, ``section`` and ``score``.
    """
    retriever = _get_retriever()
    enriched = f"{condition} {query}".strip()
    hits = retriever.retrieve(enriched, top_k=k)
    results: list[dict] = []
    for hit in hits:
        metadata = hit.get("metadata") or {}
        section = metadata.get("heading") or (
            f"page {metadata.get('page')}" if metadata.get("page") is not None else ""
        )
        results.append(
            {
                "chunk_id": hit.get("chunk_id"),
                "text": hit.get("document", ""),
                "source": metadata.get("source_name", ""),
                "section": section,
                "score": hit.get("combined_score"),
            }
        )
    return results


@tool
def check_drug_interactions(medications: list[str]) -> list[dict]:
    """Flag known heart-failure-relevant drug interactions.

    Checks the patient's medication list against five common HF interaction
    pairs (by drug class).

    Args:
        medications: Medication names (brand or generic, case-insensitive).

    Returns:
        List of flagged interactions, each with ``drugs``, ``severity`` and
        ``description``.
    """
    present: dict[str, list[str]] = {}
    for med in medications or []:
        drug_class = DRUG_CLASSES.get(str(med).strip().lower())
        if drug_class:
            present.setdefault(drug_class, []).append(str(med))

    flagged: list[dict] = []
    for class_a, class_b, severity, description in INTERACTION_PAIRS:
        if class_a in present and class_b in present:
            flagged.append(
                {
                    "drugs": present[class_a] + present[class_b],
                    "classes": [class_a, class_b],
                    "severity": severity,
                    "description": description,
                }
            )
    return flagged
