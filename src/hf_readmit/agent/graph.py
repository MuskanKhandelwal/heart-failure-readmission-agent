"""LangGraph discharge-planning agent: a 5-node StateGraph.

Pipeline::

    assess_risk ──(low risk)──────────────────────────────► END
        │
        └─(medium/high)─► retrieve_guidelines ─► propose_plan ─► safety_check
                                                      ▲               │
                                                      └──(retry)──────┤
                                                                      │(proceed)
                                                                      ▼
                                                       format_discharge_summary ─► END

Every LLM-backed node (propose / safety / format) has a deterministic offline
fallback so the agent runs without an ``ANTHROPIC_API_KEY``. When a key is
present the same nodes call the model via :class:`~hf_readmit.llm.client.LLMClient`
and parse its JSON output, falling back to the deterministic path on any parse
failure (recording a ``llm_parse_fallback`` flag).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from langgraph.graph import END, START, StateGraph

from hf_readmit.agent import prompts
from hf_readmit.agent.catalog import INTERVENTION_CATALOG, catalog_by_id
from hf_readmit.agent.state import AgentState
from hf_readmit.agent.tools import (
    CONDITION_NAMES,
    DRUG_CLASSES,
    build_patient_feature_vector,
    check_drug_interactions,
    get_patient_conditions,
    get_patient_risk_score,
    search_guidelines,
)
from hf_readmit.llm.client import LLMClient

logger = logging.getLogger(__name__)

MAX_RETRIES = 2

# Standard heart-failure red-flag symptoms surfaced in every discharge summary.
RED_FLAG_SYMPTOMS = [
    "Weight gain of more than 2-3 lbs in a day or 5 lbs in a week",
    "Worsening shortness of breath or difficulty breathing when lying flat",
    "New or worsening swelling in the legs, ankles, or abdomen",
    "Persistent cough or wheezing",
    "Lightheadedness, fainting, or rapid/irregular heartbeat",
]

_llm: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Return the process-wide LLM client (lazily constructed)."""
    global _llm
    if _llm is None:
        _llm = LLMClient()
    return _llm


# --------------------------------------------------------------------------- #
# Grounding helpers
# --------------------------------------------------------------------------- #
def _supports(intervention: dict, chunk_text: Optional[str]) -> bool:
    """Whether a cited chunk's text supports an intervention.

    Offline-generated interventions carry a ``match_term`` (the catalog keyword
    that matched the chunk); we require it to appear in the chunk text. For
    LLM-generated interventions without a ``match_term`` we require at least one
    substantive token from the intervention to appear in the chunk text.
    """
    if not chunk_text:
        return False
    term = intervention.get("match_term")
    if term:
        return term.lower() in chunk_text
    tokens = re.findall(r"[a-zA-Z]{5,}", f"{intervention.get('description', '')} {intervention.get('intervention_id', '')}")
    return any(tok.lower() in chunk_text for tok in tokens)


def _split_grounded(state: AgentState) -> tuple[list[dict], list[dict]]:
    """Partition proposed interventions into (grounded, ungrounded)."""
    by_id = {c["chunk_id"]: (c.get("text") or "").lower() for c in (state.get("retrieved_chunks") or [])}
    grounded, ungrounded = [], []
    for iv in state.get("proposed_interventions") or []:
        cid = iv.get("citation_chunk_id")
        if cid in by_id and _supports(iv, by_id[cid]):
            grounded.append(iv)
        else:
            ungrounded.append(iv)
    return grounded, ungrounded


def _extract_json(text: str):
    """Best-effort extraction of a JSON object/array from model output."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"(\[.*\]|\{.*\})", cleaned, flags=re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise


# --------------------------------------------------------------------------- #
# Input safety / triage helpers
# --------------------------------------------------------------------------- #
_INJECTION_PATTERNS = (
    "ignore previous instructions",
    "ignore all previous",
    "disregard guidelines",
    "disregard the guidelines",
    "you are now",
    "system:",
)

# Recognized medication names. A production system would use an RxNorm/drug-DB
# lookup; this curated set distinguishes real drugs from fabricated ones well
# enough to catch obvious hallucination bait (e.g. "cardiofilin").
_KNOWN_DRUGS = set(DRUG_CLASSES) | {
    "tiotropium", "prednisone", "doxorubicin", "warfarin", "aspirin", "clopidogrel",
    "atorvastatin", "rosuvastatin", "insulin", "empagliflozin", "dapagliflozin",
    "sacubitril/valsartan", "hydralazine", "isosorbide", "ivabradine",
}

# Valid AHA/ACC HF stages are A-D. A "stage" given as a number, roman numeral,
# or a letter outside A-D is not a real HF classification.
_FAKE_STAGE_RE = re.compile(r"\bstage\s*(?:[0-9]+|[ivx]+|[e-z])\b", re.IGNORECASE)


def _iter_strings(obj):
    """Recursively yield every string value within a nested dict/list/scalar."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _iter_strings(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _iter_strings(value)


def _get_age(patient_input: dict):
    """Read patient age from either ``age_at_admit`` or the narrative ``age`` key."""
    age = patient_input.get("age_at_admit", patient_input.get("age"))
    try:
        return float(age) if age is not None else None
    except (TypeError, ValueError):
        return None


def _medication_names(patient_input: dict) -> list[str]:
    """Extract medication name strings, handling both schemas.

    Supports ``medications`` as a list of strings or a list of dicts with a
    ``name`` field (the narrative scenario schema).
    """
    names: list[str] = []
    for med in patient_input.get("medications") or []:
        if isinstance(med, dict):
            name = med.get("name")
            if name:
                names.append(str(name))
        elif med:
            names.append(str(med))
    return names


def _detect_injection(patient_input: dict) -> bool:
    """True if any input string looks like a prompt-injection / override attempt."""
    for text in _iter_strings(patient_input):
        low = text.strip().lower()
        if not low:
            continue
        if low.startswith("{") or low.startswith("["):  # JSON-like override attempt
            return True
        if any(pattern in low for pattern in _INJECTION_PATTERNS):
            return True
    return False


def _detect_fake_hf_stage(patient_input: dict) -> bool:
    """True if the input references an invalid HF stage (valid stages are A-D)."""
    return any(
        "stage" in text.lower() and _FAKE_STAGE_RE.search(text)
        for text in _iter_strings(patient_input)
    )


def _detect_unverified_drug(med_names: list[str]) -> Optional[str]:
    """Return the first medication name that is not a recognized drug, if any."""
    for name in med_names:
        token = name.strip().lower()
        if not token or ":" in token or token.startswith("{"):  # injection-y, handled elsewhere
            continue
        if token in _KNOWN_DRUGS:
            continue
        base = re.split(r"[\s/]", token)[0]
        if base in _KNOWN_DRUGS:
            continue
        if re.fullmatch(r"[a-z][a-z\-]{3,}", base):  # looks like a drug name but unrecognized
            return name
    return None


def _missing_data_flags(patient_input: dict) -> list[str]:
    """Diagnostic flags for incomplete patient input (Fix 3)."""
    flags: list[str] = []
    if not _medication_names(patient_input):
        flags.append("incomplete_medication_data")
    p6 = int(patient_input.get("prior_ip_admits_6mo", 0) or 0)
    p12 = int(patient_input.get("prior_ip_admits_12mo", 0) or 0)
    age = _get_age(patient_input)
    if p6 == 0 and p12 == 0 and age is not None and age > 50:
        flags.append("incomplete_admission_history")
    if all(int(patient_input.get(flag, 0) or 0) == 0 for flag in CONDITION_NAMES):
        flags.append("incomplete_comorbidity_data")
    if age is None or age == 0:
        flags.append("missing_demographics")
    return flags


def _input_safety_gate(patient_input: dict) -> Optional[tuple[list[str], str]]:
    """Pre-processing triage: return (flags, reason) to refuse/escalate, else None.

    Runs BEFORE any tool so suspicious or out-of-scope inputs never trigger risk
    scoring or retrieval. Handles prompt injection, pediatric patients, invalid
    HF stages, and unrecognized (likely fabricated) medications.
    """
    if _detect_injection(patient_input):
        return ["suspicious_patient_input", "injection_attempt"], "potential prompt injection in patient input"
    age = _get_age(patient_input)
    if age is not None and age < 18:
        return ["out_of_guideline_scope", "pediatric_not_supported"], "pediatric patient (age < 18) is outside the adult HF guideline scope"
    if _detect_fake_hf_stage(patient_input):
        return ["potential_hallucination", "unverified_claim"], "input references an invalid HF stage (valid AHA stages are A-D)"
    unknown_drug = _detect_unverified_drug(_medication_names(patient_input))
    if unknown_drug:
        return ["potential_hallucination", "unverified_drug"], f"unrecognized medication '{unknown_drug}' could not be verified"
    return None


def _refusal_summary(patient_id: str, reason: str, flags: list[str]) -> dict:
    """Structured discharge summary used when input is refused/escalated at the gate."""
    return {
        "patient_summary": f"Automated discharge planning was halted for patient {patient_id}.",
        "risk_assessment": "Not assessed — input was flagged before processing.",
        "medications": [],
        "follow_up_plan": ["Escalate to a clinician for manual review."],
        "red_flag_symptoms": RED_FLAG_SYMPTOMS,
        "patient_education": [],
        "citations": [],
        "refusal_reason": reason,
        "flags": flags,
    }


# --------------------------------------------------------------------------- #
# Deterministic offline fallbacks
# --------------------------------------------------------------------------- #
def _offline_propose(state: AgentState) -> list[dict]:
    """Build grounded interventions by matching the catalog to retrieved chunks."""
    chunks = state.get("retrieved_chunks") or []
    interventions: list[dict] = []
    for entry in INTERVENTION_CATALOG:
        matched_chunk = None
        matched_term = None
        for chunk in chunks:
            text = (chunk.get("text") or "").lower()
            for kw in entry.keywords:
                if kw.lower() in text:
                    matched_chunk, matched_term = chunk, kw
                    break
            if matched_chunk:
                break
        if not matched_chunk:
            continue
        interventions.append(
            {
                "intervention_id": entry.intervention_id,
                "description": entry.name,
                "rationale": (
                    f"{entry.indication}. Supported by retrieved guidance in "
                    f"{matched_chunk.get('source', 'guideline')}."
                ),
                "citation_chunk_id": matched_chunk["chunk_id"],
                "evidence_level": entry.evidence_level,
                "match_term": matched_term,
            }
        )
    return interventions


def _build_summary(state: AgentState, interactions: list[dict]) -> dict:
    """Deterministically assemble a structured discharge summary from state."""
    pid = state.get("patient_id", "")
    conditions = state.get("conditions") or []
    interventions = state.get("proposed_interventions") or []
    chunks_by_id = {c["chunk_id"]: c for c in (state.get("retrieved_chunks") or [])}
    meds = _medication_names(state.get("patient_input") or {})

    medications = list(meds)
    for interaction in interactions:
        medications.append(f"WARNING ({interaction['severity']}): {interaction['description']}")

    catalog = catalog_by_id()
    follow_up = []
    education = ["Daily weight monitoring", "Low-sodium diet", "Adhere to all prescribed medications"]
    citations = []
    seen_chunks = set()
    for iv in interventions:
        follow_up.append(f"{iv['description']} ({iv.get('evidence_level', '')})")
        entry = catalog.get(iv.get("intervention_id"))
        if entry and "education" in " ".join(entry.keywords):
            education.append(entry.name)
        cid = iv.get("citation_chunk_id")
        if cid and cid not in seen_chunks and cid in chunks_by_id:
            seen_chunks.add(cid)
            chunk = chunks_by_id[cid]
            citations.append(
                {"chunk_id": cid, "source": chunk.get("source", ""), "section": chunk.get("section", "")}
            )

    risk_score = state.get("risk_score")
    return {
        "patient_summary": (
            f"Patient {pid} with {', '.join(conditions) or 'heart failure'}; "
            f"{state.get('risk_category', 'unknown')} population-relative 30-day readmission risk."
        ),
        "risk_assessment": (
            f"Calibrated 30-day readmission probability "
            f"{risk_score:.3f} ({state.get('risk_category')} band)."
            if isinstance(risk_score, float)
            else "Risk assessment unavailable."
        ),
        "medications": medications,
        "follow_up_plan": follow_up or ["Primary care follow-up within 14 days"],
        "red_flag_symptoms": RED_FLAG_SYMPTOMS,
        "patient_education": education,
        "citations": citations,
    }


def _minimal_plan(state: AgentState) -> dict:
    """Abbreviated discharge plan for low-risk patients."""
    pid = state.get("patient_id", "")
    conditions = state.get("conditions") or []
    meds = _medication_names(state.get("patient_input") or {})
    risk_score = state.get("risk_score")
    return {
        "patient_summary": (
            f"Patient {pid} with {', '.join(conditions) or 'heart failure'}; "
            f"low population-relative 30-day readmission risk."
        ),
        "risk_assessment": (
            f"Calibrated 30-day readmission probability {risk_score:.3f} (low band); "
            f"abbreviated discharge plan appropriate."
            if isinstance(risk_score, float)
            else "Low risk."
        ),
        "medications": meds,
        "follow_up_plan": [
            "Routine primary care follow-up within 14 days",
            "Continue current heart failure medications",
        ],
        "red_flag_symptoms": RED_FLAG_SYMPTOMS,
        "patient_education": ["Daily weight monitoring", "Low-sodium diet", "Medication adherence"],
        "citations": [],
    }


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #
def assess_risk(state: AgentState) -> dict:
    """Node 1: triage input, score risk, derive conditions, short-circuit when safe.

    An input safety gate runs first (before any tool): suspicious or out-of-scope
    inputs are refused/escalated immediately with no tool calls. Otherwise risk is
    scored, conditions derived, missing-data flags added, and low-risk patients
    WITH NO MEDICATIONS take an abbreviated plan; low-risk patients with meds still
    run the full pipeline so drug-interaction checks are never skipped.
    """
    patient_input = state.get("patient_input") or {}
    patient_id = state.get("patient_id", "")

    # Input safety gate (runs before any tool call).
    gate = _input_safety_gate(patient_input)
    if gate is not None:
        gate_flags, reason = gate
        return {
            "flags": list(dict.fromkeys((state.get("flags") or []) + gate_flags)),
            "discharge_summary": _refusal_summary(patient_id, reason, gate_flags),
        }

    features = build_patient_feature_vector(patient_input)
    risk = get_patient_risk_score.invoke({"patient_id": patient_id, "patient_features": features})
    conditions = get_patient_conditions.invoke({"patient_input": patient_input})

    flags = list(state.get("flags") or [])
    flags.extend(_missing_data_flags(patient_input))

    update: dict = {
        "risk_score": risk["probability"],
        "risk_category": risk["risk_category"],
        "top_drivers": risk["top_drivers"],
        "conditions": conditions,
    }

    # Optional clinical interpretation (cosmetic; logged, not persisted).
    llm = get_llm_client()
    if not llm.offline:
        try:
            user = (
                f"Risk score: {risk['probability']:.3f} ({risk['risk_category']}).\n"
                f"Top drivers: {json.dumps(risk['top_drivers'])}\n"
                f"Conditions: {conditions}"
            )
            interpretation = llm.call(user, prompts.ASSESS_RISK_PROMPT, max_tokens=400, task="assess_risk")
            logger.info("Risk interpretation: %s", interpretation)
        except Exception as exc:  # pragma: no cover - interpretation is non-critical
            logger.debug("Risk interpretation skipped: %s", exc)

    # Abbreviated plan ONLY when low risk AND no medications to reconcile/check.
    if risk["risk_category"] == "low" and not _medication_names(patient_input):
        update["discharge_summary"] = _minimal_plan({**state, **update})
        flags.append("low_risk_minimal_plan")

    update["flags"] = list(dict.fromkeys(flags))
    return update


def retrieve_guidelines(state: AgentState) -> dict:
    """Node 2: retrieve and deduplicate guideline chunks for each condition."""
    conditions = state.get("conditions") or []
    merged: dict[str, dict] = {}
    for condition in conditions:
        query = f"{condition} management discharge readmission prevention"
        hits = search_guidelines.invoke({"query": query, "condition": condition, "k": 5})
        for hit in hits:
            if hit.get("chunk_id"):
                merged[hit["chunk_id"]] = hit

    chunks = list(merged.values())
    update: dict = {"retrieved_chunks": chunks}

    flags = list(state.get("flags") or [])
    if not chunks:
        flags.append("no_guidelines_found")

    # Fix 4: flag conditions named in the input that the guideline corpus does
    # not cover (out-of-guideline scope), e.g. sarcoidosis, malignancy, cirrhosis.
    patient_input = state.get("patient_input") or {}
    chunk_blob = " ".join((c.get("text") or "").lower() for c in chunks)
    narrative_conditions = " ".join(str(c).lower() for c in (patient_input.get("comorbidities") or []))
    oog_terms = ["sarcoid", "malignancy", "cancer", "cirrhosis", "congenital", "amyloid"]
    if any(term in narrative_conditions and term not in chunk_blob for term in oog_terms):
        flags.append("out_of_guideline_scope")

    if flags != (state.get("flags") or []):
        update["flags"] = list(dict.fromkeys(flags))
    return update


def propose_plan(state: AgentState) -> dict:
    """Node 3: propose grounded, catalog-preferring interventions."""
    llm = get_llm_client()
    flags = list(state.get("flags") or [])

    if llm.offline:
        return {"proposed_interventions": _offline_propose(state)}

    chunks = state.get("retrieved_chunks") or []
    catalog_summary = [
        {"intervention_id": e.intervention_id, "name": e.name, "evidence_level": e.evidence_level}
        for e in INTERVENTION_CATALOG
    ]
    retry_note = ""
    if state.get("grounding_failures"):
        retry_note = (
            "\nPreviously rejected (ungrounded) citations — do NOT reuse unsupported claims: "
            f"{state['grounding_failures']}"
        )
    user = (
        f"Risk: {state.get('risk_score')} ({state.get('risk_category')}).\n"
        f"Conditions: {state.get('conditions')}\n"
        f"Top drivers: {json.dumps(state.get('top_drivers') or [])}\n"
        f"Catalog: {json.dumps(catalog_summary)}\n"
        f"Retrieved chunks: {json.dumps([{'chunk_id': c['chunk_id'], 'text': (c.get('text') or '')[:800]} for c in chunks])}"
        f"{retry_note}"
    )
    try:
        raw = llm.call(user, prompts.PROPOSE_PLAN_PROMPT, max_tokens=1500, task="propose_plan")
        interventions = _extract_json(raw)
        if not isinstance(interventions, list):
            raise ValueError("propose_plan did not return a JSON array")
        return {"proposed_interventions": interventions}
    except Exception as exc:
        logger.warning("propose_plan LLM parse failed, using offline fallback: %s", exc)
        return {
            "proposed_interventions": _offline_propose(state),
            "flags": flags + ["llm_parse_fallback"],
        }


def safety_check(state: AgentState) -> dict:
    """Node 4: verify grounding, flag drug interactions, decide retry vs proceed."""
    grounded, ungrounded = _split_grounded(state)

    flags = list(state.get("flags") or [])
    meds = _medication_names(state.get("patient_input") or {})
    interactions = check_drug_interactions.invoke({"medications": meds})
    for interaction in interactions:
        classes = set(interaction.get("classes", []))
        flags.append(f"drug_interaction:{interaction['severity']}:{'+'.join(interaction['classes'])}")
        # Map interaction classes to specific clinical safety flags (Fix 2).
        flags.append("contraindicated_combination")
        if "ace_inhibitor" in classes and ("arb" in classes or "k_sparing_diuretic" in classes):
            flags.append("hyperkalemia_risk")
        if "beta_blocker" in classes and "non_dhp_ccb" in classes:
            flags.append("bradycardia_risk")
        if "digoxin" in classes and "amiodarone" in classes:
            flags.append("digoxin_toxicity_risk")
        if "nsaid" in classes and "diuretic" in classes:
            flags.extend(["nsaid_diuretic_interaction", "renal_risk"])
    flags = list(dict.fromkeys(flags))

    retry_count = state.get("retry_count", 0)
    if ungrounded and retry_count < MAX_RETRIES:
        # Retry: keep proposed_interventions intact so propose_plan can revise.
        failures = list(state.get("grounding_failures") or [])
        failures.extend([iv.get("citation_chunk_id") for iv in ungrounded])
        return {
            "retry_count": retry_count + 1,
            "grounding_failures": failures,
            "flags": flags,
        }

    # Proceed: keep only grounded interventions.
    if ungrounded:
        flags.append("max_retries_reached")
    return {"proposed_interventions": grounded, "flags": flags}


def format_discharge_summary(state: AgentState) -> dict:
    """Node 5: produce the final structured discharge summary."""
    llm = get_llm_client()
    meds = _medication_names(state.get("patient_input") or {})
    interactions = check_drug_interactions.invoke({"medications": meds})

    if llm.offline:
        return {"discharge_summary": _build_summary(state, interactions)}

    chunks_by_id = {c["chunk_id"]: c for c in (state.get("retrieved_chunks") or [])}
    user = (
        f"Patient: {state.get('patient_id')}\n"
        f"Conditions: {state.get('conditions')}\n"
        f"Risk: {state.get('risk_score')} ({state.get('risk_category')})\n"
        f"Medications: {meds}\n"
        f"Drug interactions: {json.dumps(interactions)}\n"
        f"Grounded interventions: {json.dumps(state.get('proposed_interventions') or [])}\n"
        f"Chunk sources: {json.dumps({cid: {'source': c.get('source'), 'section': c.get('section')} for cid, c in chunks_by_id.items()})}"
    )
    try:
        raw = llm.call(user, prompts.FORMAT_SUMMARY_PROMPT, max_tokens=1500, task="format_summary")
        summary = _extract_json(raw)
        if not isinstance(summary, dict):
            raise ValueError("format_discharge_summary did not return a JSON object")
        return {"discharge_summary": summary}
    except Exception as exc:
        logger.warning("format LLM parse failed, using offline fallback: %s", exc)
        return {"discharge_summary": _build_summary(state, interactions)}


# --------------------------------------------------------------------------- #
# Routers
# --------------------------------------------------------------------------- #
def route_after_assess(state: AgentState) -> str:
    """End early when assess_risk already produced a summary (gate refusal or
    low-risk abbreviated plan); otherwise continue to retrieval."""
    return "end" if state.get("discharge_summary") is not None else "retrieve"


def route_after_safety(state: AgentState) -> str:
    """Loop back to propose_plan while ungrounded interventions remain."""
    _, ungrounded = _split_grounded(state)
    if ungrounded and state.get("retry_count", 0) <= MAX_RETRIES:
        return "retry"
    return "proceed"


# --------------------------------------------------------------------------- #
# Graph construction
# --------------------------------------------------------------------------- #
def build_graph() -> StateGraph:
    """Build and return the (uncompiled) discharge-planning StateGraph.

    Callers compile it with ``build_graph().compile()``.
    """
    graph = StateGraph(AgentState)

    graph.add_node("assess_risk", assess_risk)
    graph.add_node("retrieve_guidelines", retrieve_guidelines)
    graph.add_node("propose_plan", propose_plan)
    graph.add_node("safety_check", safety_check)
    graph.add_node("format_discharge_summary", format_discharge_summary)

    graph.add_edge(START, "assess_risk")
    graph.add_conditional_edges(
        "assess_risk",
        route_after_assess,
        {"end": END, "retrieve": "retrieve_guidelines"},
    )
    graph.add_edge("retrieve_guidelines", "propose_plan")
    graph.add_edge("propose_plan", "safety_check")
    graph.add_conditional_edges(
        "safety_check",
        route_after_safety,
        {"retry": "propose_plan", "proceed": "format_discharge_summary"},
    )
    graph.add_edge("format_discharge_summary", END)
    return graph
