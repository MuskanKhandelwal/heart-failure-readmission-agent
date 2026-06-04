"""Built-in sample patients for demoing and testing the agent.

The production cohort uses opaque SynPUF ``DESYNPUF_ID`` identifiers. For
reproducible demos and tests we ship a small set of synthetic patients keyed by
friendly ids (``TEST001`` …). Each record uses the same raw-input schema the
agent expects: demographics, utilization history, ``sp_*`` comorbidity flags,
and a medication list.
"""

from __future__ import annotations

# TEST001: complex, high-utilization elderly patient with multiple comorbidities
# and an intentionally interacting medication list (ACE-I + K-sparing diuretic,
# and NSAID + loop diuretic) to exercise the full pipeline and the safety checks.
_TEST001 = {
    "patient_id": "TEST001",
    "age_at_admit": 88,
    "length_of_stay": 18,
    "prior_ip_admits_6mo": 6,
    "prior_ip_admits_12mo": 12,
    "inpatient_reimbursement": 50000.0,
    "sp_chf": 1,
    "sp_ckd": 1,
    "sp_diabetes": 1,
    "sp_ischemic_hd": 1,
    "sp_copd": 1,
    "sp_depression": 1,
    "sp_alzheimer": 1,
    "sp_stroke": 1,
    "medications": [
        "lisinopril",      # ACE inhibitor
        "spironolactone",  # K-sparing diuretic -> hyperkalemia risk with ACE-I
        "metoprolol",      # beta-blocker
        "furosemide",      # loop diuretic
        "ibuprofen",       # NSAID -> reduces diuretic efficacy
    ],
}

# TEST002: younger, low-utilization patient with isolated HF and no interacting
# medications -> low population-relative risk -> abbreviated plan / early END.
_TEST002 = {
    "patient_id": "TEST002",
    "age_at_admit": 58,
    "length_of_stay": 2,
    "prior_ip_admits_6mo": 0,
    "prior_ip_admits_12mo": 0,
    "inpatient_reimbursement": 2500.0,
    "sp_chf": 1,
    "sp_ckd": 0,
    "sp_diabetes": 0,
    "sp_ischemic_hd": 0,
    "sp_copd": 0,
    "sp_depression": 0,
    "sp_alzheimer": 0,
    "sp_stroke": 0,
    "medications": ["metoprolol", "furosemide"],
}

SAMPLE_PATIENTS: dict[str, dict] = {
    "TEST001": _TEST001,
    "TEST002": _TEST002,
}


def get_sample_patient(patient_id: str) -> dict:
    """Return a copy of the sample patient with the given id.

    Args:
        patient_id: Friendly sample id (e.g. ``"TEST001"``).

    Returns:
        A deep-ish copy of the patient input dict (medications list copied).

    Raises:
        KeyError: If no sample patient matches ``patient_id``.
    """
    if patient_id not in SAMPLE_PATIENTS:
        raise KeyError(
            f"Unknown sample patient '{patient_id}'. "
            f"Available: {sorted(SAMPLE_PATIENTS)}"
        )
    record = dict(SAMPLE_PATIENTS[patient_id])
    record["medications"] = list(record.get("medications", []))
    return record
