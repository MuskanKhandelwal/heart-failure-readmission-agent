"""Curated catalog of evidence-based heart failure interventions.

These entries encode the AHA/ACC 2022 HF guideline recommendations the agent is
allowed to propose. The ``propose_plan`` node prefers interventions from this
catalog: when a catalog entry's keywords match a retrieved guideline chunk, the
proposed intervention cites *both* the catalog entry (for the evidence class and
guideline section) and the retrieved chunk_id (for provenance/grounding).

The catalog is intentionally small and hand-curated rather than generated, so the
agent cannot invent interventions that lack a guideline basis.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Intervention(BaseModel):
    """A single evidence-based discharge intervention."""

    intervention_id: str = Field(..., description="Stable catalog identifier")
    name: str = Field(..., description="Human-readable intervention name")
    indication: str = Field(..., description="When this intervention applies")
    evidence_level: str = Field(
        ..., description="Guideline class of recommendation (Class I/II/III)"
    )
    guideline_section: str = Field(..., description="Source guideline section")
    keywords: list[str] = Field(
        default_factory=list,
        description="Lowercase terms used to match retrieved guideline chunks",
    )


INTERVENTION_CATALOG: list[Intervention] = [
    Intervention(
        intervention_id="hf-followup-7day",
        name="7-day post-discharge follow-up visit",
        indication="All HF patients discharged from an index admission",
        evidence_level="Class I",
        guideline_section="AHA/ACC 2022 HF Guideline — Transitions of Care",
        keywords=["follow-up", "follow up", "7 days", "early follow", "post-discharge", "transition"],
    ),
    Intervention(
        intervention_id="hf-med-reconciliation",
        name="Medication reconciliation at discharge",
        indication="All HF patients at discharge",
        evidence_level="Class I",
        guideline_section="AHA/ACC 2022 HF Guideline — Transitions of Care",
        keywords=["medication reconciliation", "reconcil", "discharge medication"],
    ),
    Intervention(
        intervention_id="hf-daily-weight",
        name="Daily weight monitoring",
        indication="All HF patients for congestion self-monitoring",
        evidence_level="Class I",
        guideline_section="AHA/ACC 2022 HF Guideline — Self-Care",
        keywords=["daily weight", "weight monitoring", "weigh", "body weight"],
    ),
    Intervention(
        intervention_id="hf-fluid-restriction",
        name="Fluid restriction counseling",
        indication="Selected HF patients with congestion or hyponatremia",
        evidence_level="Class IIa",
        guideline_section="AHA/ACC 2022 HF Guideline — Self-Care",
        keywords=["fluid restriction", "fluid intake", "fluid"],
    ),
    Intervention(
        intervention_id="hf-acei-arb-arni",
        name="ACE inhibitor / ARB / ARNI optimization",
        indication="HFrEF; ARNI preferred where tolerated",
        evidence_level="Class I",
        guideline_section="AHA/ACC 2022 HF Guideline — GDMT (RAAS)",
        keywords=["arni", "sacubitril", "ace inhibitor", "acei", "arb", "renin-angiotensin", "valsartan"],
    ),
    Intervention(
        intervention_id="hf-beta-blocker",
        name="Beta-blocker optimization",
        indication="HFrEF; evidence-based beta-blocker titration",
        evidence_level="Class I",
        guideline_section="AHA/ACC 2022 HF Guideline — GDMT (Beta-Blockers)",
        keywords=["beta blocker", "beta-blocker", "carvedilol", "metoprolol", "bisoprolol"],
    ),
    Intervention(
        intervention_id="hf-diuretic-adjust",
        name="Diuretic dose adjustment",
        indication="HF patients with volume overload",
        evidence_level="Class I",
        guideline_section="AHA/ACC 2022 HF Guideline — Diuretics",
        keywords=["diuretic", "loop diuretic", "furosemide", "decongestion", "volume"],
    ),
    Intervention(
        intervention_id="hf-cardiac-rehab",
        name="Cardiac rehabilitation referral",
        indication="Stable HF patients to improve functional status",
        evidence_level="Class IIa",
        guideline_section="AHA/ACC 2022 HF Guideline — Exercise & Rehabilitation",
        keywords=["cardiac rehabilitation", "cardiac rehab", "exercise training", "rehabilitation"],
    ),
    Intervention(
        intervention_id="hf-sodium-education",
        name="Patient education — sodium restriction",
        indication="All HF patients for dietary self-care",
        evidence_level="Class I",
        guideline_section="AHA/ACC 2022 HF Guideline — Self-Care / Education",
        keywords=["sodium", "salt", "dietary", "education", "diet"],
    ),
    Intervention(
        intervention_id="hf-telehealth-followup",
        name="Telehealth follow-up enrollment",
        indication="Patients with access barriers or for remote monitoring",
        evidence_level="Class IIb",
        guideline_section="AHA/ACC 2022 HF Guideline — Remote Monitoring",
        keywords=["telehealth", "telemonitoring", "remote monitoring", "telephone"],
    ),
]


def catalog_by_id() -> dict[str, Intervention]:
    """Return the catalog indexed by ``intervention_id``."""
    return {item.intervention_id: item for item in INTERVENTION_CATALOG}
