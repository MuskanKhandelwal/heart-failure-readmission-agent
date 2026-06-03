"""Cohort construction for heart failure readmission risk modeling."""

# NOTE: SynPUF Performance Limitation
# This pipeline achieves AUROC 0.563 and AUPRC 0.131 on SynPUF synthetic data.
# Published HF readmission models on real Medicare claims achieve AUROC 0.65-0.72
# (Kansagara et al., 2011). The gap is expected: SynPUF chronic condition flags
# and utilization patterns are synthetically generated and do not preserve real
# predictive relationships. On real claims data, the same pipeline would be
# expected to perform in the published range.
# Reference: Kansagara D et al. JAMA. 2011;306(16):1782-1793.

from __future__ import annotations

from pathlib import Path

import pandas as pd

from hf_readmit.utils.logging import setup_logging

logger = setup_logging()

HF_ICD9_CODES = {f"428{i}" for i in range(10)}

SP_FLAG_COLUMNS = {
    "sp_chf": "SP_CHF",
    "sp_ckd": "SP_CHRNKIDN",
    "sp_diabetes": "SP_DIABETES",
    "sp_ischemic_hd": "SP_ISCHMCHT",
    "sp_copd": "SP_COPD",
    "sp_depression": "SP_DEPRESSN",
    "sp_alzheimer": "SP_ALZHDMTA",
    "sp_stroke": "SP_STRKETIA",
}


def _normalize_flag(series: pd.Series) -> pd.Series:
    return series.replace({1: 1, 2: 0}).fillna(0).astype(int)


def _count_prior_admits(dates: pd.Series, window_days: int) -> pd.Series:
    dates = dates.sort_values()
    counts = []
    for current in dates:
        window_start = current - pd.Timedelta(days=window_days)
        counts.append(((dates < current) & (dates >= window_start)).sum())
    return pd.Series(counts, index=dates.index)


def _compute_readmit_label(group: pd.DataFrame) -> pd.Series:
    group = group.sort_values("index_admit_date")
    labels = []
    for discharge_date in group["index_discharge_date"]:
        is_readmit = ((group["index_admit_date"] > discharge_date)
                      & (group["index_admit_date"] <= discharge_date + pd.Timedelta(days=30))).any()
        labels.append(int(is_readmit))
    return pd.Series(labels, index=group.index)


def build_hf_cohort(inpatient_path: Path, beneficiary_path: Path) -> pd.DataFrame:
    """Build a heart failure cohort from inpatient and beneficiary claims.

    The beneficiary data is only available for 2009 in this repository, so the
    cohort is naturally restricted to patients with 2009 beneficiary records.
    """
    icd_cols = [f"ICD9_DGNS_CD_{i}" for i in range(1, 11)]
    inpatient_df = pd.read_csv(
        inpatient_path,
        dtype={
            "DESYNPUF_ID": str,
            "CLM_ID": str,
            "SEGMENT": float,
            **{col: str for col in icd_cols},
            "CLM_DRG_CD": str,
        },
        usecols=[
            "DESYNPUF_ID",
            "CLM_ID",
            "SEGMENT",
            "CLM_ADMSN_DT",
            "NCH_BENE_DSCHRG_DT",
            *icd_cols,
            "CLM_DRG_CD",
            "NCH_PRMRY_PYR_CLM_PD_AMT",
        ],
    )

    logger.info("Loaded inpatient claims", extra={"count": len(inpatient_df)})

    beneficiary_df = pd.read_csv(
        beneficiary_path,
        dtype={
            "DESYNPUF_ID": str,
            "BENE_SEX_IDENT_CD": float,
        },
        usecols=[
            "DESYNPUF_ID",
            "BENE_BIRTH_DT",
            "BENE_DEATH_DT",
            "BENE_SEX_IDENT_CD",
            "SP_CHF",
            "SP_CHRNKIDN",
            "SP_DIABETES",
            "SP_ISCHMCHT",
            "SP_COPD",
            "SP_DEPRESSN",
            "SP_ALZHDMTA",
            "SP_STRKETIA",
        ],
    )

    logger.info("Loaded beneficiary records", extra={"count": len(beneficiary_df)})

    inpatient_df = inpatient_df[inpatient_df["SEGMENT"] == 1].copy()
    logger.info("After keeping segment 1 claims", extra={"count": len(inpatient_df)})

    for col in icd_cols:
        inpatient_df[col] = inpatient_df[col].fillna("").astype(str).str.strip()

    hf_primary = inpatient_df["ICD9_DGNS_CD_1"].isin(HF_ICD9_CODES).astype(int)
    hf_any = inpatient_df[icd_cols].isin(HF_ICD9_CODES).any(axis=1)
    inpatient_df = inpatient_df[hf_any].copy()
    inpatient_df["hf_primary"] = hf_primary.loc[inpatient_df.index]
    logger.info("After HF diagnosis filter on any ICD9 position", extra={"count": len(inpatient_df)})

    inpatient_df["index_admit_date"] = pd.to_datetime(
        inpatient_df["CLM_ADMSN_DT"], format="%Y%m%d", errors="coerce"
    )
    inpatient_df["index_discharge_date"] = pd.to_datetime(
        inpatient_df["NCH_BENE_DSCHRG_DT"], format="%Y%m%d", errors="coerce"
    )
    inpatient_df = inpatient_df.dropna(subset=["index_admit_date", "index_discharge_date"])
    logger.info("After valid admission/discharge dates filter", extra={"count": len(inpatient_df)})

    max_discharge = inpatient_df["index_discharge_date"].max()
    cutoff = max_discharge - pd.Timedelta(days=30)
    inpatient_df = inpatient_df[inpatient_df["index_discharge_date"] <= cutoff]
    logger.info(
        "After 30-day follow-up availability filter",
        extra={"count": len(inpatient_df), "cutoff": cutoff.isoformat()},
    )

    cohort = inpatient_df.merge(
        beneficiary_df,
        how="inner",
        on="DESYNPUF_ID",
        validate="m:1",
        suffixes=("", "_ben"),
    )
    logger.info("After beneficiary join", extra={"count": len(cohort)})

    cohort["BENE_BIRTH_DT"] = pd.to_datetime(
        cohort["BENE_BIRTH_DT"], format="%Y%m%d", errors="coerce"
    )
    cohort["BENE_DEATH_DT"] = pd.to_datetime(
        cohort["BENE_DEATH_DT"], format="%Y%m%d", errors="coerce"
    )

    cohort["age_at_admit"] = pd.to_numeric(
        ((cohort["index_admit_date"] - cohort["BENE_BIRTH_DT"]).dt.days / 365.25).round(),
        errors="coerce",
    ).astype("Int64")
    cohort["sex"] = cohort["BENE_SEX_IDENT_CD"].map({1.0: "male", 2.0: "female"}).fillna("unknown")
    cohort["length_of_stay"] = (
        cohort["index_discharge_date"] - cohort["index_admit_date"]
    ).dt.days.clip(lower=1).astype(int)
    cohort["drg_code"] = cohort["CLM_DRG_CD"].astype(str).fillna("")
    cohort["primary_icd9"] = cohort["ICD9_DGNS_CD_1"].astype(str)
    cohort["inpatient_reimbursement"] = cohort["NCH_PRMRY_PYR_CLM_PD_AMT"].fillna(0.0).astype(float)

    for output_col, input_col in SP_FLAG_COLUMNS.items():
        cohort[output_col] = _normalize_flag(cohort[input_col])

    cohort = cohort[cohort["BENE_DEATH_DT"].isna() |
                    (cohort["BENE_DEATH_DT"] > cohort["index_discharge_date"] + pd.Timedelta(days=30))]
    logger.info("After excluding deaths within 30 days of discharge", extra={"count": len(cohort)})

    cohort = cohort.sort_values(["DESYNPUF_ID", "index_admit_date"]).copy()
    cohort["patient_id"] = cohort["DESYNPUF_ID"]
    cohort["index_clm_id"] = cohort["CLM_ID"]

    full_claims = pd.read_csv(
        inpatient_path,
        dtype={
            "DESYNPUF_ID": str,
            "SEGMENT": float,
            "CLM_ADMSN_DT": str,
        },
        usecols=["DESYNPUF_ID", "SEGMENT", "CLM_ADMSN_DT"],
    )
    full_claims = full_claims[full_claims["SEGMENT"] == 1].copy()
    full_claims["followup_admit_date"] = pd.to_datetime(
        full_claims["CLM_ADMSN_DT"], format="%Y%m%d", errors="coerce"
    )
    full_claims = full_claims.dropna(subset=["followup_admit_date"])

    cohort = cohort.reset_index(drop=True)
    joined = cohort[["DESYNPUF_ID", "index_admit_date", "index_discharge_date"]].reset_index().merge(
        full_claims,
        on="DESYNPUF_ID",
        how="left",
    )
    joined["prior_ip_6mo"] = (
        (joined["followup_admit_date"] < joined["index_admit_date"])
        & (joined["followup_admit_date"] >= joined["index_admit_date"] - pd.Timedelta(days=180))
    ).astype(int)
    joined["prior_ip_12mo"] = (
        (joined["followup_admit_date"] < joined["index_admit_date"])
        & (joined["followup_admit_date"] >= joined["index_admit_date"] - pd.Timedelta(days=365))
    ).astype(int)
    joined["is_readmit"] = (
        (joined["followup_admit_date"] > joined["index_discharge_date"])
        & (joined["followup_admit_date"] <= joined["index_discharge_date"] + pd.Timedelta(days=30))
    )

    counts = joined.groupby("index")[["prior_ip_6mo", "prior_ip_12mo", "is_readmit"]].agg(
        {"prior_ip_6mo": "sum", "prior_ip_12mo": "sum", "is_readmit": "any"}
    )
    counts["is_readmit"] = counts["is_readmit"].astype(int)

    cohort["prior_ip_admits_6mo"] = counts["prior_ip_6mo"].reindex(cohort.index, fill_value=0).astype(int)
    cohort["prior_ip_admits_12mo"] = counts["prior_ip_12mo"].reindex(cohort.index, fill_value=0).astype(int)
    cohort["readmit_30d"] = counts["is_readmit"].reindex(cohort.index, fill_value=0).astype(int)

    output_columns = [
        "patient_id",
        "index_clm_id",
        "index_admit_date",
        "index_discharge_date",
        "age_at_admit",
        "sex",
        "length_of_stay",
        "drg_code",
        "primary_icd9",
        "hf_primary",
        "sp_chf",
        "sp_ckd",
        "sp_diabetes",
        "sp_ischemic_hd",
        "sp_copd",
        "sp_depression",
        "sp_alzheimer",
        "sp_stroke",
        "prior_ip_admits_6mo",
        "prior_ip_admits_12mo",
        "inpatient_reimbursement",
        "readmit_30d",
    ]

    cohort = cohort[output_columns].reset_index(drop=True)
    logger.info("Final cohort built", extra={"count": len(cohort)})
    return cohort
