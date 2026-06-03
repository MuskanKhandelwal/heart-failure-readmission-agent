"""Feature engineering for the heart failure readmission model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hf_readmit.utils.logging import setup_logging

logger = setup_logging()


def build_features(cohort_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Engineer features and labels from the cohort DataFrame.

    Args:
        cohort_df: Cohort DataFrame returned by build_hf_cohort.

    Returns:
        Tuple of feature matrix X, label vector y, and feature_names list.
    """
    df = cohort_df.copy()
    df["age_group"] = pd.cut(
        df["age_at_admit"].astype(float),
        bins=[-1, 64, 74, 84, np.inf],
        labels=["lt65", "65_74", "75_84", "85_plus"],
    )

    age_dummies = pd.get_dummies(df["age_group"], prefix="age_group")
    expected_age_cols = ["age_group_lt65", "age_group_65_74", "age_group_75_84", "age_group_85_plus"]
    age_dummies = age_dummies.reindex(columns=expected_age_cols, fill_value=0)

    df["los_log"] = np.log1p(df["length_of_stay"].astype(float))
    high_reimbursement_threshold = df["inpatient_reimbursement"].quantile(0.75)
    df["high_reimbursement"] = (df["inpatient_reimbursement"] > high_reimbursement_threshold).astype(int)

    comorbidity_columns = [
        "sp_chf",
        "sp_ckd",
        "sp_diabetes",
        "sp_ischemic_hd",
        "sp_copd",
        "sp_depression",
        "sp_alzheimer",
        "sp_stroke",
    ]
    df["comorbidity_count"] = df[comorbidity_columns].sum(axis=1).astype(int)
    df["multiple_comorbidities"] = (df["comorbidity_count"] >= 3).astype(int)
    df["ckd_hf_interaction"] = (df["sp_ckd"] * df["sp_chf"]).astype(int)
    df["diabetes_hf_interaction"] = (df["sp_diabetes"] * df["sp_chf"]).astype(int)

    X = pd.DataFrame(
        {
            "age_at_admit": df["age_at_admit"].astype(float),
            "length_of_stay": df["length_of_stay"].astype(float),
            "los_log": df["los_log"].astype(float),
            "high_reimbursement": df["high_reimbursement"].astype(int),
            "prior_ip_admits_6mo": df["prior_ip_admits_6mo"].astype(int),
            "prior_ip_admits_12mo": df["prior_ip_admits_12mo"].astype(int),
            "inpatient_reimbursement": df["inpatient_reimbursement"].astype(float),
            "comorbidity_count": df["comorbidity_count"].astype(int),
            "multiple_comorbidities": df["multiple_comorbidities"].astype(int),
            "ckd_hf_interaction": df["ckd_hf_interaction"].astype(int),
            "diabetes_hf_interaction": df["diabetes_hf_interaction"].astype(int),
        }
    )
    X = pd.concat([X, age_dummies.reset_index(drop=True)], axis=1)

    feature_names = list(X.columns)
    y = df["readmit_30d"].astype(int).reset_index(drop=True)

    logger.info("Built feature matrix", extra={"n_samples": len(X), "n_features": X.shape[1]})
    return X, y, feature_names
