"""Tests for the heart failure data and model pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from hf_readmit.data.cohort import build_hf_cohort
from hf_readmit.data.features import build_features
from hf_readmit.models.predictor import ReadmissionPredictor
from hf_readmit.models.run_training import main as run_training_pipeline

PROJECT_ROOT = Path(__file__).parent.parent


def model_artifact_path() -> Path:
    return PROJECT_ROOT / "models" / "hf_readmit_model.pkl"


@pytest.fixture(scope="session")
def trained_model_path() -> Path:
    path = model_artifact_path()
    if not path.exists():
        run_training_pipeline()
    assert path.exists()
    return path


def test_build_hf_cohort_returns_dataframe() -> None:
    inpatient_path = PROJECT_ROOT / "data" / "raw" / "DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv"
    beneficiary_path = PROJECT_ROOT / "data" / "raw" / "DE1_0_2009_Beneficiary_Summary_File_Sample_1.csv"

    cohort = build_hf_cohort(inpatient_path, beneficiary_path)

    expected_columns = [
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
    assert list(cohort.columns) == expected_columns
    assert cohort["readmit_30d"].dropna().isin([0, 1]).all()

    beneficiary = pd.read_csv(
        beneficiary_path,
        dtype={"DESYNPUF_ID": str},
        usecols=["DESYNPUF_ID", "BENE_DEATH_DT"],
    )
    beneficiary["BENE_DEATH_DT"] = pd.to_datetime(
        beneficiary["BENE_DEATH_DT"], format="%Y%m%d", errors="coerce"
    )
    merged = cohort.merge(
        beneficiary,
        left_on="patient_id",
        right_on="DESYNPUF_ID",
        how="left",
    )
    assert not ((merged["BENE_DEATH_DT"] <= merged["index_discharge_date"] + pd.Timedelta(days=30)).fillna(False)).any()


def test_label_rate_reasonable() -> None:
    inpatient_path = PROJECT_ROOT / "data" / "raw" / "DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv"
    beneficiary_path = PROJECT_ROOT / "data" / "raw" / "DE1_0_2009_Beneficiary_Summary_File_Sample_1.csv"
    cohort = build_hf_cohort(inpatient_path, beneficiary_path)
    rate = cohort["readmit_30d"].mean()
    assert 0.05 <= rate <= 0.40


def test_build_features_shape() -> None:
    inpatient_path = PROJECT_ROOT / "data" / "raw" / "DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv"
    beneficiary_path = PROJECT_ROOT / "data" / "raw" / "DE1_0_2009_Beneficiary_Summary_File_Sample_1.csv"
    cohort = build_hf_cohort(inpatient_path, beneficiary_path)
    X, y, feature_names = build_features(cohort)

    assert X.shape[0] == len(cohort)
    assert y.shape[0] == len(cohort)
    assert len(feature_names) == X.shape[1]
    assert "los_log" in X.columns
    assert "age_group_lt65" in X.columns
    assert "ckd_hf_interaction" in X.columns


def test_predictor_output_schema(trained_model_path: Path) -> None:
    predictor = ReadmissionPredictor(trained_model_path)
    inpatient_path = PROJECT_ROOT / "data" / "raw" / "DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv"
    beneficiary_path = PROJECT_ROOT / "data" / "raw" / "DE1_0_2009_Beneficiary_Summary_File_Sample_1.csv"
    cohort = build_hf_cohort(inpatient_path, beneficiary_path)
    X, _, _ = build_features(cohort)
    patient_features = X.iloc[0].to_dict()
    patient_features["patient_id"] = cohort.iloc[0]["patient_id"]

    result = predictor.predict(patient_features)
    assert result.patient_id == patient_features["patient_id"]
    assert 0.0 <= result.probability <= 1.0
    assert result.risk_category in {"low", "medium", "high"}
    assert isinstance(result.top_drivers, list)
    assert isinstance(result.model_version, str)
