"""Model serving stub for readmission risk predictions."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from hf_readmit.models.explain import get_patient_shap


class PredictionResult(BaseModel):
    patient_id: str = Field(...)
    probability: float = Field(..., ge=0.0, le=1.0)
    risk_category: str = Field(...)
    top_drivers: list[dict] = Field(default_factory=list)
    model_version: str = Field(...)


class ReadmissionPredictor:
    def __init__(self, model_path: Path | str):
        self.model_path = Path(model_path)
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)

        shap_path = self.model_path.parent / "shap_output.pkl"
        self.shap_output = None
        if shap_path.exists():
            with open(shap_path, "rb") as f:
                self.shap_output = pickle.load(f)

        self.model_version = self.model_path.stem

    def predict(self, patient_features: dict) -> PredictionResult:
        if "patient_id" not in patient_features:
            raise ValueError("patient_features must include patient_id")

        features = dict(patient_features)
        patient_id = str(features.pop("patient_id"))
        X = pd.DataFrame([features])

        proba = float(self.model.predict_proba(X)[0, 1])
        if proba < 0.2:
            risk_category = "low"
        elif proba <= 0.5:
            risk_category = "medium"
        else:
            risk_category = "high"

        top_drivers: list[dict] = []
        if self.shap_output is not None:
            try:
                top_drivers = get_patient_shap(self.shap_output, 0)
            except Exception:
                top_drivers = []

        result = PredictionResult(
            patient_id=patient_id,
            probability=proba,
            risk_category=risk_category,
            top_drivers=top_drivers,
            model_version=self.model_version,
        )
        return result
