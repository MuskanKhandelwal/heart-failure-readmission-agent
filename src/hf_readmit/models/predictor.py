"""Model serving for readmission risk predictions.

SHAP explanations are computed for the *specific patient being scored* rather
than reusing a precomputed training row. The model is a
``CalibratedClassifierCV`` wrapping one fitted XGBoost estimator per CV fold; we
build a ``shap.TreeExplainer`` per fold's base estimator (cached) and average the
per-fold attributions for the patient's feature vector.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

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

        # The column order the model was trained on, used to align inference rows.
        self.feature_names = list(getattr(self.model, "feature_names_in_", []))
        # SHAP explainers are built lazily on first prediction and cached.
        self._explainers: list | None = None
        self.model_version = self.model_path.stem

    def _base_estimators(self) -> list:
        """Return the fitted tree estimators underlying the calibrated model.

        For ``CalibratedClassifierCV`` (cv > 1) there is one fitted base estimator
        per fold under ``calibrated_classifiers_``. Falls back to the model's own
        ``estimator``/``base_estimator`` (prefit) or the model itself.
        """
        estimators: list = []
        for calibrated in getattr(self.model, "calibrated_classifiers_", []) or []:
            estimator = getattr(calibrated, "estimator", None) or getattr(calibrated, "base_estimator", None)
            if estimator is not None:
                estimators.append(estimator)
        if not estimators:
            fallback = (
                getattr(self.model, "estimator", None)
                or getattr(self.model, "base_estimator", None)
                or self.model
            )
            estimators = [fallback]
        return estimators

    def _get_explainers(self) -> list:
        """Build and cache a TreeExplainer for each fold base estimator."""
        if self._explainers is None:
            import shap

            self._explainers = [shap.TreeExplainer(est) for est in self._base_estimators()]
        return self._explainers

    def _patient_top_drivers(self, X: pd.DataFrame) -> list[dict]:
        """Compute this patient's top-5 SHAP drivers from their feature vector.

        Args:
            X: Single-row feature frame already aligned to ``feature_names``.

        Returns:
            Top-5 SHAP drivers for the patient, or an empty list on failure.
        """
        try:
            fold_values = []
            for explainer in self._get_explainers():
                explanation = explainer(X)
                values = np.asarray(explanation.values)
                # Binary classifiers may return (1, n_features) or (1, n_features, 2).
                if values.ndim == 3:
                    values = values[..., 1]
                fold_values.append(values)
            # Average attributions across CV-fold explainers -> shape (1, n_features).
            mean_values = np.mean(fold_values, axis=0)
            shap_output = {
                "shap_values": mean_values,
                "feature_names": self.feature_names or list(X.columns),
            }
            return get_patient_shap(shap_output, 0)
        except Exception:
            return []

    def predict(self, patient_features: dict) -> PredictionResult:
        if "patient_id" not in patient_features:
            raise ValueError("patient_features must include patient_id")

        features = dict(patient_features)
        patient_id = str(features.pop("patient_id"))
        X = pd.DataFrame([features])
        if self.feature_names:
            # Align columns to the model's training order for both scoring and SHAP.
            X = X.reindex(columns=self.feature_names)

        proba = float(self.model.predict_proba(X)[0, 1])
        if proba < 0.2:
            risk_category = "low"
        elif proba <= 0.5:
            risk_category = "medium"
        else:
            risk_category = "high"

        top_drivers = self._patient_top_drivers(X)

        return PredictionResult(
            patient_id=patient_id,
            probability=proba,
            risk_category=risk_category,
            top_drivers=top_drivers,
            model_version=self.model_version,
        )
