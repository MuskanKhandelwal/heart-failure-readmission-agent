"""SHAP explanation utilities for the readmission model."""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import shap

from hf_readmit.utils.logging import setup_logging

logger = setup_logging()


def compute_shap_values(model, X, feature_names: list[str]) -> dict:
    """Compute SHAP values using the underlying XGBoost model.

    Args:
        model: CalibratedClassifierCV wrapper around an XGBoost model.
        X: Feature matrix for which to compute SHAP values.
        feature_names: List of feature names.

    Returns:
        Dict containing shap_values, expected_value, and feature_names.
    """
    base_model = getattr(model, "base_estimator", None) or getattr(model, "estimator", None)
    if base_model is None:
        raise ValueError("Calibrated model does not expose base_estimator or estimator for SHAP")

    explainer = shap.TreeExplainer(base_model)
    shap_values = explainer(X)
    values = shap_values.values
    expected_value = shap_values.base_values

    project_root = Path(__file__).resolve().parents[3]
    model_dir = project_root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    plot_path = model_dir / "shap_summary.png"

    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)

    logger.info("Computed SHAP values", extra={"n_samples": len(X), "n_features": len(feature_names)})
    return {
        "shap_values": values,
        "expected_value": expected_value,
        "feature_names": feature_names,
    }


def get_patient_shap(shap_output: dict, patient_idx: int) -> list[dict]:
    """Get the top 5 SHAP drivers for a single patient."""
    values = shap_output["shap_values"]
    names = shap_output["feature_names"]

    if patient_idx < 0 or patient_idx >= values.shape[0]:
        raise IndexError("patient_idx is out of range for SHAP output")

    patient_values = values[patient_idx]
    indices = sorted(range(len(patient_values)), key=lambda i: abs(patient_values[i]), reverse=True)[:5]

    drivers = []
    for idx in indices:
        val = float(patient_values[idx])
        drivers.append(
            {
                "feature": names[idx],
                "shap_value": val,
                "direction": "increases_risk" if val > 0 else "decreases_risk",
            }
        )
    return drivers
