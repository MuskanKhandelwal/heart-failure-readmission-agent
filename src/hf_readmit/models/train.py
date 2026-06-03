"""Model training pipeline for the heart failure readmission model."""

from __future__ import annotations

import pickle
from pathlib import Path

import mlflow
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import matplotlib.pyplot as plt

from hf_readmit.utils.logging import setup_logging

logger = setup_logging()


def train_model(X: "pd.DataFrame", y: "pd.Series", experiment_name: str) -> dict:
    """Train an XGBoost readmission model and calibrate it."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    scale_pos_weight = (len(y_train) - int(y_train.sum())) / max(int(y_train.sum()), 1)

    base_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="auc",
    )
    base_model.fit(X_train, y_train)

    calibrated_model = CalibratedClassifierCV(
        estimator=base_model,
        method="isotonic",
        cv=3,
    )
    calibrated_model.fit(X_train, y_train)

    probabilities = calibrated_model.predict_proba(X_test)[:, 1]
    auroc = roc_auc_score(y_test, probabilities)
    auprc = average_precision_score(y_test, probabilities)
    brier = brier_score_loss(y_test, probabilities)

    project_root = Path(__file__).resolve().parents[3]
    model_dir = project_root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "hf_readmit_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(calibrated_model, f)

    calibration_plot_path = model_dir / "calibration_plot.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    CalibrationDisplay.from_estimator(calibrated_model, X_test, y_test, ax=ax, n_bins=10)
    ax.set_title("Calibration plot")
    fig.tight_layout()
    fig.savefig(calibration_plot_path)
    plt.close(fig)

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=experiment_name):
        mlflow.log_params(
            {
                "n_estimators": 200,
                "max_depth": 4,
                "learning_rate": 0.05,
                "scale_pos_weight": float(scale_pos_weight),
                "random_state": 42,
                "calibration_method": "isotonic",
            }
        )
        mlflow.log_metric("auroc", float(auroc))
        mlflow.log_metric("auprc", float(auprc))
        mlflow.log_metric("brier_score", float(brier))
        mlflow.log_artifact(str(calibration_plot_path), artifact_path="model_plots")
        mlflow.log_artifact(str(model_path), artifact_path="model_artifacts")

    metrics = {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "brier_score": float(brier),
        "positive_label_rate": float(y.mean()),
    }

    logger.info("Model trained", extra={"metrics": metrics})
    return {
        "model": calibrated_model,
        "X_test": X_test,
        "y_test": y_test,
        "feature_names": list(X.columns),
        "metrics": metrics,
        "model_path": model_path,
        "calibration_plot_path": calibration_plot_path,
    }
