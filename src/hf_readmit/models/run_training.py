"""End-to-end model training runner for the heart failure readmission pipeline."""

from __future__ import annotations

from pathlib import Path

from hf_readmit.data.cohort import build_hf_cohort
from hf_readmit.data.features import build_features
from hf_readmit.models.explain import compute_shap_values
from hf_readmit.models.train import train_model
from hf_readmit.utils.logging import setup_logging

logger = setup_logging()


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    inpatient_path = project_root / "data" / "raw" / "DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv"
    beneficiary_path = project_root / "data" / "raw" / "DE1_0_2009_Beneficiary_Summary_File_Sample_1.csv"

    cohort = build_hf_cohort(inpatient_path, beneficiary_path)
    X, y, feature_names = build_features(cohort)

    experiment_name = "hf_readmit_training"
    trained = train_model(X, y, experiment_name=experiment_name)

    shap_output = compute_shap_values(trained["model"], X, feature_names)

    model_dir = project_root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    shap_path = model_dir / "shap_output.pkl"
    with open(shap_path, "wb") as f:
        import pickle

        pickle.dump(shap_output, f)

    metrics = trained["metrics"]
    print("Training completed")
    print("Metrics:")
    for name, value in metrics.items():
        print(f"- {name}: {value:.4f}")

    logger.info("Training pipeline complete", extra={"metrics": metrics})


if __name__ == "__main__":
    main()
