from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient


TRACKING_URI = "sqlite:///mlflow.db"
MODEL_NAME = "churn-model"


def promouvoir_meilleur():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)

    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    candidates = []

    for version in versions:
        run = client.get_run(version.run_id)

        if run.info.status == "FINISHED" and "auc" in run.data.metrics:
            candidates.append(
                (run.data.metrics["auc"], int(version.version))
            )

    if not candidates:
        raise RuntimeError("Aucune version enregistrée avec une métrique AUC.")

    best_auc, best_version = max(candidates)
    client.set_registered_model_alias(
        MODEL_NAME,
        "champion",
        best_version,
    )

    print(f"Champion: version {best_version}, AUC={best_auc:.4f}")
    return best_version


def exporter_champion():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)

    champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
    model_uri = f"models:/{MODEL_NAME}@champion"
    model = mlflow.sklearn.load_model(model_uri)

    output_path = Path("model/model.joblib")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)

    print(f"Modèle exporté: {output_path}")
    print(f"Version exportée: {champion.version}")
    return str(output_path)


if __name__ == "__main__":
    promouvoir_meilleur()
    exporter_champion()
