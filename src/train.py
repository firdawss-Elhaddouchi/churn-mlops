import argparse

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def entrainer(data, model, C=1.0, n_estimators=200, max_depth=6):
    """Entraîne un modèle churn et l'enregistre dans MLflow."""
    if isinstance(data, str):
        data = pd.read_csv(data)

    features = data.drop(columns=["churn", "client_id"])
    labels = data["churn"]

    categorical_columns = features.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()
    numerical_columns = features.select_dtypes(
        include=["number"]
    ).columns.tolist()

    preprocessing = ColumnTransformer(
        transformers=[
            ("numerical", StandardScaler(), numerical_columns),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns,
            ),
        ]
    )

    if model == "logistic_regression":
        classifier = LogisticRegression(
            C=C,
            max_iter=1000,
            random_state=42,
        )
    elif model == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1,
        )
    else:
        raise ValueError(
            "model doit être 'logistic_regression' ou 'random_forest'."
        )

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("classifier", classifier),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.20,
        stratify=labels,
        random_state=42,
    )

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("churn-telecom")

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "model": model,
                "C": C,
                "n_estimators": n_estimators,
                "max_depth": str(max_depth),
                "test_size": 0.20,
                "random_state": 42,
            }
        )

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "auc": roc_auc_score(y_test, probabilities),
            "f1": f1_score(y_test, predictions),
            "accuracy": accuracy_score(y_test, predictions),
        }
        mlflow.log_metrics(metrics)

        signature = infer_signature(X_test, pipeline.predict(X_test))

        model_options = {}
        if model == "random_forest":
            model_options["skops_trusted_types"] = [
                "sklearn.tree._tree.Tree"
            ]

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            registered_model_name="churn-model",
            signature=signature,
            input_example=X_test.head(5),
            **model_options,
        )

        print(f"Run ID: {run.info.run_id}")
        print(f"Modèle: {model}")
        print(f"Métriques: {metrics}")
        return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["logistic_regression", "random_forest"],
        default="logistic_regression",
    )
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=str, default="6")
    parser.add_argument(
        "--data",
        default="data/processed/clients_prepared.csv",
    )
    args = parser.parse_args()

    max_depth = (
        None if args.max_depth.lower() == "none" else int(args.max_depth)
    )

    entrainer(
        data=args.data,
        model=args.model,
        C=args.C,
        n_estimators=args.n_estimators,
        max_depth=max_depth,
    )
