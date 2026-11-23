import os
import sys
from pathlib import Path

import pendulum
from airflow.sdk import dag, task


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_PATH = PROJECT_ROOT / "data/processed/clients_prepared.csv"

MODEL_CONFIGS = [
    {
        "model": "logistic_regression",
        "C": 0.1,
        "n_estimators": 200,
        "max_depth": 6,
    },
    {
        "model": "logistic_regression",
        "C": 1.0,
        "n_estimators": 200,
        "max_depth": 6,
    },
    {
        "model": "random_forest",
        "C": 1.0,
        "n_estimators": 200,
        "max_depth": 6,
    },
    {
        "model": "random_forest",
        "C": 1.0,
        "n_estimators": 300,
        "max_depth": None,
    },
]


@dag(
    dag_id="mlops_entrainement",
    description="Entraînement parallèle et promotion du meilleur modèle churn",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    schedule=None,
    catchup=False,
    tags=["mlops", "mlflow", "churn"],
)
def mlops_entrainement():

    @task
    def verifier_donnees():
        os.chdir(PROJECT_ROOT)
        import pandas as pd
        from src.schema import validate_clients

        data = pd.read_csv(DATA_PATH)
        validate_clients(data)
        print(f"Données validées : {len(data)} lignes.")
        return str(DATA_PATH)

    @task
    def entrainer_modele(config):
        os.chdir(PROJECT_ROOT)
        import pandas as pd
        from src.train import entrainer

        data = pd.read_csv(DATA_PATH)
        metrics = entrainer(data=data, **config)
        return {
            "model": config["model"],
            "auc": float(metrics["auc"]),
        }

    @task
    def porte_qualite(resultats):
        meilleur_auc = max(resultat["auc"] for resultat in resultats)
        print(f"Meilleure AUC : {meilleur_auc:.4f}")
        if meilleur_auc < 0.75:
            raise ValueError(
                f"La meilleure AUC ({meilleur_auc:.4f}) est inférieure à 0.75."
            )
        return meilleur_auc

    @task
    def promouvoir():
        os.chdir(PROJECT_ROOT)
        from src.promote import promouvoir_meilleur

        return promouvoir_meilleur()

    @task
    def exporter():
        os.chdir(PROJECT_ROOT)
        from src.promote import exporter_champion

        return exporter_champion()

    donnees_ok = verifier_donnees()
    resultats = entrainer_modele.expand(config=MODEL_CONFIGS)
    donnees_ok >> resultats

    qualite_ok = porte_qualite(resultats)
    promotion = promouvoir()
    export = exporter()

    qualite_ok >> promotion >> export


mlops_entrainement()

