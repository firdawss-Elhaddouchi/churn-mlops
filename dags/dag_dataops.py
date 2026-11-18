from datetime import timedelta
from pathlib import Path
import shutil
import sys

import pandas as pd
import pendulum
from airflow.sdk import dag, task

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import clean_clients
from src.schema import validate_clients


@dag(
    schedule="@daily",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    default_args={
        "retries": 2,
        "retry_delay": timedelta(minutes=1),
    },
    tags=["dataops"],
)
def dataops_preparation():
    @task
    def extract() -> str:
        """Return the path to the raw dataset."""
        raw_path = PROJECT_ROOT / "data" / "raw" / "clients.csv"
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw dataset not found: {raw_path}")
        print(f"Extracted source path: {raw_path}")
        return str(raw_path)

    @task
    def clean_data(input_path: str) -> str:
        """Clean the dataset and return the intermediate file path."""
        raw = pd.read_csv(input_path)
        cleaned = clean_clients(raw)

        output_path = PROJECT_ROOT / "data" / "interim" / "clients_clean.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned.to_csv(output_path, index=False)

        print(f"Cleaned dataset saved to: {output_path}")
        return str(output_path)

    @task
    def validate_data(input_path: str) -> str:
        """Validate the cleaned file and return its path if valid."""
        df = pd.read_csv(input_path)
        validate_clients(df)

        churn_rate = df["churn"].mean()
        if not 0.10 <= churn_rate <= 0.60:
            raise ValueError(f"Churn rate outside the required range: {churn_rate:.2%}")

        print(f"Schema valid. Churn rate: {churn_rate:.2%}")
        return input_path

    @task
    def publish(input_path: str) -> str:
        """Copy the validated dataset to the processed folder."""
        output_path = PROJECT_ROOT / "data" / "processed" / "clients.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_path, output_path)

        print(f"Published dataset to: {output_path}")
        return str(output_path)

    raw_path = extract()
    cleaned_path = clean_data(raw_path)
    validated_path = validate_data(cleaned_path)
    publish(validated_path)


dataops_preparation()
