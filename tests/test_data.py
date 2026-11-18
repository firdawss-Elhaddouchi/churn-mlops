import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from src.pipeline import clean_clients
from src.schema import validate_clients


DATA_PATH = "data/raw/clients.csv"


def load_raw_data():
    return pd.read_csv(DATA_PATH)


def test_raw_data_violates_contract():
    raw = load_raw_data()

    with pytest.raises(SchemaErrors):
        validate_clients(raw)


def test_cleaned_data_respects_contract():
    raw = load_raw_data()
    cleaned = clean_clients(raw)

    validate_clients(cleaned)


def test_churn_rate_is_between_10_and_60_percent():
    raw = load_raw_data()
    cleaned = clean_clients(raw)

    churn_rate = cleaned["churn"].mean()

    assert 0.10 <= churn_rate <= 0.60
