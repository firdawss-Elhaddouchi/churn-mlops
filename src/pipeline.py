import pandas as pd


MEASUREMENT_COLUMNS = [
    "anciennete_mois",
    "charges_mensuelles",
    "data_go_mois",
    "nb_appels_service_client",
]


def clean_clients(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy without modifying the input DataFrame."""
    cleaned = df.copy()

    # Keep one row for each client.
    cleaned = cleaned.drop_duplicates(subset="client_id", keep="first")

    # Replace negative measurements with missing values.
    for column in MEASUREMENT_COLUMNS:
        cleaned.loc[cleaned[column] < 0, column] = float("nan")

    # Fill missing measurements with each column's median.
    for column in MEASUREMENT_COLUMNS:
        if cleaned[column].isna().any():
            cleaned[column] = cleaned[column].fillna(cleaned[column].median())

    return cleaned.reset_index(drop=True)
