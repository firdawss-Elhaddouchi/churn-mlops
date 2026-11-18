import pandera.pandas as pa


CLIENTS_SCHEMA = pa.DataFrameSchema(
    {
        "client_id": pa.Column(str, unique=True),
        "anciennete_mois": pa.Column(int, checks=pa.Check.ge(0)),
        "contrat": pa.Column(
            str,
            checks=pa.Check.isin(["Annuel", "Biennal", "Mensuel"]),
        ),
        "forfait": pa.Column(
            str,
            checks=pa.Check.isin(["Postpaye", "Prepaye", "Pro"]),
        ),
        "region": pa.Column(
            str,
            checks=pa.Check.isin(["Centre", "Nord", "Oriental", "Sud"]),
        ),
        "charges_mensuelles": pa.Column(float, checks=pa.Check.gt(0)),
        "data_go_mois": pa.Column(float, checks=pa.Check.ge(0)),
        "nb_appels_service_client": pa.Column(
            int,
            checks=pa.Check.ge(0),
        ),
        "churn": pa.Column(int, checks=pa.Check.isin([0, 1])),
    },
    strict=True,
    coerce=True,
)


def validate_clients(df):
    """Validate a clients DataFrame and return it if it is valid."""
    return CLIENTS_SCHEMA.validate(df, lazy=True)
