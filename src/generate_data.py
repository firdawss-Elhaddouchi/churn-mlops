"""Génère un jeu de données synthétique de clients télécom (churn)."""
import argparse

import numpy as np
import pandas as pd


def generate(n: int, seed: int, drift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    anciennete = rng.integers(0, 73, n)
    contrat = rng.choice(
        ["Mensuel", "Annuel", "Biennal"], n, p=[0.55, 0.25, 0.20]
    )
    forfait = rng.choice(
        ["Prepaye", "Postpaye", "Pro"], n, p=[0.5, 0.4, 0.1]
    )
    region = rng.choice(["Nord", "Centre", "Sud", "Oriental"], n)

    # drift > 0 : clients qui consomment plus et appellent plus le service client
    charges = rng.normal(220 + 80 * drift, 90, n).clip(50, 900).round(2)
    data_go = rng.gamma(2.0, 6.0 * (1 + drift), n).round(1)
    appels_sc = rng.poisson(1.5 + 1.5 * drift, n)

    logit = (
        -2.0
        - 0.04 * anciennete
        + 0.006 * charges
        + 0.45 * appels_sc
        + np.where(contrat == "Mensuel", 1.2, 0.0)
        - np.where(contrat == "Biennal", 1.0, 0.0)
    )
    churn = (
        rng.random(n) < 1 / (1 + np.exp(-logit))
    ).astype(int)

    df = pd.DataFrame({
        "client_id": [f"C{seed:03d}{i:06d}" for i in range(n)],
        "anciennete_mois": anciennete,
        "contrat": contrat,
        "forfait": forfait,
        "region": region,
        "charges_mensuelles": charges,
        "data_go_mois": data_go,
        "nb_appels_service_client": appels_sc,
        "churn": churn,
    })

    # Données « sales » pour le TP2 : valeurs manquantes, négatives, doublons
    idx = rng.choice(n, int(0.03 * n), replace=False)
    df.loc[idx[: len(idx) // 2], "charges_mensuelles"] = np.nan
    df.loc[idx[len(idx) // 2:], "data_go_mois"] = -1
    df = pd.concat([df, df.sample(int(0.01 * n), random_state=seed)])

    return df.reset_index(drop=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--drift", type=float, default=0.0)
    p.add_argument("--out", default="data/raw/clients.csv")
    a = p.parse_args()

    df = generate(a.n, a.seed, a.drift)
    df.to_csv(a.out, index=False)
    print(f"{len(df)} lignes écrites dans {a.out}")


