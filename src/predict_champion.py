import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient


mlflow.set_tracking_uri("sqlite:///mlflow.db")

client = MlflowClient()
champion = client.get_model_version_by_alias("churn-model", "champion")
model = mlflow.sklearn.load_model("models:/churn-model@champion")

data = pd.read_csv("data/processed/clients_prepared.csv")
X = data.drop(columns=["churn", "client_id"]).head(3)
probabilities = model.predict_proba(X)[:, 1]

print(f"Champion version: {champion.version}")
for client_id, probability in zip(data["client_id"].head(3), probabilities):
    print(f"Client {client_id}: churn probability = {probability:.2%}")
