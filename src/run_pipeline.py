from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import clean_clients
from src.schema import validate_clients


input_path = PROJECT_ROOT / "data" / "raw" / "clients.csv"
output_path = PROJECT_ROOT / "data" / "processed" / "clients_prepared.csv"

raw = pd.read_csv(input_path)
cleaned = clean_clients(raw)
validate_clients(cleaned)

churn_rate = cleaned["churn"].mean()
if not 0.10 <= churn_rate <= 0.60:
    raise ValueError(f"Churn rate outside the required range: {churn_rate:.2%}")

output_path.parent.mkdir(parents=True, exist_ok=True)
cleaned.to_csv(output_path, index=False)

print(f"Prepared {len(cleaned)} rows.")
print(f"Churn rate: {churn_rate:.2%}")
print(f"Output: {output_path}")
