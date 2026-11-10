
# Churn MLOps — Data Versioning with Git and DVC

This project is part of a practical assignment on DataOps and MLOps. It demonstrates how to generate, version, store, restore, and compare datasets using Git and DVC.

The current application focuses on Git and DVC. Airflow orchestration is planned for a later part of the project.

## Project objectives

- Create a reproducible project structure.
- Generate synthetic telecom customer data.
- Track code and project metadata with Git.
- Track dataset versions with DVC.
- Store DVC data in a separate remote location.
- Restore and compare dataset versions.
- Retrieve data from a fresh repository clone.

## Tools

- Python
- Git and GitHub
- DVC
- Ubuntu on WSL2

## Project structure

```text
churn-mlops/
├── data/
│   └── raw/
│       ├── .gitignore
│       └── clients.csv.dvc
├── src/
│   └── generate_data.py
├── .dvc/
├── .dvcignore
├── .gitignore
├── README.md
└── requirements.txt
```

The generated `clients.csv` file is excluded from Git and tracked by DVC. The `.dvc` file stores metadata that lets DVC identify and retrieve the corresponding data.

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

## Generate the dataset

Generate the first dataset version:

```bash
python src/generate_data.py --n 5000 --seed 42
```

Generate a second version:

```bash
python src/generate_data.py --n 7000 --seed 42
```

The generator creates synthetic telecom customer data. It also introduces duplicate rows, so the CSV line count can be greater than the value passed to `--n`.

## Track data with DVC

Initialize DVC and add the generated dataset:

```bash
dvc init
dvc add data/raw/clients.csv
```

Commit the DVC metadata and the code to Git:

```bash
git add .dvc .dvcignore data/raw/.gitignore data/raw/clients.csv.dvc src/generate_data.py
git commit -m "Track initial dataset with DVC"
git tag data-v1
```

## Configure the local DVC remote

This assignment uses a local folder to simulate a remote storage location:

```bash
dvc remote add -d storage ../dvc-storage
dvc push
```

This remote is a local directory, not a cloud service. The DVC configuration is committed to Git, while the actual dataset content is stored separately by DVC.

## Create and compare a new version

After generating the second dataset, track and push it:

```bash
dvc add data/raw/clients.csv
dvc push
git add data/raw/clients.csv.dvc
git commit -m "Track data v2 with DVC"
git tag data-v2
git push origin main --tags
```

Compare the two dataset versions:

```bash
dvc diff data-v1 data-v2
```

## Restore an earlier version

Check out the first version and restore its data:

```bash
git switch --detach data-v1
dvc checkout
```

Return to the main branch and restore its data:

```bash
git switch main
dvc checkout
```

Git restores the versioned project files. `dvc checkout` restores the dataset version referenced by the current Git revision.

## Retrieve data after cloning

A regular Git clone contains the DVC metadata, but not the ignored dataset file. Retrieve the data with:

```bash
dvc pull
```

If the local remote path is different in the cloned copy, configure its location locally:

```bash
dvc remote modify --local storage url /path/to/dvc-storage
dvc pull
```

The `--local` option keeps this machine-specific path out of the shared Git configuration.

## Notes

- Do not commit `clients.csv` directly to Git. DVC stores the data separately and Git tracks the small `.dvc` metadata file.
- The local DVC remote is suitable for this exercise on one machine. A shared cloud or network remote is needed for collaboration across different computers.
- Airflow is outside the scope of the current application and will be covered later.

## Author

**Firdawss Elhaddouchi**  
Data Engineering (ID3)  
École Nationale des Sciences Appliquées d'Al Hoceima
