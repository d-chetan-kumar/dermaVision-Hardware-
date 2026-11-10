"""
import_kaggle_results.py
================================================================================
DermaVision Revision V1: Ingest Results from Kaggle Run
--------------------------------------------------------------------------------
Extracts and moves trained checkpoints, training histories, figures, and
metrics from 'dermavision_revision_v1_results.zip' into the project repository.
================================================================================
"""

import os
import shutil
import zipfile
from pathlib import Path

def main():
    zip_path = Path("dermavision_revision_v1_results.zip")
    if not zip_path.exists():
        # Check Downloads folder as convenience
        user_dl = Path.home() / "Downloads" / "dermavision_revision_v1_results.zip"
        if user_dl.exists():
            print(f"Found archive in Downloads: {user_dl}")
            shutil.copy(user_dl, zip_path)
        else:
            print(f"Archive '{zip_path.name}' not found.")
            print("Please place the downloaded 'dermavision_revision_v1_results.zip' in this directory.")
            return

    extract_dir = Path("temp_kaggle_results")
    extract_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_dir)

    models_dir = Path("models/revision_v1")
    results_dir = Path("results/revision_v1")
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    for f in extract_dir.glob("*"):
        if f.suffix in [".keras", ".pth", ".h5", ".pt"]:
            dest = models_dir / f.name
            shutil.copy2(f, dest)
            copied.append(f"Model: {dest}")
        elif f.suffix in [".csv", ".json", ".png", ".txt", ".npy"]:
            dest = results_dir / f.name
            shutil.copy2(f, dest)
            copied.append(f"Artifact: {dest}")

    shutil.rmtree(extract_dir, ignore_errors=True)
    print("="*60)
    print("SUCCESS: Ingested Kaggle Results into Local Repository:")
    for item in copied:
        print(f"  + {item}")
    print("="*60)

if __name__ == "__main__":
    main()

