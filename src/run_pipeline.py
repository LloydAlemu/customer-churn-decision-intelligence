"""
Runs the full pipeline end-to-end:
1) Clean raw data -> data/processed/telco_clean.csv
2) Build features/splits -> data/processed/X_train.csv, X_test.csv, y_train.csv, y_test.csv
3) Train model -> reports/models/logreg_churn.pkl
4) Generate SHAP plot -> reports/figures/shap_feature_importance.png
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def run(cmd: list[str]):
    print(f"\n▶ Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)

    # Print output to help debugging in cloud
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")

def main():
    # Run each step (reuses your existing scripts)
    run([sys.executable, "src/data/clean_data.py"])
    run([sys.executable, "src/features/build_features.py"])
    run([sys.executable, "src/models/train_model.py"])
    run([sys.executable, "src/models/explain_shap.py"])

    print("\n✅ Pipeline completed successfully.")

if __name__ == "__main__":
    main()
