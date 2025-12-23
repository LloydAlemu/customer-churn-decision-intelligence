import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path("data/processed")
MODEL_PATH = Path("reports/models/logreg_churn.pkl")
FIG_DIR = Path("reports/figures")

def main():
    # Load training data
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")

    # Load trained model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    # Create SHAP explainer (correct argument for current SHAP)
    explainer = shap.LinearExplainer(
        model,
        X_train,
        feature_perturbation="interventional"
    )

    shap_values = explainer.shap_values(X_train)

    # Ensure output directory exists
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Global feature importance plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_train,
        plot_type="bar",
        show=False
    )
    plt.title("Global Feature Importance (SHAP)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_feature_importance.png", dpi=150)
    plt.close()

    print("✅ SHAP explanation complete")
    print(f"Saved plot to: {FIG_DIR / 'shap_feature_importance.png'}")

if __name__ == "__main__":
    main()
