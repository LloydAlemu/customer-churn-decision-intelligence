import pandas as pd
from pathlib import Path
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("reports/models")
MODEL_PATH = MODEL_DIR / "logreg_churn.pkl"

def load_splits():
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_test  = pd.read_csv(DATA_DIR / "X_test.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze("columns")
    y_test  = pd.read_csv(DATA_DIR / "y_test.csv").squeeze("columns")
    return X_train, X_test, y_train, y_test

def train_logreg(X_train, y_train):
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        n_jobs=None
    )
    model.fit(X_train, y_train)
    return model

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print("✅ Evaluation (Logistic Regression)")
    print(f"Accuracy: {acc:.3f}")
    print(f"ROC-AUC:  {auc:.3f}")
    print("\nConfusion Matrix [ [TN FP], [FN TP] ]:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

def main():
    X_train, X_test, y_train, y_test = load_splits()
    model = train_logreg(X_train, y_train)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"✅ Model saved to: {MODEL_PATH}")
    evaluate(model, X_test, y_test)

if __name__ == "__main__":
    main()
