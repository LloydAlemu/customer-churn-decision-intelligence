import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/processed/telco_clean.csv")
OUT_DIR = Path("data/processed")

def build_features():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Cleaned data not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # Separate target
    y = df["churn"]
    X = df.drop(columns=["churn", "Churn"])  # drop original text target

    # Identify categorical and numeric columns
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = X.select_dtypes(exclude="object").columns.tolist()

    # One-hot encode categoricals
    X_encoded = pd.get_dummies(
        X,
        columns=cat_cols,
        drop_first=True
    )

    # Train / test split (stratified = important for churn)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # Save outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(OUT_DIR / "X_train.csv", index=False)
    X_test.to_csv(OUT_DIR / "X_test.csv", index=False)
    y_train.to_csv(OUT_DIR / "y_train.csv", index=False)
    y_test.to_csv(OUT_DIR / "y_test.csv", index=False)

    print("✅ Features built successfully")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Positive churn rate (train): {y_train.mean():.3f}")
    print(f"Positive churn rate (test): {y_test.mean():.3f}")

if __name__ == "__main__":
    build_features()
