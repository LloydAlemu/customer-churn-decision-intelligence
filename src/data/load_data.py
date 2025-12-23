import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/telco_churn.csv")

def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    return df


def data_profile(df: pd.DataFrame):
    print("=== BASIC INFO ===")
    print(f"Shape: {df.shape}")
    print("\nColumns:")
    print(df.columns.tolist())

    print("\n=== MISSING VALUES ===")
    print(df.isna().sum())

    print("\n=== TARGET DISTRIBUTION (Churn) ===")
    if "Churn" in df.columns:
        print(df["Churn"].value_counts(normalize=True))
    else:
        print("⚠️ 'Churn' column not found")


if __name__ == "__main__":
    df = load_data()
    data_profile(df)
