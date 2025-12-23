import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/telco_churn.csv")
OUT_DIR = Path("data/processed")
OUT_PATH = OUT_DIR / "telco_clean.csv"

def clean_telco(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1) Standardize whitespace in string columns
    obj_cols = df.select_dtypes(include="object").columns
    for c in obj_cols:
        df[c] = df[c].astype(str).str.strip()

    # 2) Fix TotalCharges: it's sometimes stored as text with blanks
    # Convert to numeric; blanks become NaN
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # 3) Create numeric target: churn (1=Yes, 0=No)
    df["churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # 4) Drop columns we don't want for modeling
    # customerID is an identifier (leaks nothing useful)
    df = df.drop(columns=["customerID"])

    # 5) Handle missing TotalCharges (usually tenure=0 customers)
    # Simple, defensible fix: fill with 0.0 (no charges yet)
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # 6) Sanity checks
    if df["churn"].isna().any():
        raise ValueError("Found unexpected values in Churn column. Expected only Yes/No.")
    if df.isna().any().any():
        # After our cleaning, there should be no missing values
        missing = df.isna().sum()[df.isna().sum() > 0]
        raise ValueError(f"Missing values still present:\n{missing}")

    return df

def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw data file not found: {RAW_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(RAW_PATH)
    df_clean = clean_telco(df_raw)

    df_clean.to_csv(OUT_PATH, index=False)

    print("✅ Saved cleaned dataset:")
    print(f"Path: {OUT_PATH}")
    print(f"Shape: {df_clean.shape}")
    print("\nTarget distribution (churn):")
    print(df_clean["churn"].value_counts(normalize=True))

    print("\nDtypes (check TotalCharges is numeric):")
    print(df_clean.dtypes[["MonthlyCharges", "TotalCharges", "tenure", "churn"]])

if __name__ == "__main__":
    main()
