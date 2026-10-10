"""
Train the Bengaluru house-price model.
Expected dataset: data/Bengaluru_House_Data.csv
Typical Kaggle columns: area_type, availability, location, size, society,
total_sqft, bath, balcony, price
"""
from pathlib import Path
import re
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "Bengaluru_House_Data.csv"
MODEL_PATH = ROOT / "models" / "house_price_pipeline.joblib"
META_PATH = ROOT / "models" / "model_metadata.json"
FEATURES = ["location", "total_sqft", "bhk", "bath", "balcony"]
TARGET = "price"

def parse_sqft(value):
    """Convert values such as '1200', '1000 - 1200', or '34.46Sq. Meter'."""
    if pd.isna(value):
        return np.nan
    s = str(value).strip().lower().replace(",", "")
    # For ranges, use the midpoint.
    nums = re.findall(r"\d+(?:\.\d+)?", s)
    if not nums:
        return np.nan
    vals = [float(x) for x in nums[:2]]
    result = sum(vals) / len(vals)
    if "meter" in s or "sqm" in s:
        result *= 10.7639
    elif "yard" in s:
        result *= 9
    return result

def parse_bhk(value):
    if pd.isna(value):
        return np.nan
    m = re.search(r"\d+", str(value))
    return float(m.group()) if m else np.nan

def load_and_clean(path):
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    required = {"location", "total_sqft", "size", "bath", "price"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing columns: {sorted(missing)}. "
            f"Columns found: {list(df.columns)}"
        )
    df["location"] = df["location"].astype("string").str.strip()
    df["total_sqft"] = df["total_sqft"].apply(parse_sqft)
    df["bhk"] = df["size"].apply(parse_bhk)
    if "balcony" not in df.columns:
        df["balcony"] = np.nan
    for col in ["bath", "balcony", "price"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[FEATURES + [TARGET]].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["location", "total_sqft", "bhk", "price"])
    df = df[(df["total_sqft"] > 100) & (df["total_sqft"] < 50000)]
    df = df[(df["price"] > 0) & (df["price"] < 100000)]
    df = df[(df["bhk"] >= 1) & (df["bhk"] <= 20)]
    df = df[(df["bath"].isna()) | ((df["bath"] >= 1) & (df["bath"] <= 30))]
    df["location"] = df["location"].replace("", np.nan)
    df = df.dropna(subset=["location"])
    # Retain locations with enough examples; rare locations become "Other".
    counts = df["location"].value_counts()
    common = set(counts[counts >= 5].index)
    df["location"] = df["location"].where(df["location"].isin(common), "Other")
    # Broad sanity check: price per square foot, in dataset units (price is lakh).
    ppsf = df["price"] * 100000 / df["total_sqft"]
    df = df[(ppsf > 100) & (ppsf < 100000)]
    return df

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}\n"
            "Download the Bengaluru House Price dataset from Kaggle and place its CSV "
            "at data/Bengaluru_House_Data.csv."
        )
    df = load_and_clean(DATA_PATH)
    if len(df) < 30:
        raise ValueError(f"Only {len(df)} usable rows found; check the dataset.")
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    numeric = ["total_sqft", "bhk", "bath", "balcony"]
    categorical = ["location"]
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ])
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1
        )),
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "rows_used": int(len(df)),
        "mae_lakh": round(float(mean_absolute_error(y_test, pred)), 3),
        "rmse_lakh": round(float(np.sqrt(mean_squared_error(y_test, pred))), 3),
        "r2": round(float(r2_score(y_test, pred)), 4),
        "features": FEATURES,
        "target_unit": "lakh rupees",
        "dataset": DATA_PATH.name,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    # Save location statistics for a transparent, rule-based comparison.
    stats = df.groupby("location").apply(
        lambda g: pd.Series({
            "median_price_per_sqft": float((g["price"] * 100000 / g["total_sqft"]).median()),
            "median_price_lakh": float(g["price"].median()),
            "sample_count": int(len(g)),
        }), include_groups=False
    ).to_dict(orient="index")
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "location_stats": stats,
                   "locations": sorted(df["location"].dropna().unique().tolist())},
                  f, indent=2)
    print("Training complete.")
    print(json.dumps(metrics, indent=2))
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metadata saved to: {META_PATH}")

if __name__ == "__main__":
    main()
