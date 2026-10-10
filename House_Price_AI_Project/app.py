from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "house_price_pipeline.joblib"
META_PATH = ROOT / "models" / "model_metadata.json"
DATA_PATH = ROOT / "data" / "Bengaluru_House_Data.csv"

st.set_page_config(page_title="AI House Price & Investment Assistant",
                   page_icon="🏠", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metadata():
    with open(META_PATH, encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_dataset():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame()

st.title("🏠 House Price Prediction & Real Estate Investment Assistant")
st.caption("AI-assisted estimates using the Kaggle Bengaluru House Price dataset")
st.info("Educational project: predictions and recommendations are estimates, not financial advice. "
        "The supplied dataset is historical and does not by itself forecast future appreciation or rent.")

if not MODEL_PATH.exists() or not META_PATH.exists():
    st.error("Trained model not found. First run: `python train_model.py`")
    st.stop()

model = load_model()
meta = load_metadata()
locations = meta.get("locations", [])
if not locations:
    st.error("No locations found in model metadata. Retrain using the expected dataset.")
    st.stop()

tab_predict, tab_insights, tab_about = st.tabs(
    ["🔮 Predict & Assess", "📊 Dataset Insights", "ℹ️ Model & Limitations"]
)

with tab_predict:
    st.subheader("Enter property details")
    with st.form("property_form"):
        c1, c2, c3 = st.columns(3)
        location = c1.selectbox("Location", locations, index=0)
        sqft = c2.number_input("Total area (sq ft)", min_value=100.0,
                               max_value=50000.0, value=1200.0, step=50.0)
        bhk = c3.number_input("BHK", min_value=1, max_value=20, value=2, step=1)
        bath = c1.number_input("Bathrooms", min_value=1, max_value=30, value=2, step=1)
        balcony = c2.number_input("Balconies", min_value=0, max_value=10, value=1, step=1)
        asking_price = c3.number_input(
            "Seller asking price (₹ lakh; optional comparison)",
            min_value=0.0, max_value=100000.0, value=0.0, step=1.0,
            help="Enter 0 if you don't know the asking price."
        )
        submitted = st.form_submit_button("Predict property price", type="primary")

    if submitted:
        sample = pd.DataFrame([{
            "location": location, "total_sqft": sqft, "bhk": float(bhk),
            "bath": float(bath), "balcony": float(balcony)
        }])
        prediction = max(0.0, float(model.predict(sample)[0]))
        estimated_ppsf = prediction * 100000 / sqft
        stats = meta.get("location_stats", {}).get(location)
        a, b, c = st.columns(3)
        a.metric("Estimated price", f"₹{prediction:,.2f} lakh")
        b.metric("Approx. price per sq ft", f"₹{estimated_ppsf:,.0f}")
        c.metric("Property area", f"{sqft:,.0f} sq ft")

        st.subheader("Rule-based investment screening")
        if stats:
            area_median = float(stats["median_price_per_sqft"])
            delta_pct = (estimated_ppsf - area_median) / area_median * 100 if area_median else 0
            st.write(f"Location median price per sq ft in training data: ₹{area_median:,.0f}")
            st.write(f"Model estimate compared with that median: {delta_pct:+.1f}%")
            if asking_price > 0:
                asking_ppsf = asking_price * 100000 / sqft
                asking_vs_median = (asking_ppsf - area_median) / area_median * 100 if area_median else 0
                st.write(f"Seller asking price per sq ft: ₹{asking_ppsf:,.0f}")
                st.write(f"Asking price compared with location median: {asking_vs_median:+.1f}%")
                if asking_price <= prediction * 0.95 and asking_vs_median < 0:
                    st.success("Screening result: Potentially worth further investigation — asking price is below both the model estimate and the location median.")
                elif asking_price >= prediction * 1.10 or asking_vs_median > 15:
                    st.warning("Screening result: Review carefully — asking price is notably above the model estimate or location median.")
                else:
                    st.info("Screening result: Compare more properties and verify condition, legal documents, and recent local transactions.")
            else:
                if delta_pct < -10:
                    st.info("Screening result: Model estimate is below the location median. Verify comparable properties and property condition.")
                elif delta_pct > 10:
                    st.info("Screening result: Model estimate is above the location median. Check whether size, amenities, or property condition justify the difference.")
                else:
                    st.info("Screening result: Estimate is near the location median. Compare recent transactions and ownership costs.")
        else:
            st.info("No location-level median is available. Compare the estimate with recent verified listings.")

        st.caption("This is a simple screening rule, not a trained investment-return model. "
                   "It does not calculate rental yield, future appreciation, liquidity, taxes, or loan costs.")

with tab_insights:
    st.subheader("Dataset overview")
    df = load_dataset()
    metrics = meta.get("metrics", {})
    m1, m2, m3 = st.columns(3)
    m1.metric("Rows used for training", f"{metrics.get('rows_used', 'N/A'):,}")
    m2.metric("Test MAE", f"₹{metrics.get('mae_lakh', 'N/A')} lakh")
    m3.metric("Test R²", str(metrics.get("r2", "N/A")))
    if not df.empty:
        df.columns = [str(c).strip().lower() for c in df.columns]
        if "price" in df:
            prices = pd.to_numeric(df["price"], errors="coerce").dropna()
            st.write("Price distribution (raw dataset)")
            st.bar_chart(prices.clip(upper=prices.quantile(0.98)).value_counts().sort_index().iloc[::max(1, len(prices)//40)])
        if "location" in df:
            st.write("Most frequent locations (raw dataset)")
            st.bar_chart(df["location"].fillna("Unknown").value_counts().head(15))
        st.caption("Charts use the original CSV and may include rows removed during training cleanup.")
    else:
        st.warning("Dataset CSV not found in the data folder; model metrics are still available.")

with tab_about:
    st.subheader("Model details")
    st.write("Algorithm: Random Forest Regressor with preprocessing pipeline.")
    st.write("Inputs: location, total square feet, BHK, bathrooms, balconies.")
    st.write("Target: price in lakh rupees, as represented in the Kaggle dataset.")
    st.write("Evaluation: MAE, RMSE and R² on a held-out test split.")
    st.json(metrics)
    st.subheader("Important limitations")
    st.markdown("""
    - Results depend on dataset quality, age, coverage and local market changes.
    - Unseen or rare locations may be grouped as `Other`.
    - The dataset does not necessarily contain rental income, maintenance costs,
      appreciation history or future-market labels.
    - The investment assessment is a transparent comparison rule, not proof that
      a property will be profitable.
    - Verify current listings, title documents, approvals, building condition,
      taxes, commute, rental demand and transaction costs before any purchase.
    """)
