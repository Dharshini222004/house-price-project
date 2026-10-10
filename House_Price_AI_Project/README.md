# House Price Prediction and Real Estate Investment Recommendation System Using AI

## Project overview
A Python machine-learning project using the Kaggle Bengaluru House Price dataset. It predicts estimated house prices and performs a transparent, rule-based investment screening against location-level historical price-per-square-foot statistics.

## Folder structure
```text
house_price_ai_project/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   └── Bengaluru_House_Data.csv
└── models/
    ├── house_price_pipeline.joblib  # created after training
    └── model_metadata.json          # created after training
```

## 1. Add the Kaggle dataset
Download the Bengaluru House Price dataset from Kaggle. Put its CSV file here:
`data/Bengaluru_House_Data.csv`

The expected common columns are `location`, `size`, `total_sqft`, `bath`, and `price`. If the Kaggle download has a different filename, rename it as above.

## 2. Install Python packages
Open Anaconda Prompt or terminal inside this folder:
```bash
python -m pip install -r requirements.txt
```

## 3. Train the model
```bash
python train_model.py
```
This creates the `models/` directory and saves the trained pipeline and evaluation metadata.

## 4. Run the web application
```bash
streamlit run app.py
```
Open the local URL printed in the terminal, usually `http://localhost:8501`.

## Model approach
- Data cleaning: parse square-foot ranges, extract BHK count, remove unusable values.
- Preprocessing: median imputation and scaling for numeric inputs; most-frequent imputation and one-hot encoding for location.
- Model: Random Forest Regressor.
- Evaluation: MAE, RMSE, R² on a held-out test set.
- Investment screen: compares model-estimated price per sq ft and optional seller asking price with the location median observed in the training data.

## Academic note
The investment screen is a basic rule-based module, not a model of future appreciation or rental yield. To develop a genuine investment-return model, obtain additional reliable data such as dated transaction prices, rents, vacancy rates, maintenance costs, taxes, loan interest, and locality-level time series.

## Common issues
- `FileNotFoundError`: confirm the CSV path and filename.
- Missing columns: inspect the dataset headers and update `load_and_clean()` if your Kaggle version differs.
- `streamlit` not recognized: run `python -m streamlit run app.py`.
- Poor predictions: inspect outliers and data quality; compare metrics with cross-validation and alternative models.
