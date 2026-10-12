from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error


# 1. Define project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_DIR / "data" / "train.csv"

MODEL_PATH = (
    PROJECT_DIR / "models" / "improved_random_forest.joblib"
)


# 2. Load dataset and saved model
df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])

saved_model = joblib.load(MODEL_PATH)

model = saved_model["model"]
features = saved_model["features"]


# 3. Re-create the model's features
df = df.sort_values(
    ["store", "item", "date"]
).reset_index(drop=True)

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day_of_week"] = df["date"].dt.dayofweek
df["day_of_month"] = df["date"].dt.day
df["day_of_year"] = df["date"].dt.dayofyear

grouped_sales = df.groupby(
    ["store", "item"]
)["sales"]

df["sales_lag_7"] = grouped_sales.shift(7)
df["sales_lag_14"] = grouped_sales.shift(14)

df["sales_rolling_mean_7"] = (
    df.groupby(["store", "item"])["sales"]
    .transform(
        lambda s: s.shift(1).rolling(7).mean()
    )
)

df["sales_rolling_mean_30"] = (
    df.groupby(["store", "item"])["sales"]
    .transform(
        lambda s: s.shift(1).rolling(30).mean()
    )
)

df = df.dropna().reset_index(drop=True)


# 4. Select the test period
test_df = df[
    df["date"] >= pd.Timestamp("2017-01-01")
].copy()

X_test = test_df[features]
y_test = test_df["sales"]


# 5. Generate predictions using the saved model
predictions = model.predict(X_test)


# 6. Calculate evaluation metrics
mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)


# 7. Display results
print("SAVED MODEL EVALUATION")
print("----------------------")
print("Model loaded successfully.")
print("Test rows:", len(test_df))
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

print("\nSAMPLE PREDICTIONS")
print("------------------")

sample = test_df[
    ["date", "store", "item", "sales"]
].head(10).copy()

sample["predicted_sales"] = predictions[:10]

print(sample.to_string(index=False))