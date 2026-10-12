from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# 1. Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "train.csv"
MODEL_DIR = PROJECT_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# 2. Load and sort data
df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    ["store", "item", "date"]
).reset_index(drop=True)

print("Dataset loaded:", df.shape)


# 3. Create calendar features
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day_of_week"] = df["date"].dt.dayofweek
df["day_of_month"] = df["date"].dt.day
df["day_of_year"] = df["date"].dt.dayofyear


# 4. Create historical sales features
grouped_sales = df.groupby(
    ["store", "item"]
)["sales"]

df["sales_lag_7"] = grouped_sales.shift(7)
df["sales_lag_14"] = grouped_sales.shift(14)

# Shift first so today's sales are never used
# to calculate today's rolling features.
df["sales_rolling_mean_7"] = (
    df.groupby(["store", "item"])["sales"]
    .transform(
        lambda series: series.shift(1).rolling(7).mean()
    )
)

df["sales_rolling_mean_30"] = (
    df.groupby(["store", "item"])["sales"]
    .transform(
        lambda series: series.shift(1).rolling(30).mean()
    )
)


# 5. Remove rows without sufficient history
df = df.dropna().reset_index(drop=True)


# 6. Define features and target
FEATURES = [
    "store",
    "item",
    "year",
    "month",
    "day_of_week",
    "day_of_month",
    "day_of_year",
    "sales_lag_7",
    "sales_lag_14",
    "sales_rolling_mean_7",
    "sales_rolling_mean_30",
]

TARGET = "sales"


# 7. Split chronologically
split_date = pd.Timestamp("2017-01-01")

train_df = df[df["date"] < split_date].copy()
test_df = df[df["date"] >= split_date].copy()

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]

print("\nTRAINING AND TESTING")
print("Training rows:", len(train_df))
print("Testing rows:", len(test_df))


# 8. Previous-week baseline
baseline_predictions = test_df["sales_lag_7"]

baseline_mae = mean_absolute_error(
    y_test, baseline_predictions
)

baseline_rmse = np.sqrt(
    mean_squared_error(y_test, baseline_predictions)
)

print("\nPREVIOUS-WEEK BASELINE")
print(f"MAE:  {baseline_mae:.2f}")
print(f"RMSE: {baseline_rmse:.2f}")


# 9. Train Random Forest
model = RandomForestRegressor(
    n_estimators=100,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
)

print("\nTraining improved Random Forest...")

model.fit(X_train, y_train)


# 10. Evaluate the model
predictions = model.predict(X_test)

model_mae = mean_absolute_error(
    y_test, predictions
)

model_rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

print("\nIMPROVED RANDOM FOREST RESULTS")
print(f"MAE:  {model_mae:.2f}")
print(f"RMSE: {model_rmse:.2f}")


# 11. Save the trained model and feature names
model_path = MODEL_DIR / "improved_random_forest.joblib"

import joblib

joblib.dump(
    {
        "model": model,
        "features": FEATURES,
        "last_training_date": train_df["date"].max(),
    },
    model_path,
)

print("\nModel saved to:", model_path)
print("\nImproved forecasting experiment completed!")