from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Project paths

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "train.csv"

# 2. Load and sort dataset

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

grouped_sales = df.groupby(["store", "item"])["sales"]

df["sales_lag_7"] = grouped_sales.shift(7)
df["sales_lag_14"] = grouped_sales.shift(14)

df["sales_rolling_mean_7"] = (
df.groupby(["store", "item"])["sales"]
.transform(lambda series: series.shift(1).rolling(7).mean())
)

df["sales_rolling_mean_30"] = (
df.groupby(["store", "item"])["sales"]
.transform(lambda series: series.shift(1).rolling(30).mean())
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

# 7. Split data chronologically

split_date = pd.Timestamp("2017-01-01")

train_df = df[df["date"] < split_date].copy()
test_df = df[df["date"] >= split_date].copy()

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]

print("\nDATA SPLIT")
print("Training rows:", len(train_df))
print("Testing rows:", len(test_df))

# 8. Previous-week baseline

baseline_predictions = test_df["sales_lag_7"].to_numpy()

# 9. Train compact Random Forest

model = RandomForestRegressor(
n_estimators=50,
max_depth=12,
min_samples_leaf=5,
random_state=42,
n_jobs=-1,
)

print("\nTraining compact Random Forest...")

model.fit(X_train, y_train)

model_predictions = model.predict(X_test)

# 10. Calculate evaluation metrics
def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    wape = (
np.sum(np.abs(actual - predicted))
/ np.sum(np.abs(actual))
* 100
)
    return mae, rmse, wape




baseline_mae, baseline_rmse, baseline_wape = calculate_metrics(
y_test.to_numpy(),
baseline_predictions,
)

model_mae, model_rmse, model_wape = calculate_metrics(
y_test.to_numpy(),
model_predictions,
)

# 11. Calculate percentage improvement

mae_improvement = (
(baseline_mae - model_mae) / baseline_mae
) * 100

rmse_improvement = (
(baseline_rmse - model_rmse) / baseline_rmse
) * 100

# 12. Display benchmark results

results = pd.DataFrame({
"Model": [
"Previous-week baseline",
"Compact Random Forest",
],
"MAE": [
baseline_mae,
model_mae,
],
"RMSE": [
baseline_rmse,
model_rmse,
],
"WAPE (%)": [
baseline_wape,
model_wape,
],
})

print("\nBENCHMARK RESULTS")
print(
results.to_string(
index=False,
float_format=lambda value: f"{value:.2f}",
)
)

print("\nIMPROVEMENT OVER BASELINE")
print(f"MAE improvement:  {mae_improvement:.2f}%")
print(f"RMSE improvement: {rmse_improvement:.2f}%")

print("\nBenchmark completed successfully.")
