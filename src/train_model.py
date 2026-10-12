from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# 1. Define project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_DIR / "data" / "train.csv"
MODEL_DIR = PROJECT_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# 2. Load the dataset
df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(["date", "store", "item"]).reset_index(drop=True)

print("Dataset loaded successfully!")
print("Rows:", len(df))
print("Date range:", df["date"].min().date(), "to", df["date"].max().date())


# 3. Create date-based features
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day_of_week"] = df["date"].dt.dayofweek
df["day_of_month"] = df["date"].dt.day
df["day_of_year"] = df["date"].dt.dayofyear

FEATURES = [
    "store",
    "item",
    "year",
    "month",
    "day_of_week",
    "day_of_month",
    "day_of_year",
]

TARGET = "sales"


# 4. Split by date
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


# 5. Create a baseline prediction
# Use average training sales for each store-item pair.
baseline_averages = train_df.groupby(
    ["store", "item"]
)["sales"].mean()

baseline_predictions = pd.MultiIndex.from_frame(
    test_df[["store", "item"]]
).map(baseline_averages)

baseline_predictions = np.asarray(
    baseline_predictions, dtype=float
)

baseline_mae = mean_absolute_error(
    y_test, baseline_predictions
)

baseline_rmse = np.sqrt(
    mean_squared_error(y_test, baseline_predictions)
)

print("\nBASELINE RESULTS")
print(f"MAE:  {baseline_mae:.2f}")
print(f"RMSE: {baseline_rmse:.2f}")


# 6. Train Random Forest
model = RandomForestRegressor(
    n_estimators=100,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
)

print("\nTraining Random Forest...")
model.fit(X_train, y_train)


# 7. Evaluate Random Forest
predictions = model.predict(X_test)

model_mae = mean_absolute_error(
    y_test, predictions
)

model_rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

print("\nRANDOM FOREST RESULTS")
print(f"MAE:  {model_mae:.2f}")
print(f"RMSE: {model_rmse:.2f}")


# 8. Compare results
results = pd.DataFrame({
    "Model": ["Store-Item Average Baseline", "Random Forest"],
    "MAE": [baseline_mae, model_mae],
    "RMSE": [baseline_rmse, model_rmse],
})

print("\nMODEL COMPARISON")
print(results.to_string(index=False))


# 9. Save the trained model
model_path = MODEL_DIR / "random_forest.joblib"

joblib.dump(
    {
        "model": model,
        "features": FEATURES,
        "last_training_date": train_df["date"].max(),
    },
    model_path,
)

print("\nModel saved to:", model_path)
print("\nForecasting pipeline completed successfully!")