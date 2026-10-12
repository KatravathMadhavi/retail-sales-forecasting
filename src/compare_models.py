from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# 1. Set project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "train.csv"


# 2. Load the dataset
df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    ["date", "store", "item"]
).reset_index(drop=True)


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


# 4. Split chronologically
split_date = pd.Timestamp("2017-01-01")

train_df = df[df["date"] < split_date]
test_df = df[df["date"] >= split_date]

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]


# 5. Define models
models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        min_samples_leaf=20,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    ),
}


# 6. Train and evaluate
results = []

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    results.append({
        "Model": name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2)
    })

    print(f"{name} MAE: {mae:.2f}")
    print(f"{name} RMSE: {rmse:.2f}")


# 7. Display comparison
results_df = pd.DataFrame(results)

results_df = results_df.sort_values("MAE")

print("\nFINAL MODEL COMPARISON")
print(results_df.to_string(index=False))


# 8. Save comparison results
OUTPUT_PATH = PROJECT_DIR / "model_comparison.csv"

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nComparison saved to:", OUTPUT_PATH)