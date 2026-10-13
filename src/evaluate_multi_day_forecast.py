from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


# 1. Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_DIR / "data" / "train.csv"
MODEL_PATH = PROJECT_DIR / "models" / "smaller_random_forest.joblib"

DETAILS_PATH = PROJECT_DIR / "multi_day_evaluation.csv"
SUMMARY_PATH = PROJECT_DIR / "multi_day_evaluation_summary.csv"


# 2. Load the dataset and saved model
print("Loading dataset...")

df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    ["store", "item", "date"]
).reset_index(drop=True)

print("Dataset loaded:", df.shape)

print("Loading saved Random Forest model...")

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
features = artifact["features"]

# Avoid unnecessary parallel-processing overhead.
model.n_jobs = 1

print("Saved model loaded successfully.")


# 3. Select a representative sample
# Six products across all ten stores = 60 store-product combinations.
selected_items = [1, 10, 20, 30, 40, 50]

sample = df[
    df["item"].isin(selected_items)
].copy()

print("Sample rows:", len(sample))
print("Stores:", sample["store"].nunique())
print("Products:", sample["item"].nunique())


# 4. Prepare historical sales for every selected combination
sales_histories = {}

for (store, item), group in sample.groupby(["store", "item"]):
    sales_histories[(store, item)] = (
        group.sort_values("date")["sales"].astype(float).tolist()
    )


# 5. Configure forecast origins
# The saved model was trained using data through 2016.
origins = pd.to_datetime([
    "2017-02-01",
    "2017-04-01",
    "2017-06-01",
    "2017-08-01",
    "2017-10-01",
    "2017-12-01",
])

HORIZON = 7

# Keep original sales indexed by store, item, and date.
actual_lookup = sample.set_index(
    ["store", "item", "date"]
)["sales"]

all_results = []


# 6. Evaluate forecasts at each origin
for origin in origins:

    print(f"\nEvaluating forecast origin: {origin.date()}")

    # Reconstruct the actual history available before this origin.
    histories = {}

    for (store, item), group in sample[
        sample["date"] < origin
    ].groupby(["store", "item"]):

        histories[(store, item)] = (
            group.sort_values("date")["sales"]
            .astype(float)
            .tolist()
        )

    combinations = sorted(histories.keys())

    if not combinations:
        print("No historical data available. Skipping.")
        continue

    # Check that the historical data ends immediately before the origin.
    for key in combinations:
        expected_history_length = len(
            sample[
                (sample["store"] == key[0])
                & (sample["item"] == key[1])
                & (sample["date"] < origin)
            ]
        )

        if expected_history_length < 30:
            raise ValueError(f"Insufficient history for {key}.")

    # Generate forecasts one horizon at a time.
    for horizon in range(1, HORIZON + 1):

        forecast_date = origin + pd.Timedelta(days=horizon - 1)

        feature_rows = []
        baseline_values = []

        for store, item in combinations:

            history = histories[(store, item)]

            if len(history) < 30:
                raise ValueError(
                    f"Insufficient history for Store {store}, Item {item}."
                )

            feature_rows.append({
                "store": store,
                "item": item,
                "year": forecast_date.year,
                "month": forecast_date.month,
                "day_of_week": forecast_date.dayofweek,
                "day_of_month": forecast_date.day,
                "day_of_year": forecast_date.dayofyear,
                "sales_lag_7": history[-7],
                "sales_lag_14": history[-14],
                "sales_rolling_mean_7": np.mean(history[-7:]),
                "sales_rolling_mean_30": np.mean(history[-30:]),
            })

            # Sales from seven days before the forecast date.
            baseline_values.append(history[-7])

        feature_df = pd.DataFrame(feature_rows)[features]

        # Predict all sampled combinations in one batch.
        predictions = np.maximum(
            0.0,
            model.predict(feature_df)
        )

        # Record predictions and update histories recursively.
        for index, (store, item) in enumerate(combinations):

            key = (store, item, forecast_date)

            if key not in actual_lookup.index:
                continue

            actual = float(actual_lookup.loc[key])
            predicted = float(predictions[index])
            baseline = float(baseline_values[index])

            all_results.append({
                "Forecast Origin": origin,
                "Date": forecast_date,
                "Store": store,
                "Item": item,
                "Horizon": horizon,
                "Actual Sales": actual,
                "Predicted Sales": predicted,
                "Baseline Sales": baseline,
            })

            # Predicted sales, not future actual sales, feed
            # into the next recursive forecast step.
            histories[(store, item)].append(predicted)


# 7. Save detailed predictions
results_df = pd.DataFrame(all_results)

if results_df.empty:
    raise RuntimeError(
        "No evaluation results were generated. Check the dataset and model."
    )

results_df.to_csv(DETAILS_PATH, index=False)

print("\nDetailed predictions saved to:")
print(DETAILS_PATH)


# 8. Calculate metrics for each forecast horizon
summary = []

for horizon, group in results_df.groupby("Horizon"):

    actual = group["Actual Sales"]
    predicted = group["Predicted Sales"]
    baseline = group["Baseline Sales"]

    summary.append({
        "Horizon (days)": int(horizon),
        "Random Forest MAE": round(
            mean_absolute_error(actual, predicted), 3
        ),
        "Random Forest RMSE": round(
            np.sqrt(mean_squared_error(actual, predicted)), 3
        ),
        "Baseline MAE": round(
            mean_absolute_error(actual, baseline), 3
        ),
        "Baseline RMSE": round(
            np.sqrt(mean_squared_error(actual, baseline)), 3
        ),
        "Evaluated Records": len(group),
    })

summary_df = pd.DataFrame(summary)

summary_df.to_csv(SUMMARY_PATH, index=False)

print("\n" + "=" * 70)
print("MULTI-DAY FORECAST EVALUATION")
print("=" * 70)

print(summary_df.to_string(index=False))

print("\nSummary saved to:")
print(SUMMARY_PATH)

print("\nEvaluation completed successfully!")