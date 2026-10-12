import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from pathlib import Path
OUTPUT_DIR = Path(__file__).resolve().parent.parent
df = pd.read_csv("data/train.csv")

# Convert the date column from string to datetime
df["date"] = pd.to_datetime(df["date"])

# Display dataset information
print("\nDATASET INFORMATION")
print("Start date:", df["date"].min())
print("End date:", df["date"].max())
print("Number of stores:", df["store"].nunique())
print("Number of products:", df["item"].nunique())

# 1. Total sales over time

daily_sales = df.groupby("date")["sales"].sum()

print("\nDAILY SALES SUMMARY")
print(daily_sales.describe())

plt.figure(figsize=(12, 5))
plt.plot(daily_sales.index, daily_sales.values)
plt.title("Total Daily Sales Across All Stores and Products")
plt.xlabel("Date")
plt.ylabel("Units Sold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "daily_sales.png")
plt.close()

# 2. Sales by store

store_sales = df.groupby("store")["sales"].sum().sort_values(
    ascending=False
)

print("\nTOTAL SALES BY STORE")
print(store_sales)

plt.figure(figsize=(10, 5))
store_sales.plot(kind="bar")
plt.title("Total Sales by Store")
plt.xlabel("Store")
plt.ylabel("Units Sold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "store_sales.png")
plt.close()

# 3. Sales by product

item_sales = df.groupby("item")["sales"].sum().sort_values(
    ascending=False
)

print("\nTOP 10 PRODUCTS BY SALES")
print(item_sales.head(10))

plt.figure(figsize=(10, 5))
item_sales.head(10).plot(kind="bar")
plt.title("Top 10 Products by Total Sales")
plt.xlabel("Product")
plt.ylabel("Units Sold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "top_products.png")
plt.close()

print("\nAnalysis completed successfully!")