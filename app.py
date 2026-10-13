from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import mysql.connector


# 1. Project paths
PROJECT_DIR = Path(__file__).resolve().parent

DATA_PATH = PROJECT_DIR / "data" / "train.csv"

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "smaller_random_forest.joblib"
)


# 2. Page configuration
st.set_page_config(
    page_title="Retail Sales Forecasting",
    page_icon="📊",
    layout="wide"
)


# 3. Load data and model
# 3. Load data and model

@st.cache_data
def load_data():
# Use MySQL locally when credentials are configured.
# Otherwise, load the dataset from the project folder.
 if "mysql" in st.secrets:
  connection = mysql.connector.connect(
host="127.0.0.1",
port=3306,
user="root",
password=st.secrets["mysql"]["password"],
database="retail_forecasting",
connection_timeout=3,
)


  try:
        data = pd.read_sql(
            "SELECT date, store, item, sales FROM sales",
            connection
        )
  finally:
    connection.close()

 else:
    data = pd.read_csv(DATA_PATH)

 data["date"] = pd.to_datetime(data["date"])

 return data




@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


df = load_data()
saved_model = load_model()

model = saved_model["model"]
features = saved_model["features"]


# 4. Dashboard heading
st.title("Retail Sales Forecasting & Analytics")

st.write(
    "Explore historical retail sales and estimate daily "
    "sales for individual stores and products."
)


# 5. Summary metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Records", f"{len(df):,}")

with col2:
    st.metric("Number of Stores", df["store"].nunique())

with col3:
    st.metric("Number of Products", df["item"].nunique())


st.divider()
# 5.1 Monthly Sales Trends
st.subheader("📈 Monthly Sales Trends")

monthly_df = df.copy()

monthly_df["month"] = (
    monthly_df["date"].dt.to_period("M").dt.to_timestamp()
)

monthly_sales = (
    monthly_df.groupby("month", as_index=False)["sales"]
    .sum()
)

monthly_sales = monthly_sales.rename(
    columns={"sales": "total_sales"}
)

monthly_sales = monthly_sales.sort_values("month")

st.line_chart(
    monthly_sales,
    x="month",
    y="total_sales",
    x_label="Month",
    y_label="Total Units Sold",
)

st.caption(
    "Monthly sales trends across all stores and products, "
    "January 2013 to December 2017."
)

st.divider()

# 5.2 Store-Wise Sales Comparison
st.subheader("🏬 Store-Wise Sales Comparison")

store_sales = (
    df.groupby("store", as_index=False)["sales"]
    .sum()
    .sort_values("store")
)

store_sales = store_sales.rename(
    columns={"sales": "total_sales"}
)

st.bar_chart(
    store_sales,
    x="store",
    y="total_sales",
    x_label="Store",
    y_label="Total Units Sold",
)

st.caption(
    "Comparison of total units sold by each store "
    "across the available historical data."
)

st.divider()

# 5.3 Top 10 Best-Selling Products
st.subheader("🏆 Top 10 Best-Selling Products")

top_products = (
    df.groupby("item", as_index=False)["sales"]
    .sum()
    .sort_values("sales", ascending=False)
    .head(10)
)

top_products = top_products.rename(
    columns={
        "item": "Product",
        "sales": "Total Units Sold"
    }
)

st.bar_chart(
    top_products,
    x="Product",
    y="Total Units Sold",
    x_label="Product ID",
    y_label="Total Units Sold",
)

st.caption(
    "The 10 products with the highest cumulative unit sales "
    "across all stores during the available historical period."
)

st.divider()








# 6. Store and product selection
st.subheader("Explore a Store and Product")

col1, col2 = st.columns(2)

with col1:
    selected_store = st.selectbox(
        "Select Store",
        sorted(df["store"].unique())
    )

with col2:
    selected_item = st.selectbox(
        "Select Product",
        sorted(df["item"].unique())
    )


# 7. Filter historical sales
filtered_df = df[
    (df["store"] == selected_store)
    & (df["item"] == selected_item)
].sort_values("date")

# Sales statistics for the selected store and product
st.subheader("📊 Selected Store & Product Statistics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Units Sold",
        f"{filtered_df['sales'].sum():,}"
    )

with col2:
    st.metric(
        "Average Daily Sales",
        f"{filtered_df['sales'].mean():.2f} units"
    )

with col3:
    st.metric(
        "Highest Daily Sales",
        f"{filtered_df['sales'].max():,} units"
    )

st.divider()




# 8. Display historical sales chart
st.subheader("Historical Daily Sales")

chart_df = filtered_df.set_index("date")[["sales"]]

st.line_chart(chart_df)

# 8.1 Average Sales by Day of the Week
st.subheader("📅 Average Sales by Day of the Week")

weekday_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

weekday_sales = filtered_df.copy()

weekday_sales["day_name"] = (
    weekday_sales["date"].dt.day_name()
)

weekday_summary = (
    weekday_sales.groupby("day_name")["sales"]
    .mean()
    .reindex(weekday_order)
    .reset_index()
)

weekday_summary = weekday_summary.rename(
    columns={
        "day_name": "Day",
        "sales": "Average Sales"
    }
)

st.bar_chart(
    weekday_summary,
    x="Day",
    y="Average Sales",
    x_label="Day of the Week",
    y_label="Average Units Sold"
)

st.caption(
    "Average daily units sold for the selected store and product, "
    "grouped by day of the week across the historical dataset."
)

st.divider()




# 9. Display recent historical records
st.subheader("Recent Sales Records")

st.dataframe(
    filtered_df.tail(10).sort_values(
        "date", ascending=False
    ),
    use_container_width=True
)

# Download historical sales data
st.subheader("📥 Download Sales Data")

csv_data = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Sales Data as CSV",
    data=csv_data,
    file_name=f"store_{selected_store}_item_{selected_item}_sales.csv",
    mime="text/csv"
)




# 10. Generate multi-day sales forecasts

st.subheader("📈 Multi-Day Sales Forecast")

st.caption(
    "Forecast sales for the selected store and product over "
    "the next 1–7 days. Predictions after the first day use "
    "earlier predicted sales, so errors may accumulate."
)

forecast_days = st.selectbox(
    "Select Forecast Horizon",
    options=[1, 2, 3, 4, 5, 6, 7],
    format_func=lambda days: (
        f"{days} day" if days == 1 else f"{days} days"
    ),
)

if st.button("Generate Sales Forecast"):

    history = (
        filtered_df.sort_values("date")
        .copy()
        .reset_index(drop=True)
    )

    if len(history) < 30:
        st.warning(
            "At least 30 days of historical sales are required "
            "to generate a forecast."
        )

    else:
        sales_history = history["sales"].astype(float).tolist()
        last_date = history["date"].max()

        forecast_results = []

        for day in range(1, forecast_days + 1):

            forecast_date = (
                last_date + pd.Timedelta(days=day)
            )

            input_data = {
                "store": [selected_store],
                "item": [selected_item],
                "year": [forecast_date.year],
                "month": [forecast_date.month],
                "day_of_week": [forecast_date.dayofweek],
                "day_of_month": [forecast_date.day],
                "day_of_year": [forecast_date.dayofyear],
                "sales_lag_7": [sales_history[-7]],
                "sales_lag_14": [sales_history[-14]],
                "sales_rolling_mean_7": [
                    sum(sales_history[-7:]) / 7
                ],
                "sales_rolling_mean_30": [
                    sum(sales_history[-30:]) / 30
                ],
            }

            input_df = pd.DataFrame(input_data)[features]

            prediction = float(model.predict(input_df)[0])
            prediction = max(0.0, prediction)

            forecast_results.append({
                "Date": forecast_date,
                "Predicted Sales": round(prediction, 2),
            })

            # Use the predicted sales for subsequent forecast days.
            sales_history.append(prediction)

        forecast_df = pd.DataFrame(forecast_results)

        st.success(
            f"Generated a {forecast_days}-day forecast "
            f"for Store {selected_store}, Product {selected_item}."
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Total Forecast Sales",
                f"{forecast_df['Predicted Sales'].sum():,.0f} units",
            )

        with col2:
            st.metric(
                "Average Daily Forecast",
                f"{forecast_df['Predicted Sales'].mean():,.2f} units",
            )

        st.subheader("Forecast Chart")

        st.line_chart(
            forecast_df,
            x="Date",
            y="Predicted Sales",
            x_label="Forecast Date",
            y_label="Predicted Units Sold",
        )

        st.subheader("Forecast Details")

        display_forecast = forecast_df.copy()
        display_forecast["Date"] = (
            display_forecast["Date"].dt.strftime("%Y-%m-%d")
        )

        st.dataframe(
            display_forecast,
            use_container_width=True,
            hide_index=True,
        )

        csv_forecast = (
            forecast_df.to_csv(index=False).encode("utf-8")
        )

        st.download_button(
            label="Download Forecast as CSV",
            data=csv_forecast,
            file_name=(
                f"store_{selected_store}_item_{selected_item}_"
                f"{forecast_days}_day_forecast.csv"
            ),
            mime="text/csv",
        )





# 11. Model information

st.divider()

st.subheader("Model Information")

st.write("**Algorithm:** Random Forest Regressor")
st.write("**Evaluation MAE:** 6.34 units")
st.write("**Evaluation RMSE:** 8.26 units")
st.write("**Evaluation WAPE:** 10.77%")
st.write("**Evaluation period:** January–December 2017")
st.write("**Baseline MAE:** 8.84 units")
st.write("**MAE improvement over baseline:** 28.34%")

st.caption(
"Metrics are based on a historical rolling one-day-ahead "
"evaluation. Actual sales from previous days were available "
"when generating subsequent test predictions. "
"Actual future performance may differ."
)
