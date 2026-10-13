# Retail Sales Forecasting & Analytics

A machine learning project that forecasts daily retail sales for individual stores and products. It combines historical sales analysis, feature engineering, Random Forest regression, MySQL integration, and an interactive Streamlit dashboard.

**Live Dashboard:** [Retail Sales Forecasting & Analytics](https://retail-sales-forecasting-xiadggacya4mhc8trrwxqj.streamlit.app/)

## Project Overview

Retail businesses need reliable sales forecasts to support inventory planning, product analysis, and operational decision-making.

This project uses historical store-level and product-level sales data to predict daily sales. It provides an interactive dashboard for exploring historical trends, comparing sales performance, and generating forecasts for up to seven days.

## Key Features

* **Sales Analytics:** Explore historical sales trends and patterns.
* **Store-Level Analysis:** Compare sales performance across stores.
* **Product-Level Analysis:** Examine product demand and identify high-performing products.
* **Interactive Filters:** Analyze sales using store and product selections.
* **Machine Learning Forecasting:** Generate sales predictions using a trained Random Forest regression model.
* **Multi-Day Forecasting:** Generate recursive forecasts for horizons of 1–7 days.
* **Forecast Visualization:** View predicted sales through charts and tables.
* **CSV Export:** Download forecast results for further analysis.
* **Database Integration:** Store and retrieve historical sales data using MySQL, with a CSV fallback for environments without database access.

## Technology Stack

* **Programming Language:** Python
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-learn
* **Data Visualization:** Matplotlib, Streamlit
* **Database:** MySQL
* **Model Serialization:** Joblib
* **Development Tools:** VS Code, Jupyter Notebook, Git, GitHub

## Dataset

The project uses the Store Item Demand Forecasting dataset.

| Property           | Description                         |
| ------------------ | ----------------------------------- |
| Number of records  | 913,000                             |
| Number of stores   | 10                                  |
| Number of products | 50                                  |
| Date range         | January 1, 2013 – December 31, 2017 |
| Target variable    | Daily sales                         |
| Main columns       | `date`, `store`, `item`, `sales`    |

The dataset contains daily sales records for different store-product combinations.

## Machine Learning Approach

### 1. Data Preparation

* Loaded and inspected historical sales data.
* Converted dates into appropriate datetime values.
* Sorted records by store, product, and date.
* Prepared time-based and historical sales features.

### 2. Feature Engineering

The forecasting model uses calendar information and historical sales patterns.

Features include:

* Store identifier
* Product identifier
* Year, month, day of the week, and day of the month
* Day of the year
* Sales from 7 days earlier
* Sales from 14 days earlier
* Seven-day rolling average of previous sales
* Thirty-day rolling average of previous sales

Historical features are constructed using past observations to avoid directly using the current day's sales as input.

### 3. Model Training

A Random Forest Regressor was trained using historical data before January 1, 2017.

The primary evaluation period for the one-day-ahead experiment was January–December 2017.

### 4. Forecast Generation

For multi-day forecasting, the application predicts one day at a time and uses each prediction to update the history for the next forecast.

This recursive approach allows the dashboard to generate forecasts for up to seven days.

## Model Evaluation

The project includes two separate evaluation experiments. Their results are reported independently because they use different evaluation procedures.

### Experiment 1: Rolling One-Day-Ahead Evaluation

The model was evaluated on 2017 data using a chronological train-test split.

For each prediction, historical sales observations were available to construct lag and rolling features. As the evaluation moved forward through 2017, actual observations from earlier dates in the test period could be used for subsequent predictions.

| Model                   |  MAE |  RMSE |
| ----------------------- | ---: | ----: |
| Previous-week baseline  | 8.84 | 11.60 |
| Random Forest Regressor | 6.34 |  8.26 |

The Random Forest model reduced MAE by approximately **28.3%** and RMSE by approximately **28.8%** compared with the previous-week baseline.

### Experiment 2: Recursive Multi-Day Evaluation

A separate experiment evaluated recursive forecasts over horizons of 1–7 days.

**Evaluation setup:**

* Six forecast starting dates in 2017
* All 10 stores
* Six selected products: items 1, 10, 20, 30, 40, and 50
* 60 store-product combinations per forecast starting date
* 360 evaluated records per forecast horizon
* Previous-week sales baseline
* Actual future sales used for scoring, not as inputs to subsequent predictions within a forecast

Forecast starting dates:

* February 1, 2017
* April 1, 2017
* June 1, 2017
* August 1, 2017
* October 1, 2017
* December 1, 2017

**Results:**

| Forecast Horizon | Random Forest MAE | Random Forest RMSE | Baseline MAE | Baseline RMSE |
| ---------------- | ----------------: | -----------------: | -----------: | ------------: |
| 1 day            |             7.667 |              9.978 |       10.614 |        13.642 |
| 2 days           |             7.027 |              9.636 |       10.439 |        14.118 |
| 3 days           |             7.380 |              9.688 |       10.944 |        14.395 |
| 4 days           |             7.248 |              9.169 |       10.022 |        13.087 |
| 5 days           |             6.204 |              8.140 |        9.581 |        12.887 |
| 6 days           |             7.106 |              9.166 |       10.664 |        13.899 |
| 7 days           |             6.015 |              7.803 |       10.364 |        13.700 |

The Random Forest model achieved lower MAE and RMSE than the previous-week baseline at all seven forecast horizons in this sampled evaluation.

**Metric definitions:**

* **MAE (Mean Absolute Error):** Average absolute difference between predicted and actual sales.
* **RMSE (Root Mean Squared Error):** Square root of the average squared prediction error; it penalizes larger errors more heavily.

Lower values indicate better performance.

**Evaluation limitation:** The recursive experiment covers six forecast starting dates and six selected products rather than every possible date and all 50 products. Its results should therefore be interpreted as evidence of performance on the sampled combinations, not as a guarantee of accuracy across all retail scenarios.

## Application Architecture

The project follows this general workflow:

1. Historical sales data is loaded from MySQL or the CSV fallback.
2. Pandas prepares the data for analysis.
3. The trained Random Forest model generates predictions.
4. Streamlit presents historical analytics and forecast results.
5. Users can inspect forecast tables and charts and download predictions as CSV files.

## Project Structure

retail-sales-forecasting/
├── .streamlit/
├── data/
│   └── train.csv
├── models/
│   └── smaller_random_forest.joblib
├── notebooks/
├── src/
│   ├── benchmark.py
│   ├── compact_model.py
│   ├── compare_models.py
│   ├── evaluate_model.py
│   ├── evaluate_multi_day_forecast.py
│   ├── improved_model.py
│   ├── inspect_data.py
│   ├── sales_analysis.py
│   └── train_model.py
├── app.py
├── import_to_mysql.py
├── requirements.txt
└── README.md


The structure above highlights the main project files. Other local files, generated reports, or supporting scripts may also exist.

## Installation and Setup

### Prerequisites

* Python 3.10 or a compatible Python version supported by the dependencies
* Git
* MySQL Server, if using the database integration locally

### 1. Clone the Repository

```bash
git clone https://github.com/KatravathMadhavi/retail-sales-forecasting.git
cd retail-sales-forecasting
```

### 2. Create a Virtual Environment

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Dashboard

```bash
streamlit run app.py
```

Streamlit will provide a local address to open the dashboard in your browser.

### Database Configuration

For local MySQL usage, configure the database connection using the application's expected configuration and credentials.

Do not commit database passwords, API keys, or other secrets to GitHub.

If MySQL is not configured, the application can use the CSV fallback supported by the project.

## Future Improvements

* Expand recursive evaluation across more forecast starting dates and products.
* Compare additional time-series forecasting approaches.
* Add prediction intervals to communicate forecast uncertainty.
* Improve forecast monitoring and error analysis.
* Extend the dashboard with inventory planning and demand-alert features.

## Author

Built as a hands-on machine learning project combining retail sales analysis, time-based feature engineering, Random Forest forecasting, MySQL, and Streamlit dashboard development.

