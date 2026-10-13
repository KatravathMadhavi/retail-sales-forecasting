# Retail Sales Forecasting & Analytics

## Project Overview

Retail Sales Forecasting & Analytics is a machine learning project that analyzes historical retail sales and predicts next-day unit sales for individual stores and products.

The project combines historical sales analysis, machine learning, MySQL database integration, and an interactive Streamlit dashboard to demonstrate how data can support retail decision-making.

## Key Features

* **Sales Analytics Dashboard:** Displays overall dataset statistics and historical sales patterns.
* **Monthly Sales Trends:** Visualizes monthly sales across the historical period.
* **Store-Wise Sales Comparison:** Compares cumulative unit sales across stores.
* **Top 10 Best-Selling Products:** Identifies products with the highest cumulative unit sales.
* **Sales Statistics:** Calculates total, average, and maximum daily sales for a selected store-product combination.
* **Day-of-Week Analysis:** Compares average sales across weekdays.
* **Next-Day Sales Prediction:** Estimates the next day's unit sales using a trained machine learning model.
* **CSV Export:** Allows users to download historical sales records for a selected store and product.

## Technologies Used

* Python
* Pandas and NumPy
* Matplotlib
* Scikit-learn
* Random Forest Regressor
* MySQL
* Streamlit
* Joblib

## Dataset

The project uses the Store Item Demand Forecasting dataset, containing historical daily unit sales.

* **Records:** 913,000
* **Stores:** 10
* **Products:** 50
* **Historical period:** January 2013 – December 2017
* **Target variable:** Daily units sold

The dataset contains date, store ID, product ID, and sales quantity.

## Machine Learning Approach

The project compares Linear Regression, Decision Tree, and Random Forest regression models.

The improved Random Forest model uses calendar features and historical sales features, including:

* Store and product identifiers
* Year, month, day of the week, day of the month, and day of the year
* Sales from 7 and 14 days earlier
* Rolling average sales over the previous 7 and 30 days

Historical sales features are shifted so that the target day's actual sales are not used to construct its prediction features.

## Model Evaluation

The compact Random Forest model was evaluated on a chronological test period covering January–December 2017 and compared with a previous-week sales baseline.

| Model                  |  MAE |  RMSE |   WAPE |
| ---------------------- | ---: | ----: | -----: |
| Previous-week baseline | 8.84 | 11.60 | 15.03% |
| Compact Random Forest  | 6.34 |  8.26 | 10.77% |

**Evaluation summary:**

* **MAE:** 6.34 sales units for the compact Random Forest, compared with 8.84 for the baseline.
* **RMSE:** 8.26 for the compact Random Forest, compared with 11.60 for the baseline.
* **WAPE:** 10.77% for the compact Random Forest, compared with 15.03% for the baseline.
* **MAE improvement:** 28.34% compared with the previous-week baseline.
* **RMSE improvement:** 28.76% compared with the previous-week baseline.

Mean Absolute Error (MAE) measures the average absolute difference between predicted and actual sales. Root Mean Squared Error (RMSE) penalizes larger prediction errors more heavily. Weighted Absolute Percentage Error (WAPE) expresses the total absolute prediction error as a percentage of total actual sales.

The data was split chronologically, using observations before January 2017 for training and observations from 2017 for testing. Historical lag and rolling-average features were constructed using earlier sales observations.

The evaluation represents **rolling one-day-ahead predictions**: actual sales from previous days are available when constructing features for subsequent test dates. It does not represent a full-year forecast generated using only data available before January 2017.

These results are based on a historical evaluation and do not guarantee future performance.

## Database

MySQL stores the historical sales records in the `retail_forecasting` database.

The Streamlit application retrieves sales data from MySQL and loads the saved machine learning model to generate predictions.

## Running the Project

1. Clone or download the project.
2. Create and activate a Python virtual environment.
3. Install the dependencies listed in `requirements.txt`.
4. Set up MySQL and create the `retail_forecasting` database and `sales` table.
5. Import the dataset into MySQL using the provided import script. Import the records only once.
6. Configure the MySQL password in `.streamlit/secrets.toml`.
7. Run the dashboard using:

   `streamlit run app.py`

Keep database passwords and other secrets out of public repositories.

## Project Limitations

* The dataset ends on December 31, 2017, so the demonstration predicts the next day after the latest available historical date.
* The current prediction interface generates a one-day-ahead estimate rather than a complete multi-day forecast.
* Model performance may differ on new retail data.


