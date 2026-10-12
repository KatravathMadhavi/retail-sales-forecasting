import pandas as pd
import mysql.connector

# 1. Connect to MySQL
connection = mysql.connector.connect(
    host="127.0.0.1",
    port=3306,
    user="root",
    password=input("Enter your MySQL root password: "),
    database="retail_forecasting"
)

print("Connected to MySQL successfully!")

# 2. Read the dataset
df = pd.read_csv("data/train.csv", parse_dates=["date"])

print("Dataset loaded:", len(df), "rows")

# 3. Insert records in batches
cursor = connection.cursor()

sql = """
INSERT INTO sales (date, store, item, sales)
VALUES (%s, %s, %s, %s)
"""

batch_size = 5000

for start in range(0, len(df), batch_size):
    batch = df.iloc[start:start + batch_size]

    records = [
        (
            row.date.date(),
            int(row.store),
            int(row.item),
            int(row.sales)
        )
        for row in batch.itertuples(index=False)
    ]

    cursor.executemany(sql, records)
    connection.commit()

    print(f"Imported {min(start + batch_size, len(df))} / {len(df)} rows")

# 4. Verify the import
cursor.execute("SELECT COUNT(*) FROM sales")
count = cursor.fetchone()[0]

print("\nTotal records in MySQL:", count)

cursor.close()
connection.close()

print("Import completed successfully!")