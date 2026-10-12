import pandas as pd

# Load the retail sales dataset
file_path = "data/train.csv"
df = pd.read_csv(file_path)

# Display the first five rows
print("\nFIRST FIVE ROWS")
print(df.head())

# Display the number of rows and columns
print("\nDATASET SHAPE")
print(df.shape)

# Display column names
print("\nCOLUMN NAMES")
print(df.columns.tolist())

# Display data types
print("\nDATA TYPES")
print(df.dtypes)

# Check for missing values
print("\nMISSING VALUES")
print(df.isnull().sum())

# Display basic statistics
print("\nBASIC STATISTICS")
print(df.describe())