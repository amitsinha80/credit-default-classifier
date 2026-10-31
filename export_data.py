import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

# header=1 because row 0 of the UCI file holds X1..X23 labels; row 1 has real names
df = pd.read_excel("data/default of credit card clients.xls", header=1)

print("Shape (rows, columns):", df.shape)
print("\nColumns:\n", list(df.columns))
print("\nFirst 5 rows:\n", df.head())

print("\nTotal missing values:", df.isnull().sum().sum())

target = "default payment next month"
print("\nTarget counts:\n", df[target].value_counts())
print("\nTarget proportions:\n", df[target].value_counts(normalize=True).round(3))

# Check categorical codes against the UCI documentation
print("\nEDUCATION codes:", sorted(df["EDUCATION"].unique()))
print("MARRIAGE codes:", sorted(df["MARRIAGE"].unique()))
print("PAY_0 codes:", sorted(df["PAY_0"].unique()))

