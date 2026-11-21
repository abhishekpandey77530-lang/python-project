
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. Load Dataset
df = pd.read_csv("bank_data.csv")

print("Original Dataset:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

# 2. Data Cleaning
df.columns = df.columns.str.strip()
df.drop_duplicates(inplace=True)

for col in ["City", "Region"]:
    df[col] = df[col].astype("string").str.strip()

for col in ["Branch_ID", "Firm_Revenue", "Expenses", "Profit_Margin"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["Firm_Revenue"] = df["Firm_Revenue"].fillna(
    df["Firm_Revenue"].median()
)

df.dropna(inplace=True)

df = df[
    (df["Firm_Revenue"] >= 0) &
    (df["Expenses"] >= 0)
].copy()

df["Profit"] = df["Firm_Revenue"] - df["Expenses"]

# Create Output folder in the same location as Python file
folder = os.path.dirname(os.path.abspath(__file__))
output_folder = os.path.join(folder, "Output")

os.makedirs(output_folder, exist_ok=True)

cleaned_file = os.path.join(output_folder, "cleaned_data.csv")
df.to_csv(cleaned_file, index=False)

print("\nCleaned CSV saved successfully!")
print("File Location:", cleaned_file)

print("\nCleaned Dataset:")
print(df.head())

# 3. Data Analysis
print("\nTotal Revenue:", np.sum(df["Firm_Revenue"]))
print("Total Expenses:", np.sum(df["Expenses"]))
print("Total Profit:", np.sum(df["Profit"]))
print("Average Profit Margin:", np.mean(df["Profit_Margin"]))

city = df.groupby("City")["Profit"].sum()
region = df.groupby("Region")["Profit"].sum()

print("\nCity-wise Profit:")
print(city)

print("\nRegion-wise Profit:")
print(region)

print("\nBest Performing City:", city.idxmax())

# Save analysis
city.to_csv("Output/city_analysis.csv")
region.to_csv("Output/region_analysis.csv")

# 4. Graphs

# Graph 1: Revenue by City
df.groupby("City")["Firm_Revenue"].sum().plot(
    kind="bar", title="Revenue by City"
)
plt.ylabel("Revenue")
plt.tight_layout()
plt.savefig("Output/revenue.png")
plt.show()

# Graph 2: Expenses by Region
df.groupby("Region")["Expenses"].sum().plot(
    kind="bar", title="Expenses by Region"
)
plt.ylabel("Expenses")
plt.tight_layout()
plt.savefig("Output/expenses.png")
plt.show()

# Graph 3: Profit Margin
plt.hist(df["Profit_Margin"], bins=15, edgecolor="black")
plt.title("Profit Margin Distribution")
plt.xlabel("Profit Margin")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("Output/profit_margin.png")
plt.show()

# Graph 4: Revenue vs Expenses
plt.scatter(df["Firm_Revenue"], df["Expenses"])
plt.title("Revenue vs Expenses")
plt.xlabel("Revenue")
plt.ylabel("Expenses")
plt.tight_layout()
plt.savefig("Output/revenue_expenses.png")
plt.show()

print("\nProject Completed Successfully!")
