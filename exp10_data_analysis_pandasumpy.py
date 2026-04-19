import numpy as np
import pandas as pd

# 1. Import Dataset
df = pd.read_csv("emp_data.csv")
# df = pd.read_csv(r"emp_data.csv") #use for windows
print("\n--- Full Dataset (First 5 Rows) ---")
print(df.head())

# 2. Basic Data Inspection
print("\n--- First 3 Rows ---")
print(df.head(3))

print("\n--- Dataset Info ---")
print(df.info())

print("\n--- Missing Values ---")
print(df.isnull().sum())

# 3. Descriptive Statistics
print("\n--- Full Statistical Summary ---")
print(df.describe())

print("\n--- Mean Values ---")
print(df.mean(numeric_only=True))

print("\n--- Standard Deviation ---")
print(df.std(numeric_only=True))

print("\n--- Minimum Values ---")
print(df.min(numeric_only=True))

print("\n--- Maximum Values ---")
print(df.max(numeric_only=True))

# 4. Data Manipulation
# Add new column
df["Salary_in_Lakhs"] = df["Salary"] / 100000
print("\n--- Dataset After Adding New Column ---")
print(df.head())

# Filtering data
print("\n--- Employees with Salary > 50000 (First 5 Rows) ---")
high_salary = df[df["Salary"] > 50000]
print(high_salary.head())

# 5. NumPy Calculations
print("\n--- NumPy Calculations ---")
print("Mean Salary:", np.mean(df["Salary"]))
print("Max Age:", np.max(df["Age"]))
print("Min Experience:", np.min(df["Experience"]))
print("Std Dev Salary:", np.std(df["Salary"]))
