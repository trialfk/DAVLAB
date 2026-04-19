import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Generate dataset
np.random.seed(0)
df = pd.DataFrame({
    "Age": np.random.randint(18, 60, 50),
    "Salary": np.random.randint(20000, 80000, 50),
    "Department": np.random.choice(["HR", "IT", "Sales"], 50)
})

# Histogram (Seaborn)
plt.figure()
sns.histplot(df["Age"], bins=10)
plt.title("Histogram of Age")
plt.show()

# Bar Plot
plt.figure()
sns.countplot(x="Department", data=df)
plt.title("Bar Chart of Department Count")
plt.show()

# Box Plot
sns.boxplot(x=df["Salary"])
plt.title("Box Plot of Salary")
plt.show()

# Violin Plot
sns.violinplot(x=df["Department"], y=df["Salary"])
plt.title("Violin Plot")
plt.show()

# Regression Plot
sns.regplot(x=df["Age"], y=df["Salary"])
plt.title("Regression Plot")
plt.show()