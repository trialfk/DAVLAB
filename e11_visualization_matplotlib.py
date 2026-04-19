import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate dataset
np.random.seed(0)
df = pd.DataFrame({
    "Age": np.random.randint(18, 60, 50),
    "Salary": np.random.randint(20000, 80000, 50),
    "Department": np.random.choice(["HR", "IT", "Sales"], 50)
})

# Histogram
plt.hist(df["Age"])
plt.title("Histogram of Age")
plt.show()

# Bar Chart
df["Department"].value_counts().plot(kind='bar')
plt.title("Bar chart")
plt.show()

# Pie Chart
df["Department"].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title("Pie Chart")
plt.ylabel("")
plt.show()

# Box Plot
plt.figure()
plt.boxplot(df["Salary"])
plt.title("Box Plot of Salary")
plt.ylabel("Salary")
plt.show()