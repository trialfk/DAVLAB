# pip install numpy
import numpy as np
a = np.array([1, 2, 3, 4])
print("*** NumPy ***")
print("Array:", a)
print("Mean:", np.mean(a))
print("Square:", a ** 2)

# pip install pandas
import pandas as pd
data = {"Name": ["A", "B", "C"], "Marks": [85, 90, 78]}
df = pd.DataFrame(data)
print("\n*** Pandas ***")
print(df)
print("Average Marks:", df["Marks"].mean())

# pip install matplotlib
import matplotlib.pyplot as plt
x = [1, 2, 3, 4]
y = [5, 8, 13, 16]
print("\n*** Matplotlib ***")
plt.plot(x, y)
plt.title("Line Plot")
plt.show()

# pip install seaborn
import seaborn as sns
import matplotlib.pyplot as plt
tips = sns.load_dataset("tips")
print("\n*** Seaborn ***")
sns.boxplot(x="day", y="total_bill", data=tips)
plt.show()

# pip install scikit-learn
from sklearn.linear_model import LinearRegression
import numpy as np
X = np.array([[1], [2], [3], [4]])
y = np.array([2, 4, 6, 8])
model = LinearRegression()
model.fit(X, y)
print("\n*** Scikit-learn ***")
print("Prediction for 5:", model.predict([[5]]))

# pip install scipy
from scipy import integrate
result = integrate.quad(lambda x: x**2, 0, 2)
print("\n*** SciPy ***")
print("Integration Result:", result)

# pip install plotly
import plotly.express as px
df = px.data.iris()
print("\n*** Plotly ***")
fig = px.scatter(df, x="sepal_width", y="sepal_length")
fig.show()

# pip install statsmodels
import statsmodels.api as sm
import numpy as np
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([2.1, 3.9, 6.2, 8.0, 10.5, 12.1, 13.8, 16.2, 18.0, 20.1])
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print("\n*** Statsmodels ***")
print(model.summary())

# pip install tensorflow
import tensorflow as tf
x = tf.constant([[1.0, 2.0]])
w = tf.Variable([[2.0], [3.0]])
y = tf.matmul(x, w)
print("\n*** TensorFlow ***")
print("Output:", y.numpy())

# pip install torch
import torch
x = torch.tensor([[1.0, 2.0]])
w = torch.tensor([[2.0], [3.0]])
y = torch.matmul(x, w)
print("\n*** PyTorch ***")
print("Output:", y)


