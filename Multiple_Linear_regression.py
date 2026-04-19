import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
#dataset
np.random.seed(0)
n = 150
X1 = np.random.rand(n) * 10   
X2 = np.random.rand(n) * 5      
y = 3*X1 + 2*X2 + 5 + np.random.randn(n) * 2
# Combine features
X = np.column_stack((X1, X2))
# Train model
model = LinearRegression()
model.fit(X, y)
# Predictions
y_pred = model.predict(X)
# Metrics
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y, y_pred)
# Plot 
plt.scatter(X1, y, color='black', label="Data")
x_range = np.linspace(X1.min(), X1.max(), 50)
for val in [1, 3, 5]:
    y_line = model.coef_[0]*x_range + model.coef_[1]*val + model.intercept_
    plt.plot(x_range, y_line, label=f'X2 = {val}')

plt.xlabel("X1")
plt.ylabel("Y")
plt.title("Multiple Linear Regression (2D Lines)")
plt.legend()
plt.grid()
plt.show()
# Output
print(f"Coefficients: {model.coef_.round(2)}")
print(f"Intercept: {model.intercept_:.2f}")
print(f"R² Score: {r2:.3f}")
print(f"MSE: {mse:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"MAE: {mae:.3f}")