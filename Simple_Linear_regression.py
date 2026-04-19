import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# Generate data 
np.random.seed(0)
x = np.linspace(1, 20, 50).reshape(-1, 1) 
y = 3 * x.flatten() + 5 + np.random.randn(50) * 3 
# Train model
model = LinearRegression()
model.fit(x, y)
# Predictions
y_pred = model.predict(x)
# Metrics
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y, y_pred)
# Plot
plt.scatter(x, y, label="Actual Data")
plt.plot(x, y_pred, '--', label="Regression Line")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression (Generated Data)")
plt.legend()
plt.grid()
plt.show()
# Output
print(f"Slope (m): {model.coef_[0]:.2f}")
print(f"Intercept (c): {model.intercept_:.2f}")
print(f"R² Score: {r2:.3f}")
print(f"MSE: {mse:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"MAE: {mae:.3f}")