import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
np.random.seed(1)
# Generate dataset
ts = np.cumsum(np.random.randn(50))
# Train model
model = AutoReg(ts, lags=2)
model_fit = model.fit()
# In-sample predictions
pred = model_fit.predict(start=2, end=len(ts)-1)
# Future forecast (next 5 values)
future = model_fit.predict(start=len(ts), end=len(ts)+4)
# Summary
print(model_fit.summary())
# Plot
plt.plot(ts, marker='o', label='Actual')
# Predicted (in-sample)
plt.plot(range(2, len(ts)), pred, marker='x', linestyle='--', label='Predicted')
# Future forecast
plt.plot(range(len(ts), len(ts)+5), future,
         marker='x', linestyle='--', label='Future Forecast')
plt.legend()
plt.title("AR Model with Forecast")
plt.grid()
plt.show()
print("Future values:", future)