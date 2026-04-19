import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

# Generate time series non-stationary data
np.random.seed(1)
ts = np.cumsum(np.random.randn(50))
# Train ARIMA model (p=2, d=1, q=1)
model = ARIMA(ts, order=(2,1,1))
model_fit = model.fit()
# Summary
print(model_fit.summary())
# In-sample prediction
pred = model_fit.predict(start=1, end=len(ts)-1)
# Future forecast (next 5 values)
future = model_fit.forecast(steps=5)
# Plot
plt.plot(ts, marker='o', label='Actual')
plt.plot(range(1, len(ts)), pred, '--', label='Predicted')
plt.plot(range(len(ts), len(ts)+5), future, '--', label='Forecast')
plt.legend()
plt.title("ARIMA Model")
plt.grid()
plt.show()
print("\n Future values:", future)