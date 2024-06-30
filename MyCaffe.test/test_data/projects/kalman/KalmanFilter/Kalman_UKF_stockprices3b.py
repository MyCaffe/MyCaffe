
import yfinance as yf
import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.animation as animation
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression
from PIL import Image  # Import Pillow library

# Fetch the stock price data
aapl = yf.Ticker("AAPL")
data = aapl.history(period="1y")

# Extract the 'Close' prices
prices = data['Close'].values
dates = data.index

# Define the state transition function
def fx(x, dt):
    F = np.array([[1, dt],
                  [0, 1]])
    x1 = np.dot(F, x)
    return x1

# Define the measurement function
def hx(x):
    return np.array([x[0]])

# Define the initial state
dt = 1.0  # Time step (1 day)
x_initial = np.array([prices[0], 0])  # Initial state [price, velocity]
measurement_noise = 10  # increase to make the curve smoother
process_noise = 0.01      # reduce to make the curve smoother
display_uncertainty = True

# Define the process and measurement noise
points = MerweScaledSigmaPoints(2, alpha=0.1, beta=2., kappa=0)
kf = UKF(dim_x=2, dim_z=1, fx=fx, hx=hx, dt=dt, points=points)
kf.x = x_initial
kf.P *= 10  # Initial covariance
kf.R = 0.1 * measurement_noise  # Measurement noise
kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=0.1) * process_noise  # Process noise

# Run the filter on historical data
filtered_prices = []
for price in prices:
    kf.predict()
    kf.update(np.array([price]))
    filtered_prices.append(kf.x[0])

filtered_prices = np.array(filtered_prices)

# Initialize peak and trough tracking
min_peaks_required = 2
peaks = []
troughs = []

# Prepare the figure for animation
fig, ax = plt.subplots(figsize=(10, 6))
line1, = ax.plot([], [], label='Original Prices')
line2, = ax.plot([], [], linestyle='--', label='Filtered Prices')
line3, = ax.plot([], [], linestyle='--', color='green', label='Predicted Prices')
uncertainty_patch = None

ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.set_title('AAPL Stock Prices - Original vs. Filtered vs. Predicted (meas=' + str(measurement_noise) + ",proc=" + str(process_noise) + ")")
ax.legend()

# Initialize the plot limits and lines
def init():
    ax.set_xlim(dates[0], dates[-1] + pd.Timedelta(days=30))
    ax.set_ylim(min(prices) * 0.9, max(prices) * 1.1)
    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    return line1, line2, line3

# Predict future prices
future_days = 10  # Number of days to predict into the future
predicted_prices = []

# Update function for animation
def update(frame):
    global filtered_prices, predicted_prices, uncertainty_patch

    # Clear previous uncertainty patch
    if uncertainty_patch is not None:
        uncertainty_patch.remove()
        uncertainty_patch = None

    if frame < len(prices):
        kf.predict()
        kf.update(np.array([prices[frame]]))
        filtered_prices[frame] = kf.x[0]
        
        line1.set_data(dates[:frame+1], prices[:frame+1])
        line2.set_data(dates[:frame+1], filtered_prices[:frame+1])
        
        # Predict future prices for the next 10 steps
        temp_state = kf.x.copy()
        temp_P = kf.P.copy()
        temp_predicted_prices = []
        for _ in range(future_days):
            kf.predict()
            temp_predicted_prices.append(kf.x[0])
        
        future_dates = pd.date_range(start=dates[frame], periods=future_days + 1, freq='B')[1:]
        line3.set_data(future_dates, temp_predicted_prices)

        # Convert to pandas Series
        if display_uncertainty:
            temp_prices_series = pd.Series(prices)

            # Calculate rolling 20-period standard deviation
            rolling_std = temp_prices_series.rolling(window=20).std()
            rolling_std = rolling_std[-len(temp_predicted_prices):].reset_index(drop=True)

            # Calculate lower and upper bounds using the rolling standard deviation
            lower_bound = temp_predicted_prices - 1 * rolling_std
            upper_bound = temp_predicted_prices + 1 * rolling_std

            # Make sure lower_bound and upper_bound are numpy arrays
            lower_bound = np.array(lower_bound)
            upper_bound = np.array(upper_bound)
        
            uncertainty_patch = ax.fill_between(future_dates, lower_bound, upper_bound, color='blue', alpha=0.1)

        # Calculate the uncertainty (2 sigma) for the predicted prices  
        lower_bound2 = np.array(temp_predicted_prices) - 2 * np.sqrt(kf.P[0, 0])
        upper_bound2 = np.array(temp_predicted_prices) + 2 * np.sqrt(kf.P[0, 0])        
        fuchsia = '#FF00FF'
        uncertainty_patch = ax.fill_between(future_dates, lower_bound2, upper_bound2, color=fuchsia, alpha=0.2)
        
        # Restore the original state
        kf.x = temp_state
        kf.P = temp_P

    return line1, line2, line3

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(prices), init_func=init, blit=False, interval=100, repeat=False)

# Save the animation as a GIF
uncertainty = ""
if display_uncertainty:
    uncertainty = "_uncertainty"
ani.save("stock_prices_animation.m" + str(measurement_noise) + ".p" + str(process_noise) + uncertainty + ".gif", writer='pillow')

plt.show()

print('done.')

