# example_usage.py
import numpy as np
from anomaly_detection import AnomalyDetection
from real_time_prediction import RealTimePrediction
from automl import AutoML
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load dataset (e.g., Iris dataset)
data = load_iris()
X = data.data
y = data.target

# 1. Test Anomaly Detection Model
print("Testing Anomaly Detection Model:")
ad = AnomalyDetection(contamination=0.2)
ad.fit(X)
predictions = ad.predict(X)
print("Anomaly Detection Predictions:", predictions)

# 2. Test Real-Time Prediction Model
print("\nTesting Real-Time Prediction Model:")
rtp = RealTimePrediction()
rtp.train(X, y)
new_data = np.array([[5.0, 3.5, 1.5, 0.2]])  # A new data point to predict
prediction = rtp.predict(new_data)
print(f"Prediction for {new_data}: {prediction}")

# 3. Test AutoML Pipeline
print("\nTesting AutoML Pipeline:")
automl = AutoML()
best_model, score = automl.train_and_tune(X, y)
print(f"Best Model: {best_model}")
print(f"Best Score: {score}")
