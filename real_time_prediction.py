# real_time_prediction.py
import numpy as np
from sklearn.linear_model import LinearRegression

class RealTimePrediction:
    def __init__(self):
        self.model = LinearRegression()

    def train(self, X_train, y_train):
        """
        Train the model on the data.
        :param X_train: Features
        :param y_train: Target values
        """
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """
        Predict the target values for new data points.
        :param X: New data points (features)
        :return: Predicted values
        """
        return self.model.predict(X)
