# anomaly_detection.py
from sklearn.ensemble import IsolationForest
import numpy as np

class AnomalyDetection:
    def __init__(self, contamination=0.1):
        self.contamination = contamination
        self.model = IsolationForest(contamination=self.contamination)

    def fit(self, X):
        """
        Fit the model with data X.
        :param X: 2D numpy array (features)
        """
        self.model.fit(X)

    def predict(self, X):
        """
        Predict if data points are anomalies.
        :param X: 2D numpy array (features)
        :return: Anomaly labels (-1: anomaly, 1: normal)
        """
        return self.model.predict(X)

    def fit_predict(self, X):
        """
        Fit the model and predict anomalies at once.
        :param X: 2D numpy array (features)
        :return: Anomaly labels
        """
        return self.model.fit_predict(X)
