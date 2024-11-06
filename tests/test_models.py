# tests/test_models.py
import unittest
import numpy as np
from anomaly_detection import AnomalyDetection
from real_time_prediction import RealTimePrediction
from automl import AutoML
from sklearn.datasets import load_iris

class TestModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load dataset for tests
        data = load_iris()
        cls.X = data.data
        cls.y = data.target

    def test_anomaly_detection(self):
        ad = AnomalyDetection(contamination=0.2)
        ad.fit(self.X)
        predictions = ad.predict(self.X)
        self.assertEqual(len(predictions), len(self.X))  # Test if the length matches
        self.assertTrue(np.all(np.isin(predictions, [-1, 1])))  # Check if predictions are either -1 or 1

    def test_real_time_prediction(self):
        rtp = RealTimePrediction()
        rtp.train(self.X, self.y)
        new_data = np.array([[5.0, 3.5, 1.5, 0.2]])  # A new data point to predict
        prediction = rtp.predict(new_data)
        self.assertEqual(len(prediction), 1)  # Should return one prediction value

    def test_automl(self):
        automl = AutoML()
        best_model, score = automl.train_and_tune(self.X, self.y)
        self.assertIsNotNone(best_model)  # Test if a model was found
        self.assertGreater(score, 0)  # Test if the score is greater than 0

if __name__ == '__main__':
    unittest.main()
