import unittest
import numpy as np
from train_model import load_data, train_model
from sklearn.metrics import mean_squared_error


class TestModel(unittest.TestCase):
    def test_prediction_quality(self):
        X_test, y_test = load_data()
        model = train_model()
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        self.assertLess(mse, 1.0)  

if __name__ == '__main__':
    unittest.main()