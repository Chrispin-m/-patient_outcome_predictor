import unittest
from sklearn.datasets import make_classification
from patient_outcome_predictor.model import ModelTrainer

class TestModelTrainer(unittest.TestCase):
    def test_train_and_evaluate(self):
        X, y = make_classification(n_samples=100, n_features=20, random_state=42)
        model_trainer = ModelTrainer()
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]
        model_trainer.train(X_train, y_train)
        accuracy = model_trainer.evaluate(X_test, y_test)
        self.assertGreater(accuracy, 0.7)

if __name__ == '__main__':
    unittest.main()
