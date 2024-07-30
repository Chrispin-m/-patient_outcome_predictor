import unittest
import pandas as pd
from sklearn.datasets import make_classification
from patient_outcome_predictor.data_preprocessing import DataPreprocessor
from patient_outcome_predictor.model import ModelTrainer
from patient_outcome_predictor.predictor import OutcomePredictor

class TestOutcomePredictor(unittest.TestCase):
    def test_predict(self):
        X, y = make_classification(n_samples=100, n_features=20, random_state=42)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
        df['outcome'] = y

        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.preprocess(df, 'outcome')

        model_trainer = ModelTrainer()
        model_trainer.train(X_train, y_train)

        predictor = OutcomePredictor(model_trainer.get_model(), preprocessor.scaler)
        predictions = predictor.predict(X_test)
        self.assertEqual(len(predictions), len(y_test))

if __name__ == '__main__':
    unittest.main()
