# Patient Outcome Predictor

This package uses machine learning models to predict patient outcomes based on various factors such as medical history, treatment plans, and demographic data. It helps in identifying high-risk patients and optimizing treatment strategies.

## Installation

```bash
pip install patient_outcome_predictor
```
# About Dataset
Description: The dataset used in this demo contains simulated medical records for a fictional group of patients. The dataset was generated using the Python Faker library to create realistic but fake data. The dataset includes the following fields for each patient:

Patient ID: A unique identifier for each patient (integer).
Name: A randomly generated full name (string).
Date of birth: A randomly generated date of birth with ages between 1 and 100 years old (date).
Gender: A randomly selected gender (M or F) (string).
Medical conditions: A list of three random, unique words representing medical conditions (string).
Medications: A list of three random, unique words representing medications (string).
Allergies: A list of three random, unique words representing allergies (string).
Last appointment date: A randomly generated date within the range of the last 2 years (date).
Please note that this dataset is for demonstration and testing purposes only. The data is entirely fictional and should not be used for any decision-making or analysis.


# This enhanced implementation includes real-time prediction, CSV preprocessing, and a Streamlit app for demonstration. The fictional dataset provides a realistic context for testing the package.

#Example Usage

```bash
import pandas as pd
from patient_outcome_predictor import DataPreprocessor, ModelTrainer, OutcomePredictor

# this loads dataset
df = pd.read_csv('dataset.csv')

# Preprocess the data
preprocessor = DataPreprocessor()
X_train, X_test, X_val, y_train, y_test, y_val = preprocessor.preprocess(df, 'outcome')

# Train the model
model_trainer = ModelTrainer()
model_trainer.train(X_train, y_train)

# Evaluation of the model
accuracy = model_trainer.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy}')

# Predict outcomes
predictor = OutcomePredictor(model_trainer.get_model(), preprocessor.scaler)
predictions = predictor.predict(X_test)
print(predictions)

```