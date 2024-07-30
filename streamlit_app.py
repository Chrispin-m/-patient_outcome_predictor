import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from patient_outcome_predictor.data_preprocessing import DataPreprocessor
from patient_outcome_predictor.model import ModelTrainer
from patient_outcome_predictor.predictor import OutcomePredictor

# Load data
data_file = 'example_data/fictional_medical_records.csv'
target_column = 'outcome'

# Preprocess data
preprocessor = DataPreprocessor()
X_train, X_test, X_val, y_train, y_test, y_val = preprocessor.preprocess_csv(data_file, target_column)

# Train model
model_trainer = ModelTrainer()
model_trainer.train(X_train, y_train)

# Create predictor
predictor = OutcomePredictor(model_trainer.get_model(), preprocessor.scaler)

# Fit vectorizer on medical conditions
df = pd.read_csv(data_file)
predictor.fit_vectorizer(df['Medical conditions'])

# Streamlit app
st.title('Patient Outcome Predictor')

st.text_input('Enter medical conditions:', key='input_conditions')

input_conditions = st.session_state.input_conditions

if input_conditions:
    prediction = predictor.predict_from_text(input_conditions)
    st.write(f'Predicted outcome: {prediction[0]}')
