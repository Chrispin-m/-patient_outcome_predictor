import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

class OutcomePredictor:
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler
        self.vectorizer = CountVectorizer()

    def fit_vectorizer(self, text_data):
        self.vectorizer.fit(text_data)

    def predict(self, input_data):
        input_data_scaled = self.scaler.transform(input_data)
        predictions = self.model.predict(input_data_scaled)
        return predictions

    def predict_from_text(self, text_data):
        transformed_data = self.vectorizer.transform([text_data])
        return self.model.predict(transformed_data)
