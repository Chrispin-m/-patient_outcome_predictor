import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def preprocess(self, df, target_column):
        X = df.drop(columns=[target_column])
        y = df[target_column]
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
        X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        X_val = self.scaler.transform(X_val)
        return X_train, X_test, X_val, y_train, y_test, y_val

    def preprocess_csv(self, file_path, target_column):
        df = pd.read_csv(file_path)
        return self.preprocess(df, target_column)
