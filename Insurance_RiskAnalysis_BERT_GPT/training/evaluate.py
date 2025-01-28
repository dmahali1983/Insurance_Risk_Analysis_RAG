from sklearn.metrics import classification_report
import joblib
import pandas as pd
from data_preprocessing.vectorization import vectorize_text
from data_preprocessing.feature_engineering import feature_engineering

def evaluate_model():
    df = pd.read_csv("data/evaluation_data.csv")
    df = feature_engineering(df)
    df = vectorize_text(df, column='description')

    X = list(df['vector'])
    y = df['label']

    model = joblib.load("models/risk_model.pkl")
    y_pred = model.predict(X)

    report = classification_report(y, y_pred)
    print("Evaluation Report:\n", report)
