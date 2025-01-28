from data_preprocessing.vectorization import vectorize_text
from data_preprocessing.feature_engineering import feature_engineering
from models.risk_analysis_gpt import analyze_risk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

def train_model():
    df = pd.read_csv("data/training_data.csv")
    df = feature_engineering(df)
    df = vectorize_text(df, column='description')
    
    X = list(df['vector'])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    joblib.dump(model, "models/risk_model.pkl")
    print("Model training complete and saved as risk_model.pkl")
