import unittest
from data_preprocessing.data_cleaning import clean_data
from data_preprocessing.feature_engineering import feature_engineering
from data_preprocessing.vectorization import vectorize_text
import pandas as pd

class TestPipeline(unittest.TestCase):

    def test_clean_data(self):
        df = pd.DataFrame({'claim_amount': [100, None, 200, 100], 'policy_value': [500, 600, None, 500]})
        cleaned_df = clean_data(df)
        self.assertFalse(cleaned_df.isnull().values.any(), "Data cleaning failed")

    def test_feature_engineering(self):
        df = pd.DataFrame({'claim_amount': [100, 200], 'policy_value': [500, 1000]})
        df = feature_engineering(df)
        self.assertIn('risk_score', df.columns, "Feature engineering failed")

    def test_vectorization(self):
        df = pd.DataFrame({'description': ["Insurance claim for car accident", "Fire damage claim"]})
        df = vectorize_text(df, column='description')
        self.assertIn('vector', df.columns, "Vectorization failed")

if __name__ == '__main__':
    unittest.main()
