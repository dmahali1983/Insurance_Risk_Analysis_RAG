def feature_engineering(df):
    df['risk_score'] = df['claim_amount'] / df['policy_value']
    return df
