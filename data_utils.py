import pandas as pd
import numpy as np
import streamlit as st

def check_leakage(X, y, threshold=0.95):
    corr_matrix = X.join(y).corr()
    target_corr = corr_matrix['Oil_volume'].abs().sort_values(ascending=False)
    leaked_features = target_corr[target_corr > threshold].index.tolist()
    if 'Oil_volume' in leaked_features:
        leaked_features.remove('Oil_volume')
    return leaked_features

def validate_data(df):
    required_columns = ['DEPTH_MD', 'Reservoir_pressure', 'Working_hours', 'Oil_volume', 'Date', 'WELL']
    missing_required = [col for col in required_columns if col not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {', '.join(missing_required)}")
    if df.empty:
        raise ValueError("Empty dataset")
    if df['Oil_volume'].isnull().any():
        raise ValueError("Null values in target column")
    if (df['Oil_volume'] < 0).any():
        st.error("Negative Oil_volume values detected - correcting to absolute values")
        df['Oil_volume'] = df['Oil_volume'].abs()
    return df

def cap_outliers(df, cols, lower_percentile=0.01, upper_percentile=0.99):
    for col in cols:
        if df[col].dtype in ['int64', 'float64']:
            lower = df[col].quantile(lower_percentile)
            upper = df[col].quantile(upper_percentile)
            df[col] = np.clip(df[col], lower, upper)
    return df
