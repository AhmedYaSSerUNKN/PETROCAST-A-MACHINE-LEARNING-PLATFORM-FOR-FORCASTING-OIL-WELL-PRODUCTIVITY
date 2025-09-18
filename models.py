import numpy as np
import logging
import traceback
import streamlit as st
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error

# Import model classes and pipelines here
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODELS = {
    # ...same as your original dictionary...
}

def train_model(df, model_name, validate_data, cap_outliers, check_leakage):
    try:
        df = validate_data(df)
        df = df.groupby('WELL').filter(lambda x: len(x) > 5)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df = cap_outliers(df, numeric_cols)
        features = ['DEPTH_MD', 'Reservoir_pressure', 'Working_hours']
        leaked_features = check_leakage(df[features], df['Oil_volume'])
        if leaked_features:
            st.warning(f"High leakage detected in: {', '.join(leaked_features)}")
            features = [f for f in features if f not in leaked_features]
        df = df.sort_values('Date')
        total_len = len(df)
        train_size = int(total_len * 0.8)
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]
        X_train, X_test = train_df[features], test_df[features]
        y_train, y_test = train_df['Oil_volume'], test_df['Oil_volume']
        model_config = MODELS[model_name]
        model_pipe = model_config['pipeline']
        param_grid = {f'model__{key}': value for key, value in model_config['params'].items()}
        tscv = TimeSeriesSplit(n_splits=5)
        grid_search = GridSearchCV(
            estimator=model_pipe,
            param_grid=param_grid,
            cv=tscv,
            scoring='neg_mean_squared_error',
            verbose=0
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        cv_r2 = np.mean(cross_val_score(best_model, X_train, y_train, cv=tscv, scoring='r2'))
        test_r2 = r2_score(y_test, best_model.predict(X_test))
        test_mse = mean_squared_error(y_test, best_model.predict(X_test))
        cv_mse = -np.mean(cross_val_score(best_model, X_train, y_train, cv=tscv, scoring='neg_mean_squared_error'))
        return {
            'model': best_model,
            'features': features,
            'cv_r2': cv_r2,
            'test_r2': test_r2,
            'test_data': (X_test, y_test),
            'cv_mse': cv_mse,
            'test_mse': test_mse,
            'feature_stats': {
                col: {
                    'min': float(df[col].min()),
                    'max': float(df[col].max())
                } for col in features
            }
        }
    except Exception as e:
        logging.error(f"Training failed: {str(e)}\n{traceback.format_exc()}")
        st.error(f"Error training model: {str(e)}")
        return None
