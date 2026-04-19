"""
Feature Selection :
Variance filter, correlation filter, RFE, and scaling
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, RFECV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler



#  Scaling
# ─────────────────────────────────────────────


#sclaing using RobustScaler to mitate the effect of outliers and skewed distributions
def scale_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, RobustScaler]:
    """Fit RobustScaler on X_train, apply to val and test."""
    feature_names = X_train.columns.tolist()

    X_train = X_train.fillna(0)
    X_val   = X_val.fillna(0)
    test    = test.fillna(0)

    scaler  = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)

    test = test.reindex(columns=feature_names, fill_value=0)
    test_s = scaler.transform(test)

    X_train = pd.DataFrame(X_train_s, columns=feature_names)
    X_val   = pd.DataFrame(X_val_s,   columns=feature_names)
    test    = pd.DataFrame(test_s,    columns=feature_names)
    return X_train, X_val, test, scaler



#  Variance filter
# ─────────────────────────────────────────────

def variance_filter(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    test: pd.DataFrame,
    threshold: float = 0.01,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list]:
    """Drop near-constant features from all splits."""
    selector      = VarianceThreshold(threshold=threshold)
    selector.fit(X_train)
    low_var_cols  = X_train.columns[~selector.get_support()].tolist()
    print(f"Low-variance columns dropped: {low_var_cols}")
    X_train = X_train.drop(columns=low_var_cols)
    X_val   = X_val.drop(columns=low_var_cols)
    test    = test.drop(columns=low_var_cols)
    return X_train, X_val, test, low_var_cols



#  Correlation filter
# ─────────────────────────────────────────────

CORR_DROP_COLS = ["GarageCars", "1stFlrSF"]

# deleting Correlated features to reduce multicollinearity
def correlation_filter(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Drop manually identified highly-correlated redundant features."""
    cols = [c for c in CORR_DROP_COLS if c in X_train.columns]
    print(f"Correlation filter dropping: {cols}")
    X_train = X_train.drop(columns=cols)
    X_val   = X_val.drop(columns=cols)
    test    = test.drop(columns=cols)
    return X_train, X_val, test



#  Recursive Feature Elimination
# ─────────────────────────────────────────────

def rfe(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    test: pd.DataFrame,
    n_estimators: int = 100,
    cv: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, RFECV, list]:
    """
    Fit RFECV with a RandomForest base estimator.
    Returns the trimmed datasets, the fitted RFECV, and selected column names.
    """
    rf  = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    rfe = RFECV(estimator=rf, step=1, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1)
    rfe.fit(X_train, y_train)

    selected_cols = X_train.columns[rfe.support_].tolist()
    print(f"RFE: optimal features = {rfe.n_features_}")
    print(f"Selected: {selected_cols}")

    X_train_rfe = X_train.loc[:, rfe.support_]
    X_val_rfe   = X_val.loc[:, rfe.support_]
    test_rfe    = test.loc[:, rfe.support_]

    return X_train_rfe, X_val_rfe, test_rfe, rfe, selected_cols
