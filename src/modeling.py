"""
Modeling module.
Base models, hyperparameter tuning, stacking, blending, and evaluation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    StackingRegressor,
    IsolationForest,
)
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor
import lightgbm as lgb



#  Cross-validation Function
# ─────────────────────────────────────────────

def make_kfold(n_splits: int = 5, random_state: int = 42) -> KFold:
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)



#  Evaluation Fuction 
# ─────────────────────────────────────────────

def evaluate_model(
    name: str,model,X_train: pd.DataFrame,y_train: pd.Series,
    X_val: pd.DataFrame,y_val: pd.Series,kf: KFold,
) -> tuple[float, float, object]:
    """
    CV /train/val RMSE
    Returns (cv_rmse, val_rmse, model)
    """
    cv_scores = []
    for train_idx, val_idx in kf.split(X_train):
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_vl, y_vl = X_train.iloc[val_idx],   y_train.iloc[val_idx]
        m = clone(model)
        m.fit(X_tr, y_tr)
        cv_scores.append(root_mean_squared_error(y_vl, m.predict(X_vl)))
    
    # Avg CV score across folds
    cv_rmse = np.mean(cv_scores)
    model.fit(X_train, y_train)
    train_rmse = root_mean_squared_error(y_train, model.predict(X_train))
    val_rmse   = root_mean_squared_error(y_val,   model.predict(X_val))

    print(f"\n{name}:")
    print(f"  CV RMSE    : {cv_rmse:.4f} ± {np.std(cv_scores):.4f}")
    print(f"  Train RMSE : {train_rmse:.4f}")
    print(f"  Val RMSE   : {val_rmse:.4f}")
    gap = val_rmse - train_rmse
    if gap > 0.02:
        print(f"  ⚠Overfitting ~ gap: {gap:.4f}")
    else:
        print(f"  ✓Healthy — gap: {gap:.4f}")

    return cv_rmse, val_rmse, model



#  Lasso Model
# ─────────────────────────────────────────────

def tune_lasso(
    X_train: pd.DataFrame,y_train: pd.Series,
    X_val: pd.DataFrame,y_val: pd.Series,
    kf: KFold) -> tuple[float, float, Lasso]:
    
    params = {
        "alpha":    [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
        "max_iter": [1000, 5000, 10000],
    }
    grid = GridSearchCV(Lasso(random_state=42), params, cv=5,
                        scoring="neg_mean_squared_error", n_jobs=-1)
    grid.fit(X_train, y_train)
    print(f"Best Lasso params: {grid.best_params_}")
    return evaluate_model("Lasso", grid.best_estimator_, X_train, y_train, X_val, y_val, kf)


# ─────────────────────────────────────────────
#   Ridge 
# ─────────────────────────────────────────────

def ridge_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,y_val: pd.Series,
    kf: KFold) -> tuple[float, float, Ridge]:

    params = {"alpha": [0.1, 0.5, 1, 10, 50, 100]}
    grid   = GridSearchCV(Ridge(random_state=42), params, cv=5,
                          scoring="neg_mean_squared_error", n_jobs=-1)
    grid.fit(X_train, y_train)
    print(f"Best Ridge params: {grid.best_params_}")
    return evaluate_model("Ridge", grid.best_estimator_, X_train, y_train, X_val, y_val, kf)


# ─────────────────────────────────────────────
# XGBoost Model
# ─────────────────────────────────────────────

def tune_xgboost(
    X_train: pd.DataFrame,y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
    kf: KFold,n_iter: int = 30) -> tuple[float, float, XGBRegressor]:
    
    params = {
        "max_depth":        [2, 3, 4],
        "learning_rate":    [0.01, 0.03, 0.05],
        "subsample":        [0.6, 0.7, 0.8],
        "colsample_bytree": [0.6, 0.7, 0.8],
        "reg_lambda":       [1, 5, 10],
        "min_child_weight": [3, 5, 7],
    }
    search = RandomizedSearchCV(
        XGBRegressor(n_estimators=300, random_state=42),
        params, n_iter=n_iter, cv=5,
        scoring="neg_mean_squared_error", random_state=42, n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print(f"Best XGBoost params: {search.best_params_}")
    best_xgb = XGBRegressor(**search.best_params_, n_estimators=300, random_state=42)
    return evaluate_model("XGBoost", best_xgb, X_train, y_train, X_val, y_val, kf)


# ─────────────────────────────────────────────
#  LightGBM
# ─────────────────────────────────────────────

def tune_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    kf: KFold,
) -> tuple[float, float, lgb.LGBMRegressor]:
    
    params = {
        "num_leaves":    [15, 31, 63],
        "learning_rate": [0.01, 0.03, 0.05],
    }
    grid = GridSearchCV(
        lgb.LGBMRegressor(n_estimators=300, random_state=42, verbose=-1),
        params, cv=5, scoring="neg_mean_squared_error", n_jobs=-1, verbose=0,
    )
    grid.fit(X_train, y_train)
    print(f"Best LightGBM params: {grid.best_params_}")
    best_lgb = lgb.LGBMRegressor(
        **grid.best_params_,
        subsample=0.7, n_estimators=300,
        colsample_bytree=0.6, reg_alpha=0.1,
        reg_lambda=5, random_state=42, verbose=-1, n_jobs=-1,
    )
    return evaluate_model("LightGBM", best_lgb, X_train, y_train, X_val, y_val, kf)


# ─────────────────────────────────────────────
#   Random Forest Model
# ─────────────────────────────────────────────

def tune_random_forest(
    X_train: pd.DataFrame,y_train: pd.Series,
    X_val: pd.DataFrame,y_val: pd.Series,
    kf: KFold) -> tuple[float, float, RandomForestRegressor]:


    params = {
        "n_estimators":    [100, 300, 500],
        "max_depth":       [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
    }
    grid = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        params, cv=5, scoring="neg_mean_squared_error", n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    print(f"Best RF params: {grid.best_params_}")
    return evaluate_model("RandomForest", grid.best_estimator_, X_train, y_train, X_val, y_val, kf)


# ─────────────────────────────────────────────
#  Gbm Model
# ─────────────────────────────────────────────

def tune_gbm(
    X_train: pd.DataFrame,y_train: pd.Series,
    X_val: pd.DataFrame,y_val: pd.Series,
    kf: KFold) -> tuple[float, float, GradientBoostingRegressor]:
    params = {
        "n_estimators":  [100, 300, 500],
        "learning_rate": [0.001, 0.01, 0.05, 0.1, 1],
        "max_depth":     [2, 3, 4, 5],
    }
    grid = GridSearchCV(
        GradientBoostingRegressor(random_state=42),
        params, cv=5, scoring="neg_mean_squared_error", n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    print(f"Best GBM params: {grid.best_params_}")
    return evaluate_model("GBM", grid.best_estimator_, X_train, y_train, X_val, y_val, kf)



#  Stacking
# ─────────────────────────────────────────────

def build_stack(
    best_ridge, best_lasso, best_xgb, best_gbm, best_lgb,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    kf: KFold,
) -> tuple[float, float, StackingRegressor]:
    base_models = [
        ("ridge",    best_ridge),
        ("lasso",    best_lasso),
        ("xgb",      best_xgb),
        ("gbm",      best_gbm),
        ("lightgbm", best_lgb),
    ]
    stack = StackingRegressor(
        estimators=base_models,
        final_estimator=LinearRegression(),
        cv=5, n_jobs=-1,
    )
    return evaluate_model("Stacking", stack, X_train, y_train, X_val, y_val, kf)



# #  Blending
# # ─────────────────────────────────────────────

# def blend_predictions(
#     stack_model,
#     best_lgb,
#     best_ridge,
#     best_lasso,
#     best_xgb,
#     X_val: pd.DataFrame,
#     X_val_rfe: pd.DataFrame,
#     y_val: pd.Series,
#     stack_val_rmse: float,
# ) -> tuple[float, np.ndarray]:
#     stack_pred = stack_model.predict(X_val)
#     lgb_pred   = best_lgb.predict(X_val)
#     ridge_pred = best_ridge.predict(X_val_rfe)
#     lasso_pred = best_lasso.predict(X_val_rfe)
#     xgb_pred   = best_xgb.predict(X_val)

#     final_blend = (
#         0.40 * stack_pred +
#         0.10 * lgb_pred   +
#         0.20 * ridge_pred +
#         0.10 * lasso_pred +
#         0.20 * xgb_pred
#     )
#     blend_rmse = root_mean_squared_error(y_val, final_blend)
#     print(f"\nBlend Val RMSE  : {blend_rmse:.4f}")
#     print(f"Stack Val RMSE  : {stack_val_rmse:.4f}")
#     print("Blend wins " if blend_rmse < stack_val_rmse else "Stacking wins ")
#     return blend_rmse, final_blend



#  Comparison plots
# ─────────────────────────────────────────────

def plot_model_comparison(results: dict) -> None:
    """Bar plot of validation RMSE across models"""
    models = list(results.keys())
    scores = list(results.values())
    plt.figure(figsize=(8, 5))
    plt.bar(models, scores, color="steelblue", edgecolor="white")
    plt.title("Model Comparison (Validation RMSE) — lower is better")
    plt.ylabel("RMSE")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# plot_cv vs val rsults 
def plot_cv_vs_val(cv_scores: dict, val_scores: dict) -> None:
    """Grouped bar: CV RMSE vs Validation RMSE per model"""
    models = list(cv_scores.keys())
    cv  = list(cv_scores.values())
    val = list(val_scores.values())
    x   = np.arange(len(models))
    plt.figure(figsize=(10, 5))
    plt.bar(x - 0.2, cv,  0.4, label="CV RMSE")
    plt.bar(x + 0.2, val, 0.4, label="Validation RMSE")
    plt.xticks(x, models, rotation=45)
    plt.ylabel("RMSE")
    plt.title("CV vs Validation RMSE")
    plt.legend()
    plt.tight_layout()
    plt.show()
