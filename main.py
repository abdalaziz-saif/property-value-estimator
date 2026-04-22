"""
Main pipeline 

Usage:
    python main.py --data_dir data/

Dataset expected at data_dir:
    train.csv
    test.csv
    sample_submission.csv
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

from src.EDA import (
    missing_summary,
    plot_missing_effect,
    plot_missing_heatmap,
    plot_age_distributions,
    plot_discrete_univariate,
    plot_cont_univ,
    plot_cont_biv,
    print_skewness,
    plot_univariate_and_bivariate_categorical,
    plot_PairPlot,
    plot_correlation_heatmap,
    plot_target_distribution,
    get_years_features,
    get_discrete_features,
    get_continuous_features,
)
from src.features import (
    add_features,
    years_mod,
    drop_features,
    impute,
    flag,
    transform,
    bin_discrete,
    remove_manual_outliers,
    remove_isolation_forest_outliers,
    run_ordinal_enc_combined,
    run_nominal_enc_combined,
)
from src.selection import (
    scale_features,
    variance_filter,
    correlation_filter,
    rfe,
)
from src.modeling import (
    make_kfold,
    evaluate_model,
    tune_catboost,
    tune_elasticNet,
    tune_lasso,
    tune_ridge,
    tune_xgboost,
    tune_lightgbm,
    tune_elasticNet,
    tune_catboost,
    tune_random_forest,
    tune_gbm,
    build_stack,
    plot_model_comparison,
    plot_cv_vs_val,
)
from src.utils import (
    target_encode,
    save_models,
    
)


def parse_args():
    parser = argparse.ArgumentParser(description="House Price Prediction Pipeline")
    parser.add_argument("--data_dir",   default="data/",    help="Directory with CSV files")
    parser.add_argument("--models_dir", default="models/",  help="Directory to save models")
    parser.add_argument("--output_dir", default="outputs/", help="Directory for submission")
    parser.add_argument("--skip_eda",   action="store_true", help="Skip EDA plots")
    return parser.parse_args()


def main():
    args = parse_args()

   # 1-  Load data 
   # ____________________________________________________________

    print("\n-- 1. Loading Data -- ")
    train  = pd.read_csv(f"{args.data_dir}/train.csv")
    test   = pd.read_csv(f"{args.data_dir}/test.csv")

    print(f"Train shape : {train.shape}")
    print(f"Test  shape : {test.shape}")

    # 2-  EDA 
    # ──────────────────────────────────────────────────────────────
    if not args.skip_eda:
        print("\n  2. Exploratory Data Analysis ---")
        print(train.head())
        print(train.info())
        print(train.describe())

        missing_summary(train, label="train")
        missing_summary(test,  label="test")
        plot_missing_effect(train, output_dir=args.output_dir)
        plot_missing_heatmap(train, output_dir=args.output_dir)

        # Feature type separation
        numerical_feature = [c for c in train.columns if train[c].dtype in ["int64", "float64"]]
        years_feature     = get_years_features(train)
        discrete_feature  = get_discrete_features(train, numerical_feature, years_feature)
        continuous_feature = get_continuous_features(train, numerical_feature, years_feature, discrete_feature)

        # Remove target and ID from predictor lists
        for feat in ["SalePrice", "Id"]:
            if feat in continuous_feature: continuous_feature.remove(feat)
            if feat in discrete_feature: discrete_feature.remove(feat)

        # Age features (
        _tmp = years_mod(train.copy())
        plot_age_distributions(_tmp, output_dir=args.output_dir)
        plot_discrete_univariate(train, discrete_feature, output_dir=args.output_dir)
        plot_cont_univ(train, continuous_feature, output_dir=args.output_dir)
        plot_cont_biv(train, continuous_feature, output_dir=args.output_dir)
        print_skewness(train, continuous_feature)

        categorical_ordinal = [
            "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "BsmtExposure",
            "BsmtFinType1", "BsmtFinType2", "HeatingQC", "KitchenQual",
            "FireplaceQu", "GarageFinish", "GarageQual", "GarageCond",
            "PoolQC", "Fence", "LandSlope", "LotShape", "Utilities", "PavedDrive",
        ]
        categorical_nominal = [
            "MSZoning", "Street", "Alley", "LandContour", "LotConfig",
            "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle",
            "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType",
            "Foundation", "Heating", "CentralAir", "Electrical", "Functional",
            "GarageType", "MiscFeature", "SaleType", "SaleCondition",
        ]
        # Univariate and bivariate categorical plots
        plot_univariate_and_bivariate_categorical(train, categorical_ordinal, "#8B5CF6", output_dir=args.output_dir, name="categorical_ordinal")
        plot_univariate_and_bivariate_categorical(train, categorical_nominal, "#F59E0B", output_dir=args.output_dir, name="categorical_nominal")
        
        # Multivariate Plots
        plot_PairPlot(train, numerical_feature, output_dir=args.output_dir)
        plot_correlation_heatmap(train, numerical_feature, output_dir=args.output_dir)
        plot_target_distribution(train, output_dir=args.output_dir)

    # 3- FEATURE ENGINEERING 
    # ──────────────────────────────────────────────
    print("\n 3. Feature Engineering --")

    # Log-transform target
    train["SalePrice"] = np.log1p(train["SalePrice"])

    # Add features
    train = years_mod(train)
    train = add_features(train)

    test = years_mod(test)
    test = add_features(test)

    # dROP FEATURES 
    train = drop_features(train)
    test  = drop_features(test)
    print(f"After drop  — train: {train.shape}  test: {test.shape}")
    
    # IMPUTE
    train = impute(train)
    test  = impute(test)
    print(f"Missing in train: {train.isnull().sum().sum()}")
    print(f"Missing in test : {test.isnull().sum().sum()}")

    # FLAG
    train = flag(train)
    test  = flag(test)

    # TRANSFORM 
    train = transform(train)
    test  = transform(test)

    # Manual outlier removal 
    train = remove_manual_outliers(train)

    # BINNING
    train = bin_discrete(train)
    test  = bin_discrete(test)

    # Ordinal encoding (train + test)
    train, test = run_ordinal_enc_combined(train, test)

    # Nominal / one-hot encoding (train + test)
    train, test = run_nominal_enc_combined(train, test)

    # Align columns
    test = test.reindex(columns=train.drop(columns=["SalePrice"]).columns, fill_value=0)
    print(f"After encoding — train: {train.shape}  test: {test.shape}")

    #  4. SPLIT
    #  ────────────────────────────────────────────────────────────
    print("\n 4. Train / Validation Split ")
    X = train.drop(columns=["SalePrice"])
    Y = train["SalePrice"]
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Target encode Neighborhood (after split to prevent leakage 👀 )
    X_train, X_val, test = target_encode(X_train, X_val, test, y_train)

    #  5. FEATURE SELECTION 
    # ────────────────────────────────────────────────
    print("-- 5. Feature Selection --")


    # IsolationForest outlier removal fOR TOP FEATURES 
    numerical_X = X_train.select_dtypes(include=["int64", "float64"])

    top_features = (
        numerical_X
        .corrwith(y_train)
        .abs()
        .sort_values(ascending=False)
        .head(6)
        .index
        .tolist()
    )
    X_train, y_train = remove_isolation_forest_outliers(X_train, y_train, top_features)

    # Variance filter
    X_train, X_val, test, _ = variance_filter(X_train, X_val, test)

    # Correlation filter
    X_train, X_val, test = correlation_filter(X_train, X_val, test)

    # Scale
    X_train, X_val, test, scaler = scale_features(X_train, X_val, test)


    # # RFE
    # X_train_rfe, X_val_rfe, test_rfe, rfe, selected_cols = rfe(
    #     X_train, y_train, X_val, test
    # )


    # _______________________________________________________________

    #  6. MODELING 
    
    # ─────────────────────────────────────────────────────────
    print("-- 6. Modeling --")
    kf = make_kfold()

    # Linear 

    lr = LinearRegression().fit(X_train, y_train)
    print(f"Linear Regression Val RMSE: {root_mean_squared_error(y_val, lr.predict(X_val)):.4f}")

    # Tuned models
    lasso_cv, lasso_val, best_lasso   = tune_lasso(X_train, y_train, X_val, y_val, kf)
    ridge_cv, ridge_val, best_ridge   = tune_ridge(X_train, y_train, X_val, y_val, kf)
    elastic_cv, elastic_val, best_elastic   = tune_elasticNet(X_train, y_train, X_val, y_val, kf)
    xgb_cv,   xgb_val,   best_xgb    = tune_xgboost(X_train, y_train, X_val, y_val, kf)
    cat_cv,   cat_val,   best_cat    = tune_catboost(X_train, y_train, X_val, y_val, kf)
    lgb_cv,   lgb_val,   best_lgb    = tune_lightgbm(X_train, y_train, X_val, y_val, kf)
    # rf_cv,    rf_val,    best_rf     = tune_random_forest(X_train, y_train, X_val, y_val, kf)
    gbm_cv,   gbm_val,   best_gbm   = tune_gbm(X_train, y_train, X_val, y_val, kf)

    # Stacking
    stack_cv, stack_val, stack_model = build_stack(
        best_ridge, best_lasso, best_xgb, best_elastic, best_cat,
        X_train, y_train, X_val, y_val, kf,
    )


    # _____________________________________________________
    #  7. RESULT PLOTS 
    # ─────────────────────────────────────────────────────
    val_results = {
        "Ridge": ridge_val, "Lasso": lasso_val, "catboost":cat_val,
        "GBM": gbm_val,     "XGBoost": xgb_val, "elasticNet":elastic_val,
        "LightGBM": lgb_val, "Stacking": stack_val,
        
    }
    cv_results = {
        "Ridge": ridge_cv, "Lasso": lasso_cv,"elasticNet":elastic_cv,
        "GBM": gbm_cv,     "XGBoost": xgb_cv,"catboost":cat_cv,
        "LightGBM": lgb_cv, "Stacking": stack_cv,
        
    }

    plot_model_comparison(val_results, output_dir=args.output_dir)
    plot_cv_vs_val(cv_results, val_results, output_dir=args.output_dir)


    # _____________________________________________________

    #  8. SAVE MODELS 
    # ──────────────────────────────────────────────────────
    print("-- 8. Saving Models --")
    artifacts = {
        "best_xgb.pkl":    best_xgb,
        "best_lgb.pkl":    best_lgb,
        "best_gbm.pkl":    best_gbm,
        "best_cat.pkl":    best_cat,
        "best_elastic.pkl":    best_elastic,
        "best_lasso.pkl":  best_lasso,
        "best_ridge.pkl":  best_ridge,
        "stack_model.pkl": stack_model,
        "scaler.pkl":      scaler,
        # "rfe.pkl":         rfe,
    }
    save_models(artifacts, models_dir=args.models_dir)


if __name__ == "__main__":
    main()
