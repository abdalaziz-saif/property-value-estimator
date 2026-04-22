"""
Utility module.
Model persistence, submission generation, and target encoding
"""

import os
import shutil
import numpy as np
import pandas as pd
import joblib



#  Target encoding
# ─────────────────────────────────────────────

def target_encode(X_train: pd.DataFrame,X_val: pd.DataFrame,test: pd.DataFrame,y_train: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Encode Neighborhood using target mean from training fold only.
    Unseen levels fall back to the global mean.
    """
    neighborhood_means = y_train.groupby(X_train["Neighborhood"]).mean()
    global_mean        = y_train.mean()

    X_train = X_train.copy()
    X_val   = X_val.copy()
    test    = test.copy()
    
    # Handle unseen neighborhood by filling it with the global mean 
    X_train["Neighborhood"] = X_train["Neighborhood"].map(neighborhood_means).fillna(global_mean)
    X_val["Neighborhood"]   = X_val["Neighborhood"].map(neighborhood_means).fillna(global_mean)
    test["Neighborhood"]    = test["Neighborhood"].map(neighborhood_means).fillna(global_mean)
    return X_train, X_val, test



#  Save models
# ─────────────────────────────────────────────

def save_models(artifacts: dict, models_dir: str = "models") -> None:
    """
    Save a dict of {filename: object} to models_dir.
    Also zips the directory.
    """
    os.makedirs(models_dir, exist_ok=True)
    for name, obj in artifacts.items():
        path = os.path.join(models_dir, name)
        joblib.dump(obj, path)
        print(f"Saved → {path}")
        
    zip_path = models_dir.rstrip("/").rstrip("\\")
    shutil.make_archive(zip_path, "zip", models_dir)
    print(f"Archive → {zip_path}.zip")


#  load models
# ─────────────────────────────────────────────

def load_models(names: list, models_dir: str = "models") -> dict:
    """Load saved model 
    return dict """
    loaded = {}
    for name in names:
        path = os.path.join(models_dir, name)
        loaded[name] = joblib.load(path)
        print(f"Loaded ← {path}")
    return loaded


