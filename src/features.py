"""
Feature Engineering 
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


"""─────────────────────────────────────────────
  New features
─────────────────────────────────────────────"""

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add TotalSF, TotalBath, HasGarage, TotalPorchSF , QualDF, TotalBath2 and age features."""

    df = df.copy()
    df["TotalSF"]  = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["TotalBath"] = (
        df["FullBath"] + 0.5 * df["HalfBath"]
        + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"]
    )
    df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
  
    
    df['TotalPorchSF'] = (
        df['OpenPorchSF'] + df['EnclosedPorch'] + df['WoodDeckSF'] + df.get('ScreenPorch', 0) +
        df.get('3SsnPorch', 0)
    )

    # quality × size interaction 
    df['QualSF'] = df['OverallQual'] * df['GrLivArea']

    # bath squared 
    df['TotalBath2'] = df['TotalBath'] ** 2
    return df   

def years_mod(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw year columns to age features."""
    df = df.copy()
    df["HouseAge"]  = df["YrSold"] - df["YearBuilt"] .clip()
    df["garageAge"] = (df["YrSold"] - df["GarageYrBlt"]).fillna(0)
    df["remodAge"]  = df["YrSold"] - df["YearRemodAdd"]
    return df


"""─────────────────────────────────────────────
 Drop
─────────────────────────────────────────────"""

COLUMNS_TO_DROP = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "MasVnrType",
    "PoolArea", "KitchenAbvGr", "Id",
    "3SsnPorch", "MiscVal",
    "ExterCond", "BsmtCond", "BsmtFinType2", "Utilities",
    "GarageQual", "GarageCond", "OverallCond",
    "Street", "Condition2", "RoofMatl", "Exterior2nd",
    # raw year columns (replaced by age features)
    "YearBuilt", "YearRemodAdd", "GarageYrBlt", "YrSold",
]


def drop_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=COLUMNS_TO_DROP, errors="ignore")


"""─────────────────────────────────────────────
 Imputation 
─────────────────────────────────────────────"""

NONE_IMPUTE = [
    "GarageFinish", "GarageType", "BsmtQual",
    "BsmtExposure", "BsmtFinType1", "FireplaceQu",
]

ZERO_IMPUTE = [
    "MasVnrArea", "BsmtFullBath",
    "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF",
    "TotalBsmtSF", "GarageCars", "GarageArea",
    "BsmtHalfBath",
]

MODE_IMPUTE = [
    "Electrical", "MSZoning", "Exterior1st",
    "KitchenQual", "Functional", "SaleType",
]

# impute  :  absence of feature -> "NONE" and missing numeric -> 0 or mode, except LotFrontage (imputed by median of neighborhood)
def impute(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in NONE_IMPUTE:
        if col in df.columns:
            df[col] = df[col].fillna("NONE")
    for col in ZERO_IMPUTE:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    for col in MODE_IMPUTE:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
    if "LotFrontage" in df.columns and "Neighborhood" in df.columns:
        df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
            lambda x: x.fillna(x.median())
        )
        df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())
    return df



"""─────────────────────────────────────────────
 Flagging
─────────────────────────────────────────────"""

FLAG_COLS       = ["MasVnrArea", "BsmtFinSF1", "TotalBsmtSF", "2ndFlrSF",
                   "WoodDeckSF", "OpenPorchSF", 'TotalPorchSF']
FLAG_THEN_DROP  = ["BsmtFinSF2", "EnclosedPorch"]

# flag  create binary indicator for whether feature is present  and drop original feature if it's mostly zeros
def flag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in FLAG_COLS:
        if col in df.columns:
            df[f"{col}_flag"] = (df[col] > 0).astype(int)
    for col in FLAG_THEN_DROP:
        if col in df.columns:
            df[f"{col}flag"] = (df[col] > 0).astype(int)
            df = df.drop(columns=[col])
    return df


"""─────────────────────────────────────────────
 Transformation
─────────────────────────────────────────────"""

LOG_COLS = [
    "GrLivArea", "LotArea", "LotFrontage", "BsmtUnfSF",
    "1stFlrSF", "GarageArea", "MasVnrArea", "BsmtFinSF1",
    "TotalBsmtSF", "2ndFlrSF", "WoodDeckSF", "OpenPorchSF",
    "HouseAge", "remodAge", "garageAge",'TotalPorchSF', 'QualSF'
]

# Log transform skewed numerical features 
def transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in LOG_COLS:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))
    return df


"""─────────────────────────────────────────────
 OUtliers Treatment
─────────────────────────────────────────────"""

def remove_manual_outliers(train: pd.DataFrame) -> pd.DataFrame:
    """Drop the two large cheap houses that confuse tree models."""


    outlier_idx = train[
        (train['GrLivArea'] > np.log1p(4000)) &   
        (train['SalePrice'] < np.log1p(200000))    
    ].index

    train = train.drop(outlier_idx)
    print(f"REMOVED {len(outlier_idx)} ROWS")

    return train 

def remove_isolation_forest_outliers(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    top_features: list,
    contamination: float = 0.02,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    

    clean_feats = [f for f in top_features if f in X_train.columns]
    iso  = IsolationForest(contamination=contamination, random_state=random_state)
    mask = iso.fit_predict(X_train[clean_feats]) == -1
    X_train = X_train[~mask].reset_index(drop=True)
    y_train = y_train[~mask].reset_index(drop=True)
    print(f"IsolationForest removed {mask.sum()} outliers — {len(X_train)} rows remaining")
    return X_train, y_train


"""─────────────────────────────────────────────
 Binning descrete Features 
─────────────────────────────────────────────"""

def bin_discrete(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["OverallQual"]  = df["OverallQual"].replace({1: 3, 2: 3, 10: 9})
    df["BsmtFullBath"] = df["BsmtFullBath"].replace({3: 2})
    df["BsmtHalfBath"] = (df["BsmtHalfBath"] > 0).astype(int)
    df["FullBath"]     = df["FullBath"].replace({0: 1, 4: 3})
    df["HalfBath"]     = (df["HalfBath"] > 0).astype(int)
    df["BedroomAbvGr"] = df["BedroomAbvGr"].replace({0: 1, 6: 5, 7: 5, 8: 5})
    df["TotRmsAbvGrd"] = df["TotRmsAbvGrd"].replace({2: 3, 14: 12})
    df["Fireplaces"]   = df["Fireplaces"].replace({4: 3})
    df["GarageCars"]   = df["GarageCars"].replace({5: 4})
    df["MoSold"] = df["MoSold"].apply(
        lambda x: "high" if x in [4, 5, 6, 7]
        else ("mid" if x in [3, 8, 10, 11] else "low")
    )
    return df


"""─────────────────────────────────────────────
 Ordinal Encodeing 
─────────────────────────────────────────────"""

def ordinal_enc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ExterQual"]   = df["ExterQual"].replace({"Po": "Fa"}).map({"Fa": 1, "TA": 2, "Gd": 3, "Ex": 4}).fillna(2)
    df["BsmtQual"]    = df["BsmtQual"].map({"NONE": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4}).fillna(0)
    df["KitchenQual"] = df["KitchenQual"].replace({"NONE": "TA"}).map({"Fa": 1, "TA": 2, "Gd": 3, "Ex": 4}).fillna(2)
    df["HeatingQC"]   = df["HeatingQC"].replace({"Po": "Fa"}).map({"Fa": 1, "TA": 2, "Gd": 3, "Ex": 4}).fillna(2)
    df["BsmtExposure"]= df["BsmtExposure"].map({"NONE": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}).fillna(0)
    df["BsmtFinType1"]= df["BsmtFinType1"].map({"NONE": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}).fillna(0)
    df["FireplaceQu"] = df["FireplaceQu"].map({"NONE": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}).fillna(0)
    df["GarageFinish"]= df["GarageFinish"].map({"NONE": 0, "Unf": 1, "RFn": 2, "Fin": 3}).fillna(0)
    df["PavedDrive"]  = (df["PavedDrive"] == "Y").astype(int)
    df["LotShape"]    = (df["LotShape"]   == "Reg").astype(int)
    df["LandSlope"]   = (df["LandSlope"]  == "Gtl").astype(int)
    df["Functional"]  = (df["Functional"] == "Typ").astype(int)
    return df


"""─────────────────────────────────────────────
 Nominal Encoding 
─────────────────────────────────────────────"""

def encode_nominal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Binary
    df["LandContour"] = (df["LandContour"] == "Lvl").astype(int)
    df["BldgType"]    = (df["BldgType"]    == "1Fam").astype(int)
    df["RoofStyle"]   = (df["RoofStyle"]   == "Hip").astype(int)
    df["Heating"]     = (df["Heating"]     == "GasA").astype(int)
    df["CentralAir"]  = (df["CentralAir"]  == "Y").astype(int)
    df["Electrical"]  = (df["Electrical"]  == "SBrkr").astype(int)

    # Bin rare levels
    df["MSZoning"]   = df["MSZoning"].replace({"C (all)": "Other", "RH": "Other"})
    df["LotConfig"]  = df["LotConfig"].replace({"FR3": "FR2"})
    df["Condition1"] = df["Condition1"].replace({
        "PosN": "Positive", "PosA": "Positive",
        "RRNn": "Railroad",  "RRAn": "Railroad",
        "RRNe": "Railroad",  "RRAe": "Railroad",
    })
    df["HouseStyle"]  = df["HouseStyle"].replace({"1.5Unf": "Other", "2.5Unf": "Other", "2.5Fin": "Other"})

    keep_ext = ["VinylSd", "HdBoard", "MetalSd", "Wd Sdng", "Plywood", "CemntBd", "BrkFace", "WdShing"]
    df["Exterior1st"] = df["Exterior1st"].apply(lambda x: x if x in keep_ext else "Other")
    df["Foundation"]  = df["Foundation"].replace({"Stone": "Other", "Wood": "Other"})

    keep_gar = ["Attchd", "Detchd", "BuiltIn", "Basment", "NONE"]
    df["GarageType"]    = df["GarageType"].apply(lambda x: x if x in keep_gar else "Other")
    df["SaleType"]      = df["SaleType"].apply(lambda x: x if x in ["WD", "New", "COD"] else "Other")
    df["SaleCondition"] = df["SaleCondition"].apply(lambda x: x if x in ["Normal", "Partial", "Abnorml"] else "Other")

    df["MSSubClass"] = df["MSSubClass"].astype(str)
    keep_sub = ["20", "30", "50", "60", "70", "80", "90", "120", "160"]
    df["MSSubClass"] = df["MSSubClass"].apply(lambda x: x if x in keep_sub else "Other")

    ohe_cols = [
        "MSZoning", "LotConfig", "Condition1", "HouseStyle",
        "Exterior1st", "Foundation", "GarageType", "SaleType",
        "SaleCondition", "MoSold", "MSSubClass",
    ]
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)
    return df



#  run full pipeline on train+test to avoid column mismatch after encoding
# ─────────────────────────────────────────────

def run_ordinal_enc_combined(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Encode ordinal features on train and test jointly to avoid column mismatch"""
    train_size = train.shape[0]
    combined   = pd.concat([train.drop(columns=["SalePrice"]), test], axis=0).reset_index(drop=True)
    combined   = ordinal_enc(combined)
    train_enc  = combined.iloc[:train_size].copy()
    test_enc   = combined.iloc[train_size:].copy()
    train_enc["SalePrice"] = train["SalePrice"].values
    return train_enc, test_enc


def run_nominal_enc_combined(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One-hot encode nominal features on train and test """
    train_size = train.shape[0]
    combined   = pd.concat([train.drop(columns=["SalePrice"]), test], axis=0).reset_index(drop=True)
    combined   = encode_nominal(combined)
    train_enc  = combined.iloc[:train_size].copy()
    test_enc   = combined.iloc[train_size:].copy()
    train_enc["SalePrice"] = train["SalePrice"].values
    return train_enc, test_enc
