
"""
Exploratory Data Analysis 
Univariate / bivariate / multivariate plots and missing-value diagnostics
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew



def missing_summary(df: pd.DataFrame, label: str = "") -> pd.DataFrame:
    """return a DataFrame with count and percentage of missing values"""
    total = df.isna().sum()
    pct   = (df.isna().mean() * 100)
    result = (
        pd.concat([total, pct], keys=["Total", "Percentage"], axis=1)
        .loc[lambda d: d["Percentage"] > 0]
        .sort_values("Percentage", ascending=False)
    )
    result["Percentage"] = result["Percentage"].map(lambda x: f"{x:.2f}%")
    if label:
        print(f"\n── missing value ({label}) ──")
        print(result.to_string())
    return result


def plot_missing_effect(train: pd.DataFrame, output_dir: str = None) -> None:
    """Bar-plot median SalePrice for present vs absent features."""
    feature_with_na = [c for c in train.columns if train[c].isna().sum() > 1]
    fig, axs = plt.subplots(5, 4, figsize=(15, 15))
    fig.patch.set_facecolor("#D5DDE3")
    data = train.copy()
    for col, ax in zip(feature_with_na, axs.flatten()):
        data[col] = data[col].isnull().astype(int)
        data.groupby(col)["SalePrice"].median().plot.bar(
            ax=ax, color=["#64877D", "#727C85"], edgecolor="white", linewidth=0.5
        )
        ax.set_title(col)
    plt.tight_layout()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "missing_effect.png"))
        plt.close()
    else:
        plt.show()


def plot_missing_heatmap(train: pd.DataFrame, output_dir: str = None) -> None:
    """Heatmap of missing value locations."""
    plt.figure(figsize=(18, 7))
    sns.heatmap(train.isnull(), cmap=sns.color_palette(["#34495E", "seagreen"]))
    plt.title("Missing Values Heatmap")
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "missing_heatmap.png"))
        plt.close()
    else:
        plt.show()
"""________________________________________________________

Data type sepration 
__________________________________________________________"""
# years
def get_years_features(df: pd.DataFrame) -> list:
    return [c for c in df.columns if "Yr" in c or "Year" in c]

# discrete
def get_discrete_features(df: pd.DataFrame, numerical: list, years: list) -> list:
    return [ c for c in numerical if len(df[c].unique()) < 15 and c not in years]
# continous 
def get_continuous_features(df: pd.DataFrame, numerical: list, years: list, discrete: list) -> list:
    return [c for c in numerical if c not in years and c not in discrete]


"""_____________________________________________

 Univariate / Bivariate Plots
 ______________________________________________"""


# years plot _______________________
def plot_age_distributions(df: pd.DataFrame, output_dir: str = None) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    for ax, feat in zip(axes, ["HouseAge", "remodAge", "garageAge"]):
        sns.histplot(x=feat, data=df, ax=ax, color="steelblue", kde=True)
        ax.set_title(f"{feat} Distribution")
    plt.tight_layout()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "age_distributions.png"))
        plt.close()
    else:
        plt.show()

# discrete plot _______________________
def plot_discrete_univariate(df: pd.DataFrame, discrete_features: list, output_dir: str = None) -> None:
    fig, ax = plt.subplots(4, 4, figsize=(20, 20))
    for feat, axi in zip(discrete_features, ax.flatten()):
        sns.countplot(x=feat, data=df, ax=axi)
        axi.set_title(feat)
    plt.tight_layout()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "discrete_univariate.png"))
        plt.close()
    else:
        plt.show()

# contious features Plots ___________________
def plot_cont_univ(df: pd.DataFrame, continuous_features: list, output_dir: str = None) -> None:
    """Univariate histograms for all continuous numerical features."""
    fig, ax = plt.subplots(7, 3, figsize=(30, 40))
    for feat, axi in zip(continuous_features, ax.flatten()):
        sns.histplot(x=feat, data=df, ax=axi)
        axi.set_title(feat)
    plt.tight_layout()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "cont_univ.png"))
        plt.close()
    else:
        plt.show()


def plot_cont_biv(df: pd.DataFrame, continuous_features: list, output_dir: str = None) -> None:
    """Bivariate scatter plots of continuous features vs SalePrice."""
    fig, axs = plt.subplots(7, 3, figsize=(20, 30))
    for feat, ax in zip(continuous_features, axs.flatten()):
        sns.scatterplot(
            x=feat, y="SalePrice", hue="SalePrice",
            data=df[[feat, "SalePrice"]].dropna(subset=[feat]),
            ax=ax, palette="viridis_r"
        )
        ax.set_xlabel(feat)
        ax.set_ylabel("SalePrice")
        ax.set_title(f"SalePrice — {feat}")
    plt.tight_layout()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "cont_biv.png"))
        plt.close()
    else:
        plt.show()


def print_skewness(df: pd.DataFrame, continuous_features: list, threshold: float = 0.75) -> None:
    skew_vals = df[continuous_features].apply(skew).sort_values(ascending=False)
    print("\n── Skewed features (|skew| > threshold) ──")
    print(skew_vals[skew_vals.abs() > threshold])



# Categorical Features Plots ___________________________
def plot_univariate_and_bivariate_categorical(train: pd.DataFrame, features: list, color: str, output_dir: str = None, name: str = "categorical") -> None:
    """
    Univariate bar and bivariate box plots for
    categorical (ordinal or nominal) features
    """
    rows = len(features)
    fig, axes = plt.subplots(rows, 2, figsize=(16, rows * 4))
    if rows == 1:
        axes = [axes]
    for i, feat in enumerate(features):
        counts = train[feat].value_counts().sort_index()
        counts.plot(kind="bar", ax=axes[i][0], color=color, edgecolor="black")
        axes[i][0].set_title(f"{feat} Distribution")
        axes[i][0].set_xlabel(feat)
        axes[i][0].set_ylabel("Count")

        train.boxplot(column="SalePrice", by=feat, ax=axes[i][1], grid=False)
        axes[i][1].set_title(f"SalePrice vs {feat}")
        axes[i][1].set_xlabel(feat)
        axes[i][1].set_ylabel("SalePrice")
    plt.suptitle("")
    plt.tight_layout()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"{name}_bivariate.png"))
        plt.close()
    else:
        plt.show()


"""________________________________________________________________________________________________

Multivariate Analysis

______________________________________________________________________________________________"""""

# pairplot
def plot_PairPlot(train: pd.DataFrame, numerical_features: list, n: int = 6, output_dir: str = None) -> None:
    corr        = train[numerical_features].corr()["SalePrice"]
    top_features = corr.abs().sort_values(ascending=False).head(n).index.tolist()
    sns.pairplot(train[top_features])
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "pairplot.png"))
        plt.close()
    else:
        plt.show()

# correlation heatmap
def plot_correlation_heatmap(train: pd.DataFrame, numerical_features: list, output_dir: str = None) -> None:
    plt.subplots(figsize=(30, 20))
    mask = np.zeros_like(train[numerical_features].corr(), dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(
        train[numerical_features].corr(),
        cmap=sns.diverging_palette(20, 220, n=200),
        mask=mask, annot=True, center=0,
    )
    plt.title("Feature Correlation Heatmap")
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
        plt.close()
    else:
        plt.show()

# target distribution
def plot_target_distribution(train: pd.DataFrame, output_dir: str = None) -> None:
    """Plot raw and log-transformed SalePrice."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(x="SalePrice", data=train, color="skyblue", ax=axes[0])
    axes[0].set_title("SalePrice (raw)")
    sns.histplot(x=np.log1p(train["SalePrice"]), color="salmon", ax=axes[1])
    axes[1].set_title("log1p(SalePrice)")
    plt.tight_layout()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "target_distribution.png"))
        plt.close()
    else:
        plt.show()
    print(f"Skew    : {train['SalePrice'].skew():.4f}")
    print(f"Kurtosis: {train['SalePrice'].kurtosis():.4f}")
