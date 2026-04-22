# Property Value Estimator

A robust and comprehensive machine learning pipeline for predicting house prices. This project processes housing data, performs in-depth exploratory data analysis (EDA), engineers features, selects the most predictive variables, and trains a stacked ensemble of advanced regression models to estimate property values accurately.

## 🚀 Features

- **Automated Exploratory Data Analysis (EDA)**: Generates comprehensive plots and statistical summaries including missing value heatmaps, distributions, and correlation matrices.
- **Advanced Feature Engineering**: Handles missing values, performs binning, scales features, removes outliers (manual & Isolation Forest), and applies optimal encoding strategies (Ordinal, Nominal, Target Encoding).
- **Feature Selection**: Utilizes Variance filtering, Correlation filtering, and Recursive Feature Elimination (RFE) to isolate the most impactful predictors.
- **Robust Modeling**: Trains and tunes multiple state-of-the-art machine learning models:
  - Linear Regression, Ridge, Lasso, ElasticNet
  - XGBoost, LightGBM, CatBoost, Gradient Boosting Machine (GBM)
- **Model Stacking**: Combines the best-performing models into a powerful Stacking Regressor to achieve optimal performance.
- **Evaluation & Visualization**: Outputs validation metrics and generates visual comparisons of model performances (CV vs Validation).

## 📁 Project Structure

```text
property-value-estimator/
├── data/                   # Directory containing train.csv, test.csv, sample_submission.csv
├── models/                 # Directory where trained models and scalers are saved
├── notebook/               # Jupyter notebooks for interactive analysis (HOUSE.ipynb)
├── outputs/                # Directory for generated EDA plots and result charts
├── src/                    # Source code modules
│   ├── EDA.py              # Exploratory Data Analysis functions
│   ├── features.py         # Feature engineering and preprocessing logic
│   ├── modeling.py         # Model training, hyperparameter tuning, and stacking
│   ├── selection.py        # Feature selection techniques (Variance, Correlation, Scaling)
│   └── utils.py            # Utility functions (Target encoding, saving models)
├── main.py                 # Main pipeline execution script
└── requirements.txt        # Python package dependencies
```

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd property-value-estimator
   ```

2. **Set up a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

To run the full end-to-end pipeline (Data Loading ➔ EDA ➔ Feature Engineering ➔ Modeling ➔ Evaluation), execute the `main.py` script:

```bash
python main.py
```

### Command Line Arguments

You can customize the pipeline execution using the following arguments:

- `--data_dir`: Directory containing the CSV files (default: `data/`)
- `--models_dir`: Directory to save the trained model artifacts (default: `models/`)
- `--output_dir`: Directory for saving EDA plots and evaluation charts (default: `outputs/`)
- `--skip_eda`: Include this flag to skip the EDA plotting phase (useful for faster execution)

**Example with arguments:**
```bash
python main.py --data_dir data/ --models_dir models/ --output_dir outputs/ --skip_eda
```

## 📦 Output Artifacts

Running the pipeline will produce several artifacts:
- **Plots & Visualizations**: Saved in the `outputs/` directory (e.g., missing value heatmaps, feature distributions, model comparison plots).
- **Trained Models**: Saved as `.pkl` files in the `models/` directory, including the best hyperparameters for Ridge, Lasso, ElasticNet, XGBoost, LightGBM, CatBoost, GBM, and the final Stacking Regressor. The `scaler.pkl` is also saved for future data transformations.

## 📊 Pipeline Stages Explained

1. **Data Loading**: Ingests `train.csv` and `test.csv`.
2. **EDA**: Checks missing data, visualizes discrete/continuous features, prints skewness, and plots correlations.
3. **Feature Engineering**: Log-transforms the target (`SalePrice`), creates new aggregated features (e.g., total square footage, age of property), handles missing data, and categorizes outliers.
4. **Feature Selection**: Drops highly correlated features and zero-variance features to prevent multicollinearity and overfitting.
5. **Modeling**: Applies K-Fold cross-validation to tune hyperparameters for multiple base learners.
6. **Stacking & Evaluation**: Combines base models, evaluates on a validation set using RMSE, and saves the models.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! 
Feel free to check [issues page](<your-issues-url>) if you want to contribute.
