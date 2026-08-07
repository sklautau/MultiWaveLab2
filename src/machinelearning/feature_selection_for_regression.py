'''
For your use case, Leave-One-Subject-Out (LOSO) is the correct validation strategy. Below is a clean reimplementation with the following properties:

Strict LOSO split using participant_id
Scaler fit only on training folds
RFE performed inside each fold (no leakage)
Per-fold metrics + global aggregation
Feature stability analysis (how often each feature is selected)
'''

import argparse

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
)
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.dummy import DummyRegressor
import numpy as np
import pandas as pd
import os
from sklearn.base import clone
from sklearn.feature_selection import (
    RFE,
    SelectKBest,
    mutual_info_regression,
    SequentialFeatureSelector,
)
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from collections import defaultdict
import warnings
import sys

from datasets_util.naming_conventions import DatasetConfig

warnings.filterwarnings('ignore')

INPUT_FEATURES_FILE = "multimodal_features_with_metadata.csv"
OUTPUT_FILE_WITH_SELECTED_FEATURES = "dataset_selected_features_loso.csv"

# Columns that should never be used as ML predictors in this script.
EXCLUDED_FEATURE_COLUMNS = [
    "quality_indicator",
    "ecg_sqi",
]

# ==============================
# 0. Parse arguments
# ==============================
if len(sys.argv) not in (3, 4):
    print(
        "Usage: python feature_selection_loso.py <regressor_index> <n_features> [feature_selection_method]"
    )
    print("feature_selection_method options: rfe (default), sbs, mi")
    sys.exit(1)

classifier_index = int(sys.argv[1])
n_features_to_select = int(sys.argv[2])
feature_selection_method = sys.argv[3].lower() if len(sys.argv) == 4 else "rfe"

if feature_selection_method not in {"rfe", "sbs", "mi"}:
    print(
        f"Invalid feature_selection_method: {feature_selection_method}. "
        "Valid options are: rfe, sbs, mi"
    )
    sys.exit(1)

if True:
    # Create a DatasetConfig instance
    # Create a DatasetConfig instance
    # dataset_config_file = "multimodal_dataset_folders.json"

    parser = argparse.ArgumentParser(
        description="Read and process a JSON configuration file."
    )

    parser.add_argument(
        "json_file",
        metavar="JSON_FILE",
        help="Path to the input JSON file."
    )
    # parse command line
    args = parser.parse_args()

    # Create a DatasetConfig instance
    dataset_config_file = args.json_file

    datasetConfig = DatasetConfig(dataset_config_file)

# dataset_config_file = "multimodal_dataset_folders.json"
# datasetConfig = DatasetConfig(dataset_config_file)


# ==============================
# 1. Define regressors
# ==============================


# =========================================================
# Regressors compatible with RFE
# (provide coef_ or feature_importances_)
# =========================================================

regressors = [

    # Linear models -> coef_
    ("Linear Regression", LinearRegression()),

    ("Ridge", Ridge(
        alpha=1.0,
        random_state=42
    )),

    ("Lasso", Lasso(
        alpha=0.01,
        random_state=42
    )),

    ("ElasticNet", ElasticNet(
        alpha=0.01,
        l1_ratio=0.5,
        random_state=42
    )),

    # SVR linear kernel -> coef_
    ("SVR Linear", SVR(
        kernel='linear'
    )),

    # Tree-based -> feature_importances_
    ("Decision Tree", DecisionTreeRegressor(
        random_state=42,
        max_depth=5
    )),

    ("Random Forest", RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        max_depth=5
    )),

    ("Extra Trees", ExtraTreesRegressor(
        n_estimators=100,
        random_state=42,
        max_depth=5
    )),

    ("Gradient Boosting", GradientBoostingRegressor(
        n_estimators=100,
        random_state=42
    )),

    ("AdaBoost", AdaBoostRegressor(
        n_estimators=100,
        random_state=42
    )),
]

selected_name, selected_estimator = regressors[classifier_index]
print(f"Using regressor: {selected_name}")
print(f"Selecting {n_features_to_select} features")
print(f"Feature selection method: {feature_selection_method.upper()}")

# ==============================
# 2. Load data
# ==============================
input_file_name = os.path.join(
    datasetConfig.get_dataset_machine_learning_path(), INPUT_FEATURES_FILE)
df = pd.read_csv(input_file_name)
print("Reading dataset with shape:", df.shape, "from file", input_file_name)

metadata_cols = ['participant_id', 'session_id', 'datetime', 'GLC']
feature_cols = [
    col for col in df.columns
    if col not in metadata_cols and col not in EXCLUDED_FEATURE_COLUMNS
]

X = df[feature_cols].values
y = df['GLC'].values
# avoid data leakage by using participant_id for LOSO
groups = df['participant_id'].to_numpy()

subjects = np.unique(groups)
print(f"Number of subjects: {len(subjects)}")

# ==============================
# 3. LOSO Cross-Validation
# ==============================
mse_list = []
r2_list = []

feature_selection_counts = defaultdict(int)

for subject in subjects:
    print(f"\nLOSO fold - Test subject: {subject}")

    train_idx = groups != subject
    test_idx = groups == subject

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # --------------------------
    # Normalize (fit ONLY on train)
    # --------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # --------------------------
    # Feature selection on training set
    # --------------------------
    if feature_selection_method == "rfe":
        selector = RFE(
            estimator=clone(selected_estimator),
            n_features_to_select=n_features_to_select,
            step=1
        )
    elif feature_selection_method == "sbs":
        # SBS relies on internal CV at every step; use all cores for speed.
        sbs_cv = KFold(n_splits=5, shuffle=True, random_state=42)
        selector = SequentialFeatureSelector(
            estimator=clone(selected_estimator),
            n_features_to_select=n_features_to_select,
            direction="backward",
            scoring="neg_mean_squared_error",
            cv=sbs_cv,
            n_jobs=-1,
        )
    else:  # feature_selection_method == "mi"
        selector = SelectKBest(
            score_func=mutual_info_regression,
            k=n_features_to_select,
        )

    selector.fit(X_train, y_train)

    selected_mask = selector.get_support()
    selected_features = [feature_cols[i] for i in np.where(selected_mask)[0]]

    # Track feature stability
    for f in selected_features:
        feature_selection_counts[f] += 1

    # --------------------------
    # Train model on selected features
    # --------------------------
    model = clone(selected_estimator)

    X_train_sel = X_train[:, selected_mask]
    X_test_sel = X_test[:, selected_mask]

    model.fit(X_train_sel, y_train)
    y_pred = model.predict(X_test_sel)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MSE: {mse:.4f}, R²: {r2:.4f}")

    mse_list.append(mse)
    r2_list.append(r2)

# ==============================
# 4. Aggregate Results
# ==============================
print("\n==============================")
print("LOSO RESULTS")
print("==============================")

print(f"Mean MSE: {np.mean(mse_list):.4f} ± {np.std(mse_list):.4f}")
print(f"Mean R²: {np.mean(r2_list):.4f} ± {np.std(r2_list):.4f}")

# ==============================
# 5. Feature Stability Ranking
# ==============================
print("\nFeature selection frequency (stability):")

feature_freq = pd.DataFrame({
    'feature': list(feature_selection_counts.keys()),
    'count': list(feature_selection_counts.values())
})

feature_freq['frequency'] = feature_freq['count'] / len(subjects)
feature_freq = feature_freq.sort_values(by='frequency', ascending=False)

print("Best", n_features_to_select, "features by selection frequency:")
print(feature_freq.head(n_features_to_select))

print("\nAll adopted", len(feature_cols), "features:")
# Print column names as an Index object
print(feature_cols)

print("All features by selection frequency:")
print(feature_freq.to_string())

# ==============================
# 6. Build final dataset with selected features
# This selection does not use n_features_to_select specified in command line,
# but rather a stability threshold (e.g., features selected in at least 50% of folds).
# ==============================
# LOSO produces different selected features per fold.
# We define a consensus rule to obtain a single feature subset.
print("\nCreating dataset with selected features...")

# ---- Define selection threshold ----
# Example: keep features selected in at least 50% of folds
selection_threshold = 0.5

stable_features_df = feature_freq[feature_freq['frequency']
                                  >= selection_threshold]
stable_features = stable_features_df['feature'].tolist()

print(
    f"Number of stable features (threshold={selection_threshold}): {len(stable_features)}")

if len(stable_features) == 0:
    raise ValueError(
        "No features passed the stability threshold. Try lowering it.")

# ---- Create final dataframe ----
# Keep metadata + selected features
final_columns = stable_features + \
    ['GLC', 'participant_id', 'session_id', 'datetime']

df_selected = df[final_columns].copy()

print(f"Final dataset shape: {df_selected.shape}")

# ---- Save dataset ----
dataset_output_file = os.path.join(
    datasetConfig.get_dataset_machine_learning_path(), OUTPUT_FILE_WITH_SELECTED_FEATURES)
df_selected.to_csv(dataset_output_file, index=False)

print(f"Saved dataset with selected features to: {dataset_output_file}")
