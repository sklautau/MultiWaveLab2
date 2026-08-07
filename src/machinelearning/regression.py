import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from sklearn.dummy import DummyRegressor

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
)

from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)

from datasets_util.naming_conventions import DatasetConfig

warnings.filterwarnings("ignore")

INPUT_FEATURES_FILE = "multimodal_features_with_metadata.csv"


# =========================================================
# Regressors
# =========================================================
regressors = [

    # -----------------------------------------------------
    # Dummy baselines
    # -----------------------------------------------------
    ("Dummy Mean", DummyRegressor(strategy="mean")),

    ("Dummy Median", DummyRegressor(strategy="median")),

    # -----------------------------------------------------
    # Linear models
    # -----------------------------------------------------
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

    # -----------------------------------------------------
    # SVR
    # -----------------------------------------------------
    ("SVR Linear", SVR(kernel="linear")),

    ("SVR RBF", SVR(kernel="rbf")),

    # -----------------------------------------------------
    # KNN
    # -----------------------------------------------------
    ("KNN-5", KNeighborsRegressor(
        n_neighbors=5
    )),

    # -----------------------------------------------------
    # Trees
    # -----------------------------------------------------
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


def main():

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

    input_file_name = os.path.join(
        datasetConfig.get_dataset_machine_learning_path(),
        INPUT_FEATURES_FILE
    )

    df = pd.read_csv(input_file_name)

    metadata_cols = [
        "participant_id",
        "session_id",
        "datetime",
        "GLC",
    ]

    feature_cols = [
        col for col in df.columns
        if col not in metadata_cols
    ]

    X = df[feature_cols].values
    y = df["GLC"].values
    groups = df["participant_id"].values

    subjects = np.unique(groups)

    print(f"Number of subjects: {len(subjects)}")
    print(f"Number of features: {len(feature_cols)}")
    print(f"Dataset shape: {X.shape}")

    all_results = []

    for regressor_name, regressor in regressors:
        result = evaluate_regressor(
            regressor_name,
            regressor,
            X,
            y,
            groups,
            subjects
        )

        all_results.append(result)

    results_df = pd.DataFrame(all_results)

    results_df = results_df.sort_values(
        by="global_rmse",
        ascending=True
    )

    print("\n" + "=" * 70)
    print("SUMMARY OF ALL REGRESSORS")
    print("=" * 70)

    print(results_df)

    output_file = os.path.join(
        datasetConfig.get_dataset_machine_learning_path(),
        "regression_loso_all_regressors_results.csv"
    )

    results_df.to_csv(output_file, index=False)

    print(f"\nSaved results to: {output_file}")


def evaluate_regressor(regressor_name, regressor, X, y, groups, subjects):
    print("\n" + "=" * 70)
    print(f"Evaluating regressor: {regressor_name}")
    print("=" * 70)

    mse_list = []
    rmse_list = []
    mae_list = []
    r2_list = []

    all_predictions = []
    all_targets = []

    for subject in subjects:
        print(f"\nLOSO fold - Test subject: {subject}")

        train_idx = groups != subject
        test_idx = groups == subject

        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = clone(regressor)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

        mse_list.append(mse)
        rmse_list.append(rmse)
        mae_list.append(mae)
        r2_list.append(r2)

        all_predictions.extend(y_pred)
        all_targets.extend(y_test)

    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    global_mse = mean_squared_error(all_targets, all_predictions)
    global_rmse = np.sqrt(global_mse)
    global_mae = mean_absolute_error(all_targets, all_predictions)
    global_r2 = r2_score(all_targets, all_predictions)

    result = {
        "regressor": regressor_name,
        "mean_mse": np.mean(mse_list),
        "std_mse": np.std(mse_list),
        "mean_rmse": np.mean(rmse_list),
        "std_rmse": np.std(rmse_list),
        "mean_mae": np.mean(mae_list),
        "std_mae": np.std(mae_list),
        "mean_r2": np.mean(r2_list),
        "std_r2": np.std(r2_list),
        "global_mse": global_mse,
        "global_rmse": global_rmse,
        "global_mae": global_mae,
        "global_r2": global_r2,
    }

    return result


if __name__ == "__main__":
    main()
