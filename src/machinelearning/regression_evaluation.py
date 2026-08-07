'''
This module implements grouped nested cross-validation for regression
tasks, including model selection, hyperparameter tuning, and evaluation
metrics.
It supports multiple regression models.
Choosing the model and its hyperparameters is done solely using
train + validation sets. The test set is only used for final
evaluation of the best (refitted to train+validation) model.
'''
from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np
import pandas as pd

from datasets_util.naming_conventions import LOGIDENTIFIER, DatasetConfig

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV, GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, Matern, WhiteKernel

# Models
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.dummy import DummyRegressor

from segments.segments_core import DEBUGGING
from datasets_util.to_latex import to_latex_CV_results_table
from models_definition import ModelConfigDict, get_model_configs

INCLUDE_MODELS_THAT_TAKE_LONG_TIME = False


# -----------------------------
# Metrics
# -----------------------------


def regression_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> dict[str, float]:
    """Compute common regression metrics for predictions.

    Parameters
    ----------
    y_true : np.ndarray | pd.Series
        Ground-truth target values.
    y_pred : np.ndarray | pd.Series
        Predicted target values.

    Returns
    -------
    dict[str, float]
        Dictionary with RMSE, MAE, and R2 values.
    """
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


# -----------------------------
# Nested CV
# -----------------------------

def nested_cv_grouped(
    df: pd.DataFrame,
    target_col: str,
    group_col: str = "participant_id",
    outer_splits: int = 5,
    inner_splits: int = 3,
) -> pd.DataFrame:
    """Run grouped nested cross-validation across candidate regressors.

    The outer GroupKFold estimates generalization performance, while
    the inner GroupKFold tunes hyperparameters with GridSearchCV.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing features, target, and group column.
    target_col : str
        Name of the target column.
    group_col : str, optional
        Name of the grouping column used to avoid leakage across folds.
    outer_splits : int, optional
        Number of outer CV folds.
    inner_splits : int, optional
        Number of inner CV folds.

    Returns
    -------
    pd.DataFrame
        Aggregated metrics per model, sorted by mean RMSE.
    """

    X = df.drop(columns=[target_col, group_col])
    y = df[target_col]
    groups = df[group_col]

    outer_cv = GroupKFold(n_splits=outer_splits)
    inner_cv = GroupKFold(n_splits=inner_splits)

    configs = get_model_configs()

    results = []

    for model_name, config in configs.items():
        print(f"\n=== Model: {model_name} ===")

        fold_metrics = []
        train_fold_metrics = []

        try:
            for fold_idx, (train_idx, test_idx) in enumerate(
                outer_cv.split(X, y, groups)
            ):
                print(f"Outer fold {fold_idx+1}")

                # inform the list of participant_id's that are part of the test set for this fold
                test_participants = groups.iloc[test_idx].unique()
                print(
                    f"Test participants for fold {fold_idx+1}: {test_participants}")

                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                groups_train = groups.iloc[train_idx]

                # Scikit-learn requires scoring functions to follow the rule:
                # "higher is better" (maximization). So, we use "neg_root_mean_squared_error" to minimize RMSE.
                grid = GridSearchCV(
                    estimator=config["pipeline"],
                    param_grid=config["params"],
                    cv=inner_cv,
                    scoring="neg_root_mean_squared_error",
                    n_jobs=-1,
                    return_train_score=True
                )

                # CRITICAL: pass groups to inner CV
                grid.fit(X_train, y_train, groups=groups_train)

                fitted_model = grid.best_estimator_
                best_idx = grid.best_index_

                inner_train_rmse = - \
                    grid.cv_results_["mean_train_score"][best_idx]
                inner_valid_rmse = - \
                    grid.cv_results_["mean_test_score"][best_idx]

                inner_train_rmse_std = grid.cv_results_[
                    "std_train_score"][best_idx]
                inner_valid_rmse_std = grid.cv_results_[
                    "std_test_score"][best_idx]

                print(
                    f"Inner train RMSE: {inner_train_rmse:.3f} ± {inner_train_rmse_std:.3f}")
                print(
                    f"Inner validation RMSE: {inner_valid_rmse:.3f} ± {inner_valid_rmse_std:.3f}")

                # inform the best hyperparameters for this fold
                print(
                    f"Best hyperparameters for fold {fold_idx+1}: {grid.best_params_}")

                # inform the performance collected by grid object on the inner CV folds
                print(
                    f"Performance on inner CV folds for fold {fold_idx+1}: negative RMSE = {grid.best_score_}")

                y_pred = fitted_model.predict(X_test)
                metrics = regression_metrics(y_test, y_pred)
                print("Test metrics on outer test fold: ", metrics)

                # evaluate the model on the training set as well to check for overfitting
                y_train_pred = fitted_model.predict(X_train)
                train_metrics = regression_metrics(y_train, y_train_pred)
                print("Training metrics on outer test fold: ", train_metrics)

                fold_metrics.append(metrics)
                train_fold_metrics.append(train_metrics)

                fold_result = {
                    "model": model_name,
                    "fold": fold_idx + 1,
                    "best_params": grid.best_params_,

                    "inner_train_rmse": inner_train_rmse,
                    "inner_valid_rmse": inner_valid_rmse,

                    "outer_train_rmse": train_metrics["rmse"],
                    "outer_test_rmse": metrics["rmse"],

                    "outer_train_mae": train_metrics["mae"],
                    "outer_test_mae": metrics["mae"],

                    "outer_train_r2": train_metrics["r2"],
                    "outer_test_r2": metrics["r2"],
                }
                print(f"Fold {fold_idx+1} results: {fold_result}")

            results.append({
                "model": model_name,
                "status": "ok",
                "error": np.nan,
                "rmse_mean": np.mean([m["rmse"] for m in fold_metrics]),
                "rmse_std": np.std([m["rmse"] for m in fold_metrics]),
                "mae_mean": np.mean([m["mae"] for m in fold_metrics]),
                "mae_std": np.std([m["mae"] for m in fold_metrics]),
                "r2_mean": np.mean([m["r2"] for m in fold_metrics]),
                "r2_std": np.std([m["r2"] for m in fold_metrics]),
                "train_rmse_mean": np.mean([m["rmse"] for m in train_fold_metrics]),
                "train_rmse_std": np.std([m["rmse"] for m in train_fold_metrics]),
                "train_mae_mean": np.mean([m["mae"] for m in train_fold_metrics]),
                "train_mae_std": np.std([m["mae"] for m in train_fold_metrics]),
                "train_r2_mean": np.mean([m["r2"] for m in train_fold_metrics]),
                "train_r2_std": np.std([m["r2"] for m in train_fold_metrics])
            })
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            print(
                f"Skipping model {model_name} due to failure during nested CV: {error_message}")
            results.append({
                "model": model_name,
                "status": "failed",
                "error": error_message,
                "rmse_mean": np.nan,
                "rmse_std": np.nan,
                "mae_mean": np.nan,
                "mae_std": np.nan,
                "r2_mean": np.nan,
                "r2_std": np.nan,
                "train_rmse_mean": np.nan,
                "train_rmse_std": np.nan,
                "train_mae_mean": np.nan,
                "train_mae_std": np.nan,
                "train_r2_mean": np.nan,
                "train_r2_std": np.nan,
            })

    return pd.DataFrame(results).sort_values(by="rmse_mean")


def old_choose_feature_modalities(df: pd.DataFrame, modalities: list, always_keep=["GLC", "participant_id"]):
    """Keep only feature columns matching selected modality prefixes.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with feature and metadata columns.
    modalities : list
        Prefixes used to select feature columns (e.g., ["ppg_", "ecg_"]).
    always_keep : list, optional
        Columns that should be preserved even if they do not match a prefix.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe with selected feature columns.
    """
    cols_to_keep = [
        c for c in df.columns
        if c.startswith(tuple(modalities)) or c in always_keep
    ]

    df_filtered = df[cols_to_keep]

    # inform the columns that were dropped
    dropped_cols = set(df.columns) - set(cols_to_keep)
    print(f"Dropped columns: {dropped_cols}")

    print(f"Remaining columns: {df_filtered.columns}")

    return df_filtered


def remove_metadata_columns_but_id_and_glc(df: pd.DataFrame,
                                           datasetConfig: DatasetConfig) -> pd.DataFrame:
    """Drop non-feature metadata columns while keeping grouping
    information (participant_id) and target (GLC).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe without session and datetime columns.
    """
    metadata_columns = datasetConfig.get_chosen_metadata_columns()
    segments_columns = datasetConfig.get_modality_expanded_segments_columns()
    # concatenate metadata_columns and segments_columns, removing duplicates
    all_metadata_columns = list(set(metadata_columns + segments_columns))
    # remove participant_id from the list of columns to drop, since we want to keep it for grouping
    if "participant_id" in all_metadata_columns:
        all_metadata_columns.remove("participant_id")
    if "GLC" in all_metadata_columns:
        all_metadata_columns.remove("GLC")

    existing_columns = [
        col for col in all_metadata_columns if col in df.columns]

    return df.drop(columns=existing_columns)


def train_with_train_validation_sets(
    results: pd.DataFrame,
    train_df: pd.DataFrame,
    target_col: str,
    inner_splits: int = 3,
) -> tuple[BaseEstimator, str]:
    """Refit the best model from nested CV on train+validation data.

    Parameters
    ----------
    results : pd.DataFrame
        Nested CV summary where the top row contains the best model name.
    train_df : pd.DataFrame
        Combined training and validation dataframe.
    target_col : str
        Name of the target column.
    inner_splits : int, optional
        Number of grouped folds used for the refit GridSearchCV.

    Returns
    -------
    BaseEstimator
        Best estimator fitted on the full train+validation set.
    """
    # recover configuration from CV results:
    # nested_cv_grouped function only returns aggregated metrics, not the fitted estimators nor their
    # selected hyperparameters.
    # We will refit a GridSearchCV on the full training set (train + validation), using the same grouped
    # CV strategy, and then extract best_estimator_.
    best_name = str(results.iloc[0]["model"])
    configs = get_model_configs()
    config = configs[best_name]

    X_full = train_df.drop(columns=[target_col, "participant_id"])
    y_full = train_df[target_col]
    groups_full = train_df["participant_id"]

    inner_cv = GroupKFold(n_splits=inner_splits)

    grid = GridSearchCV(
        estimator=config["pipeline"],
        param_grid=config["params"],
        cv=inner_cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        return_train_score=True
    )

    grid.fit(X_full, y_full, groups=groups_full)

    fitted_model = grid.best_estimator_
    best_idx = grid.best_index_
    if DEBUGGING:
        print("All keys:", grid.cv_results_.keys())
        # print the values of all keys:
        for key in grid.cv_results_.keys():
            print(f"{key}: {grid.cv_results_[key]}")
        # inner_train_rmse = -grid.cv_results_["mean_train_score"][best_idx]
        # inner_valid_rmse = -grid.cv_results_["mean_test_score"][best_idx]

    inner_train_rmse_std = grid.cv_results_[
        "std_train_score"][best_idx]
    inner_valid_rmse_std = grid.cv_results_["std_test_score"][best_idx]

    # print(
    #    f"Inner train RMSE: {inner_train_rmse:.3f} ± {inner_train_rmse_std:.3f}")
    # print(
    #    f"Inner validation RMSE: {inner_valid_rmse:.3f} ± {inner_valid_rmse_std:.3f}")

    # inform the best hyperparameters
    print(LOGIDENTIFIER +
          f"Best hyperparameters {grid.best_params_}")

    # inform the performance collected by grid object on the inner CV folds
    print(LOGIDENTIFIER +
          f"Performance on CV folds: negative RMSE = {grid.best_score_}")

    print(LOGIDENTIFIER + f"Best model: {best_name}")
    print(LOGIDENTIFIER + f"Best hyperparameters: {grid.best_params_}")

    return fitted_model, best_name


def regression_evaluation(datasetConfig: DatasetConfig) -> None:
    """
    Execute the end-to-end grouped regression evaluation workflow.

    This routine loads train/validation/test splits organized as
    CSV files with features, selects feature
    modalities, performs grouped nested cross-validation, refits the
    best model on train+validation data, evaluates on test data, and
    reports a dummy baseline for comparison.

    Parameters
    ----------
    modalities : list[str]
        Feature prefixes to include in the evaluation.
    """
    np.random.seed(42)

    modalities = datasetConfig.modalities

    # read the sets from csv files:
    train_csv = datasetConfig.get_selected_features_file_name(
        train_or_test="train")
    test_csv = datasetConfig.get_selected_features_file_name(
        train_or_test="test")
    validation_csv = datasetConfig.get_selected_features_file_name(
        train_or_test="validation")

    # read the files
    print(f"Reading train set from: {train_csv}")
    train_set = pd.read_csv(train_csv)

    # check if the validation file exists and is not empty
    is_there_validation = False
    if os.path.exists(validation_csv) and os.path.getsize(validation_csv) > 0:
        print(f"Reading validation set from: {validation_csv}")
        validation_set = pd.read_csv(validation_csv, header=0)
        is_there_validation = True

    # remove metadata but keep participant_id and GLC columns
    train_set = remove_metadata_columns_but_id_and_glc(
        train_set, datasetConfig)
    if is_there_validation:
        validation_set = remove_metadata_columns_but_id_and_glc(
            validation_set, datasetConfig)

    # normalization is done inside the pipeline, so we don't need to normalize here

    if False:  # disable, manipulate dataframes outside this code
        # keep only the columns (features) whose names start with the prefixes provided in modalities
        if len(modalities) > 0:
            train_set = choose_feature_modalities(train_set, modalities)
            test_set = choose_feature_modalities(test_set, modalities)
            if is_there_validation:
                validation_set = choose_feature_modalities(
                    validation_set, modalities)

    if is_there_validation:
        # put together train and validation sets
        train_df = pd.concat([train_set, validation_set])
    else:
        train_df = train_set

    print(LOGIDENTIFIER +
          f"Train+validation set shape for CV (metadata is only GLC and ID): {train_df.shape}")
    print(LOGIDENTIFIER +
          f"Number of feature (not metadata) columns: {train_df.shape[1] - 2}")

    target_col = "GLC"
    results = nested_cv_grouped(
        train_df, group_col="participant_id",
        target_col=target_col, outer_splits=5, inner_splits=3)

    print("\n=== Nested CV Results ===")
    print(results)
    # get the filename from complete Path
    prefix_for_results_file = datasetConfig.get_folder_and_prefix_from_json_config_file_name()
    prefix_for_results_file = os.path.basename(prefix_for_results_file)
    # add the dataset name to the prefix for the results file
    prefix_for_results_file = datasetConfig.get_value(
        "DATASET_NAME") + "_" + prefix_for_results_file
    file_name = os.path.join(
        datasetConfig.get_dataset_machine_learning_path(),  prefix_for_results_file + "_nested_cv_results.csv")
    # fix \ and / in the file name
    file_name = os.path.normpath(file_name)
    print(f"Saving nested CV results to: {file_name}")
    results.to_csv(file_name, index=False)

    # convert to Latex:
    latex = to_latex_CV_results_table(
        input_file_name=file_name,
        # train_columns=["train_rmse", "train_mae", "train_r2"],
        train_columns=["train_rmse", "train_mae"],
        # test_columns=["rmse", "mae", "r2"],
        test_columns=["rmse", "mae"],
        table_title=f"Nested CV results for {prefix_for_results_file}"
    )
    print(latex)

    # Train the regressor: results only name the best model type; refit with the same
    # grouped GridSearchCV as nested_cv_grouped to recover hyperparameters.
    # create an instance of this object with the given hyperparameters
    print("\n=== Refit best model on train+validation sets ===")
    refitted_model, name_of_best_model = train_with_train_validation_sets(
        results, train_df, target_col=target_col, inner_splits=3
    )

    print("refitted_model:", refitted_model)

    # Test the classifier
    print(f"Reading test set from: {test_csv}")
    test_set = pd.read_csv(test_csv)
    # remove metadata but keep participant_id and GLC columns
    test_set = remove_metadata_columns_but_id_and_glc(test_set, datasetConfig)

    # Show the participants in test set
    print("Testing with participants: ", np.unique(test_set["participant_id"]))
    # now we drop the participant_id column
    X_test = test_set.drop(columns=[target_col, "participant_id"])

    print("Columns in test set:", X_test.columns)

    y_pred = refitted_model.predict(X_test)
    metrics = regression_metrics(test_set[target_col], y_pred)
    print(f"Test set metrics (best model): {metrics}")

    # Baseline for comparison: ignore features, predict training-set mean (dummy regressor)
    X_train_full = train_df.drop(columns=[target_col, "participant_id"])
    y_train_full = train_df[target_col]
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_train_full, y_train_full)
    y_dummy = dummy.predict(X_test)
    dummy_metrics = regression_metrics(test_set[target_col], y_dummy)
    print(f"Test set metrics (DummyRegressor mean baseline): {dummy_metrics}")

    # print the metrics in a more readable format with 2 decimal places
    print("\n=== Test Set Metrics for Latex: ===")
    print(
        f"& {name_of_best_model} & {metrics['rmse']:.2f} & {metrics['mae']:.2f} & {dummy_metrics['rmse']:.2f} & {dummy_metrics['mae']:.2f} \\\\")


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
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
    datasetConfig = DatasetConfig(args.json_file)

    # execute regression evaluation
    regression_evaluation(datasetConfig)
