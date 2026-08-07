import argparse
import json
import pathlib

from matplotlib.path import Path
import numpy as np
import pandas as pd
from typing import List
import os
import pandas as pd
from sklearn.impute import SimpleImputer
from pathlib import Path

from datasets_util.naming_conventions import LOGIDENTIFIER, DatasetConfig

# maximum percentage of problematic values allowed in a feature column
MAX_PROBLEM_PCT = 10.0


def diagnose_problematic_values(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    rows = []

    for col in feature_cols:
        x = pd.to_numeric(df[col], errors="coerce")

        n_total = len(x)
        n_nan = x.isna().sum()
        n_posinf = np.isposinf(x).sum()
        n_neginf = np.isneginf(x).sum()
        n_inf = n_posinf + n_neginf
        n_problem = n_nan + n_inf

        rows.append({
            "feature": col,
            "n_total": n_total,
            "n_nan_or_missing": n_nan,
            "n_posinf": n_posinf,
            "n_neginf": n_neginf,
            "n_inf": n_inf,
            "n_problematic": n_problem,
            "pct_problematic": 100 * n_problem / n_total,
            "n_valid": n_total - n_problem,
            "pct_valid": 100 * (n_total - n_problem) / n_total,
        })

    report = pd.DataFrame(rows)

    return report.sort_values(
        ["n_problematic", "pct_problematic"],
        ascending=False
    )


def cleaning_and_imputation(features_df: pd.DataFrame, datasetConfig: DatasetConfig, modality: str = "") -> pd.DataFrame:
    '''
    Remove problematic values (NaN, +Inf, -Inf) from the features DataFrame, impute missing values, and return a cleaned DataFrame.
    This function also generates a report of problematic values and saves it to a CSV file.
    Also remove columns with too many problematic values (more than MAX_PROBLEM_PCT).
    Finally, it removes constant columns and adds back metadata columns and the target column.
    '''

    features_file_prefix = datasetConfig.get_value("FEATURES_FILE_PREFIX")

    metadata_columns = datasetConfig.get_chosen_metadata_columns()
    segments_columns = datasetConfig.get_segments_columns()
    # concatenate metadata_columns and segments_columns, removing duplicates
    all_metadata_columns = list(set(metadata_columns + segments_columns))

    # ======================================================
    # Select feature columns, excluding any meta-information columns
    # ======================================================
    feature_cols = [
        c for c in features_df.columns
        if c not in all_metadata_columns
    ]
    # ======================================================
    # Select non-feature columns, including any meta-information columns that are in the DataFrame
    # ======================================================
    non_feature_cols = [
        c for c in features_df.columns
        if c in all_metadata_columns
    ]

    print(f"Dropped {len(non_feature_cols)} meta-information columns")
    print(f"Initial number of column features = {len(feature_cols)}")

    problem_report = diagnose_problematic_values(features_df, feature_cols)

    print(problem_report.head(10))

    # save report file
    output_folder = os.path.join(
        datasetConfig.features_path, features_file_prefix +
        "_stats")
    print("output_folder = ", output_folder)
    # create output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)
    report_file_path = os.path.join(
        output_folder, features_file_prefix +
        os.path.basename(datasetConfig.get_folder_and_prefix_from_json_config_file_name()) +
        "_problematic_values_report_" + modality + ".csv")
    print("report_file_path = ", report_file_path)
    problem_report.to_csv(report_file_path, index=False)
    print(
        f"Saved problematic values report to '{report_file_path}'")

    # Convert feature columns to numeric, coercing errors to NaN
    # This will convert any non-numeric values to NaN, which can then be handled
    # If a value cannot be converted to a number, Pandas replaces it with NaN.
    # target = GLC is not included in feature_cols
    X = features_df[feature_cols].apply(pd.to_numeric, errors="coerce")

    # Replace +Inf and -Inf with NaN
    X = X.replace([np.inf, -np.inf], np.nan)

    # drop columns with too many problematic values
    cols_to_drop_due_to_nan_infs = problem_report.loc[
        problem_report["pct_problematic"] > MAX_PROBLEM_PCT,
        "feature"
    ].tolist()
    X = X.drop(columns=cols_to_drop_due_to_nan_infs)

    # remaining features after dropping columns with too many problematic values
    features_without_inf_nan_and_imputed = X.columns.tolist()

    print(LOGIDENTIFIER +
          f"Dropped {len(cols_to_drop_due_to_nan_infs)} columns with more than {MAX_PROBLEM_PCT}% problematic values:")
    print(LOGIDENTIFIER + "Dropped columns:")
    print(cols_to_drop_due_to_nan_infs)

    # Impute remaining missing values
    imputer = SimpleImputer(strategy="median")

    # imputer.fit_transform(X) looses the original column names and index, so we need to create a new DataFrame with the same columns and index
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X),
        columns=features_without_inf_nan_and_imputed,
        index=X.index,
    )

    # Now remove constant or near-constant columns
    std_values = X_imputed.std(axis=0)
    constant_cols = std_values[std_values == 0].index.tolist()
    if len(constant_cols) > 0:
        print(LOGIDENTIFIER +
              f"Dropped {len(constant_cols)} constant columns:")
        print(constant_cols)

    X_imputed = X_imputed.drop(columns=constant_cols)
    print("Kept columns:")
    print(X_imputed.columns.tolist())
    print(LOGIDENTIFIER + "Number of modality features after cleaning and imputation =",
          len(X_imputed.columns.tolist()))

    # Now we add back all metadata columns that are present in the original DataFrame, and the target column
    X_imputed[non_feature_cols] = features_df[non_feature_cols]

    return X_imputed


def old_remove_missing_data(
    df: pd.DataFrame,
    drop_rows: bool = True,
    drop_cols: bool = False,
    inplace: bool = False
) -> pd.DataFrame:
    """
    Remove rows and/or columns containing any missing values.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    drop_rows : bool
        If True, drop rows with any NaN
    drop_cols : bool
        If True, drop columns with any NaN
    inplace : bool
        If True, modify df in place

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame (or None if inplace=True)
    """
    target = df if inplace else df.copy()

    if drop_rows:
        # drop rows with inf or NaN values
        target.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
        target.dropna(axis=0, how="any", inplace=True)

    if drop_cols:
        # drop columns with inf or NaN values
        target.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
        target.dropna(axis=1, how="any", inplace=True)

    return target


def old_find_duplicate_columns(df: pd.DataFrame) -> List[str]:
    """
    Returns a list of column names that appear more than once in the DataFrame.
    """
    cols = pd.Series(df.columns)
    duplicates = cols[cols.duplicated()].unique().tolist()
    return duplicates


def old_duplicate_column_counts(df: pd.DataFrame) -> pd.Series:
    """
    Returns a Series with counts of duplicated column names (only those >1).
    """
    counts = pd.Series(df.columns).value_counts()
    return counts[counts > 1]


def old_main():

    # Create a DatasetConfig instance
    dataset_config_file = "multimodal_dataset_folders.json"
    datasetConfig = DatasetConfig(dataset_config_file)

    features_folder = datasetConfig.get_dataset_features_path()
    features_file_prefix = datasetConfig.get_value("FEATURES_FILE_PREFIX")
    # train_test_prefix = datasetConfig.get_value("TRAIN_TEST_SPLIT_PREFIX")

    dataset_name = datasetConfig.get_value("DATASET_NAME")
    modalities = datasetConfig.modalities

    input_file = os.path.join(features_folder, features_file_prefix + ".csv")
    # read input file
    df = pd.read_csv(input_file)
    dups = old_find_duplicate_columns(df)

    if dups:
        print("Duplicate columns found:", dups)
    else:
        print("No duplicate columns.")

    print(df.head())
    # drop columns called has_ppg,ppg_error, etc.
    # Use errors='ignore' to safely drop columns that may not exist
    df = df.drop(columns=["has_ppg", "has_ecg",
                          "ppg_error", "has_bioimp",
                          "file_id", "modality", "segment_id"], errors='ignore')

    print(df.head())

    df_clean = old_remove_missing_data(df, drop_rows=False, drop_cols=True)
    # df_clean = remove_missing_data(df, drop_rows=True, drop_cols=False)
    print("Input DataFrame shape:", df.shape)
    print("Cleaned DataFrame shape:", df_clean.shape)

    # check whether NaN exists in cleaned DataFrame
    if df_clean.isna().any().any():
        print("Warning: Cleaned DataFrame still contains NaN values.")
    else:
        print("No NaN values in cleaned DataFrame.")

    # check whether inf exists in cleaned DataFrame
    if df_clean.isin([float('inf'), float('-inf')]).any().any():
        print("Warning: Cleaned DataFrame still contains inf values.")
    else:
        print("No inf values in cleaned DataFrame.")

    # save to file
    #
    output_file = os.path.join(
        features_folder, features_file_prefix + "_clean.csv")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_clean.to_csv(output_file, index=False)

    print(f"Cleaned feature dataset saved to: {output_file}")


if __name__ == "__main__":
    # old_main()
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

    datasetConfig = DatasetConfig(args.json_file)

    dataset_name = datasetConfig.get_value("DATASET_NAME")
    if dataset_name == "ieb1":
        print("Using IEB-1 dataset configuration.")
        # use an empty modality when dealing with IEB1 dataset,
        # because it has only the PPG
        modalities = []
        features_file_prefix = datasetConfig.get_value(
            "FEATURES_FILE_PREFIX") + "_ppg"
    elif dataset_name == "ieb2":
        print("Using IEB-2 dataset configuration.")
        # For IEB2, choose among modalities = ["ppg_", "ecg_", "bioimp_"]
        modalities = ["ppg_", "ecg_", "bioimp_"]
        features_file_prefix = datasetConfig.get_value(
            "FEATURES_FILE_PREFIX")
    elif dataset_name == "ieb3":
        raise Exception(
            "IEB-3 dataset configuration is not supported in this script.")
    else:
        raise Exception("Invalid dataset name = " + str(dataset_name))

    features_path = datasetConfig.get_dataset_features_path()
    features_path = Path(features_path) / (features_file_prefix + ".csv")
    # read features_path into a DataFrame
    df = pd.read_csv(features_path)

    df_clean = cleaning_and_imputation(df, datasetConfig, modality="")

    df_clean.to_csv(features_path.parent /
                    (features_file_prefix + "_clean_test.csv"), index=False)
