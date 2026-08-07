'''
Concatenate multimodality features into a single machine learning dataset.

1) First, using the procedure of clean_features_dataframe.py,
clean the features DataFrame by removing rows and columns with NaN
or inf values. In the end, this will define the columns that will be used
in the output concatenated DataFrame.
2) Then use the following strategy to concatenate features creating
a multimodal row, considering that number of features may differ among
the modalities and that some features may be missing for some files:
    2.1) For each group of participant_id, session_id, time_date, count how
    many features are available for each modality. The largest number Nmax will
    dictate how many rows will be created for that group. For example, if for
    a given group there are Nmax=6 features for PPG, 3 for ECG and 1 for bioimpedance,
    then Nmax=6 rows will be created for that group, with the available features for each modality.
    2.2) Create each row such that all modalities contribute with Nmax features,
    by repeating features as needed. For example, if for a given group there are Nmax=6 features for PPG, 3 for ECG and 1 for bioimpedance, then the 3 ECG features will be repeated twice and the 1 bioimpedance
    feature will be repeated 6 times to create 6 rows for that group.
    2.3) Count and indicate to stdout the case in which a group has no features for a modality, and
    skip that group, since it will not contribute to the machine learning model.
3) Save a single DataFrame csv with concatenated features and all available metadata (participant_id, session_id, time_date, etc.) for all files.
4) Save another DataFrame csv with only the columns needed for machine learning, such as X, y, where y is the glucose GLC.
'''

import os
import pandas as pd
from typing import List, Tuple
from datasets_util.naming_conventions import DatasetConfig
from features.clean_features_dataframe import old_remove_missing_data

OUTPUT_MULTIMODAL_FEATURES_FILE = "multimodal_features_with_metadata.csv"
OUTPUT_MULTIMODAL_WITHOUT_METADATA_FILE = "multimodal_features_ml.csv"

# Columns to exclude specifically from the ML-ready CSV.
ML_EXCLUDED_COLUMNS = [
    "ecg_sqi",
    "quality_indicator",
]

# Metadata columns to preserve in output
SOME_METADATA_COLUMNS = [
    "participant_id",
    "session_id",
    "datetime",
    "GLC"
]


def clean_by_removing_nan(df: pd.DataFrame, modality: str) -> pd.DataFrame:
    """
    Clean a modality-specific dataframe by removing rows/columns with NaN/inf values.
    Drop columns that are not relevant for ML.
    """
    if df.empty:
        return df

    print(f"\nCleaning {modality} dataframe...")
    print(f"  Original shape: {df.shape}")

    df_clean = df.copy()

    # Replace inf with NaN
    df_clean.replace([float('inf'), float('-inf')], pd.NA, inplace=True)

    # Drop non-feature columns first
    cols_to_drop = ["has_ppg", "has_ecg", "ppg_error", "has_bioimp",
                    "segment_id", "modality", "file_id"]
    df_clean = df_clean.drop(columns=cols_to_drop, errors='ignore')

    # Drop columns with any NaN values (problematic features)
    df_clean = old_remove_missing_data(
        df_clean, drop_rows=False, drop_cols=True)

    # Drop rows with NaN in remaining columns
    df_clean.dropna(axis=0, how="any", inplace=True)

    print(f"  Cleaned shape: {df_clean.shape}")

    if df_clean.isna().any().any():
        print(f"  Warning: {modality} still contains NaN values")
    if df_clean.isin([float('inf'), float('-inf')]).any().any():
        print(f"  Warning: {modality} still contains inf values")

    return df_clean


def load_modality_dataframes(datasetConfig: DatasetConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and clean dataframes for each modality."""
    output_dir = datasetConfig.get_dataset_machine_learning_path()

    # substitute final folder name with "ml" to match the output folder for machine learning
    output_dir = os.path.join(os.path.dirname(
        output_dir), "features_statistics_and_correlation")

    # Load dataframes
    df_ecg_path = os.path.join(output_dir, "features_ecg_segments_cleaned.csv")
    df_ppg_path = os.path.join(output_dir, "features_ppg_segments_cleaned.csv")
    df_bioimp_path = os.path.join(
        output_dir, "features_bioimp_segments_cleaned.csv")

    print("Loading modality dataframes...")
    print("Reading ECG features from:", df_ecg_path)
    df_ecg = pd.read_csv(df_ecg_path) if os.path.exists(
        df_ecg_path) else pd.DataFrame()
    print("Reading PPG features from:", df_ppg_path)
    df_ppg = pd.read_csv(df_ppg_path) if os.path.exists(
        df_ppg_path) else pd.DataFrame()
    print("Reading Bioimpedance features from:", df_bioimp_path)
    df_bioimp = pd.read_csv(df_bioimp_path) if os.path.exists(
        df_bioimp_path) else pd.DataFrame()

    # Clean dataframes
    # df_ecg = clean_by_removing_nan(df_ecg, "ECG")
    # df_ppg = clean_by_removing_nan(df_ppg, "PPG")
    # df_bioimp = clean_by_removing_nan(df_bioimp, "Bioimpedance")

    return df_ecg, df_ppg, df_bioimp


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get feature columns (excluding metadata)."""
    return [col for col in df.columns if col not in SOME_METADATA_COLUMNS]


def repeat_features(df: pd.DataFrame, n_times: int) -> pd.DataFrame:
    """Repeat rows n_times by cycling through them."""
    if df.empty or n_times <= 0:
        return df.iloc[0:0].copy()

    feature_cols = get_feature_columns(df)
    n_rows = len(df)

    if n_rows == 0:
        return pd.DataFrame()

    # Repeat rows cyclically
    repeated_indices = [i % n_rows for i in range(n_times)]
    return df.iloc[repeated_indices][feature_cols].reset_index(drop=True)


def concatenate_modality_features(
    df_ecg: pd.DataFrame,
    df_ppg: pd.DataFrame,
    df_bioimp: pd.DataFrame
) -> pd.DataFrame:
    """
    Concatenate features from all modalities using the specified strategy.
    """
    if df_ecg.empty and df_ppg.empty and df_bioimp.empty:
        raise ValueError("All modality dataframes are empty")

    print("\nConcatenating features from all modalities...")

    # Group by metadata
    all_groups = []

    # Collect all unique group keys
    group_keys = set()
    for df in [df_ecg, df_ppg, df_bioimp]:
        if not df.empty:
            group_keys.update(df[SOME_METADATA_COLUMNS].drop_duplicates(
            ).itertuples(index=False, name=None))

    print(f"Found {len(group_keys)} unique groups")

    num_skipped_groups = 0
    concatenated_rows = []

    for group_key in sorted(group_keys):
        participant_id, session_id, datetime, glc = group_key

        # Filter dataframes for this group
        mask_ecg = (df_ecg['participant_id'] == participant_id) & \
                   (df_ecg['session_id'] == session_id) & \
                   (df_ecg['datetime'] ==
                    datetime) if not df_ecg.empty else pd.Series([False])
        mask_ppg = (df_ppg['participant_id'] == participant_id) & \
                   (df_ppg['session_id'] == session_id) & \
                   (df_ppg['datetime'] ==
                    datetime) if not df_ppg.empty else pd.Series([False])
        mask_bioimp = (df_bioimp['participant_id'] == participant_id) & \
                      (df_bioimp['session_id'] == session_id) & \
                      (df_bioimp['datetime'] == datetime) if not df_bioimp.empty else pd.Series(
                          [False])

        group_ecg = df_ecg[mask_ecg] if not df_ecg.empty and mask_ecg.any(
        ) else pd.DataFrame()
        group_ppg = df_ppg[mask_ppg] if not df_ppg.empty and mask_ppg.any(
        ) else pd.DataFrame()
        group_bioimp = df_bioimp[mask_bioimp] if not df_bioimp.empty and mask_bioimp.any(
        ) else pd.DataFrame()

        # Count features per modality
        n_ecg = len(group_ecg)
        n_ppg = len(group_ppg)
        n_bioimp = len(group_bioimp)

        # Check for missing modalities
        modalities_present = sum([n_ecg > 0, n_ppg > 0, n_bioimp > 0])
        if modalities_present < 3:
            missing = []
            if n_ecg == 0:
                missing.append("ECG")
            if n_ppg == 0:
                missing.append("PPG")
            if n_bioimp == 0:
                missing.append("Bioimpedance")
            print(
                f"Skipping group (pid={participant_id}, sid={session_id}, dt={datetime}): missing {', '.join(missing)}")
            num_skipped_groups += 1
            continue

        # Determine Nmax
        nmax = max(n_ecg, n_ppg, n_bioimp)

        # Repeat features to match Nmax
        ecg_repeated = repeat_features(group_ecg, nmax)
        ppg_repeated = repeat_features(group_ppg, nmax)
        bioimp_repeated = repeat_features(group_bioimp, nmax)

        # Create concatenated rows
        for i in range(nmax):
            row = {
                "participant_id": participant_id,
                "session_id": session_id,
                "datetime": datetime,
                "GLC": glc
            }

            # Add features from each modality
            if not ecg_repeated.empty:
                row.update(ecg_repeated.iloc[i].to_dict())
            if not ppg_repeated.empty:
                row.update(ppg_repeated.iloc[i].to_dict())
            if not bioimp_repeated.empty:
                row.update(bioimp_repeated.iloc[i].to_dict())

            concatenated_rows.append(row)

    print(f"Total groups skipped: {num_skipped_groups}")
    print(f"Total concatenated rows created: {len(concatenated_rows)}")

    return pd.DataFrame(concatenated_rows)


def save_concatenated_features(
    df_concatenated: pd.DataFrame,
    datasetConfig: DatasetConfig
) -> Tuple[str, str]:
    """
    Save concatenated features to two CSV files:
    1. Full dataframe with all metadata and features
    2. ML-ready dataframe with X and y (where y=GLC)
    """
    output_dir = datasetConfig.get_dataset_machine_learning_path()

    # Save full concatenated features
    output_file_full = os.path.join(
        output_dir, OUTPUT_MULTIMODAL_FEATURES_FILE)
    df_concatenated.to_csv(output_file_full, index=False)
    print(f"\nFull concatenated features saved to: {output_file_full}")

    # Create ML-ready dataframe (X and y)
    feature_columns = [
        col for col in df_concatenated.columns
        if col not in SOME_METADATA_COLUMNS and col not in ML_EXCLUDED_COLUMNS
    ]
    df_ml = df_concatenated[feature_columns + ["GLC"]].copy()
    # df_ml.rename(columns={"GLC": "y"}, inplace=True)
    # df_ml.insert(0, "y", df_ml.pop("y"))  # Move y to first column

    # Save ML-ready features
    output_file_ml = os.path.join(
        output_dir, OUTPUT_MULTIMODAL_WITHOUT_METADATA_FILE)
    df_ml.to_csv(output_file_ml, index=False)
    print(f"ML-ready features saved to: {output_file_ml}")

    print(f"\nFinal shape - Full: {df_concatenated.shape}")
    print(f"Final shape - ML: {df_ml.shape}")

    return output_file_full, output_file_ml


if __name__ == "__main__":
    # Create a DatasetConfig instance
    dataset_config_file = "multimodal_dataset_folders.json"
    datasetConfig = DatasetConfig(dataset_config_file)

    # Load and clean modality dataframes
    df_ecg, df_ppg, df_bioimp = load_modality_dataframes(datasetConfig)

    # Concatenate features
    df_concatenated = concatenate_modality_features(df_ecg, df_ppg, df_bioimp)

    # Save concatenated features
    save_concatenated_features(df_concatenated, datasetConfig)

    print("\nFeature concatenation complete!")
