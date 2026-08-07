'''
Manually create a pandas dataframe with the
following column name:
- participant_id
in order to split the data into disjoint sets:
 training, validation and test sets.
'''

import argparse
import re
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from pathlib import Path

from datasets_util.naming_conventions import LOGIDENTIFIER, DatasetConfig

# DATASET_CONFIG_FILE = "multimodal_dataset_folders.json"
# DATASET_CONFIG_FILE = "ieb1_multimodal_dataset_folders.json"
# prefix for the output csv file names
# prefix for the output txt file names
# OUTPUT_PREAMBLE_TRAIN_TEST_SPLITS_FILE_NAMES = "split_ieb1"
# MANUAL_PREAMBLE_TRAIN_TEST_SPLITS_FILE_NAMES = "SOME"

LOW_GLC_THRESHOLD = 101.42
HIGH_GLC_THRESHOLD = 157.88

# INPUT_FEATURES_FILE = r"..\output_ieb1\ml\ufsc_dissertation_ppg_features.csv"
# INPUT_FEATURES_FILE = r"..\output_ieb1\ml\ufsc_dissertation_ppg_features_aggregated.csv"
# INPUT_FEATURES_FILE = r"..\output_ieb1\features_statistics_and_correlation\dataRecord_spectrogram_v5_averaged_cleaned.csv"
# INPUT_FEATURES_FILE = r"..\tcc_guilherme\files\dataRecord_spectrogram_v5_averaged.csv"
# INPUT_FEATURES_FILE = r"..\output_ieb1\features_statistics_and_correlation\dataRecord_spectrogram_v5_cleaned.csv"
# INPUT_FEATURES_FILE = r"..\output_ieb3\features_statistics_and_correlation\multimodal_features_with_metadata_cleaned.csv"
# INPUT_FEATURES_FILE = r"C:\git_sofis\tcc_guilherme\files\dataRecord_spectrogram_v5.csv"
# INPUT_FEATURES_FILE = r"C:\git_sofis\output_luis_ieb1\all_ppg_features.csv"
# INPUT_FEATURES_FILE = r"..\output_ieb1\ml\features_ppg_segments.csv"
# INPUT_FEATURES_FILE = r"..\output_ieb1\ml\multimodal_features_with_metadata.csv"
# INPUT_FEATURES_FILE = r"..\output_ieb3\ml\multimodal_features_with_metadata.csv"
# INPUT_FEATURES_FILE = r"..\output_ieb3\ml\features_ppg_segments.csv"
# INPUT_FEATURES_FILE = r"..\output_ieb3\ml\features_ecg_segments.csv"
# INPUT_FEATURES_FILE = r"..\output_ieb3\ml\features_bioimp_segments.csv"


def create_balanced_set(features_set: pd.DataFrame) -> pd.DataFrame:
    """
    # make sure that all participant IDs in test_set contribute
    # with the same number of rows by finding the minimum number
    # of rows per participant in the test set and excluding the
    # lowest-quality rows from participants with more rows than the minimum

     Keeps, for each participant, the rows with the highest quality_indicator values instead of arbitrarily taking the first rows. It sorts by participant_id and quality_indicator descending, then keeps the top rows up to the minimum per-participant count. If quality_indicator is missing, it now raises a clear error.
    """
    num_rows_test = len(features_set)
    if "quality_indicator" not in features_set.columns:
        raise ValueError(
            "features_set must contain a 'quality_indicator' column to balance by quality"
        )

    min_rows_per_participant = features_set.groupby(
        "participant_id").size().min()
    print(LOGIDENTIFIER +
          f" # Minimum number of rows per participant in this set: {min_rows_per_participant}")
    print(" # Keeping the rows with the highest quality_indicator for each participant.")

    features_set = (
        features_set.sort_values(
            by=["participant_id", "quality_indicator"],
            ascending=[True, False],
            kind="mergesort",
        )
        .groupby("participant_id", sort=False)
        .head(min_rows_per_participant)
        .reset_index(drop=True)
    )
    print(LOGIDENTIFIER +
          f" # Set reduced from {num_rows_test} rows to {len(features_set)} rows after removing excess rows.")
    return features_set


def find_missing_participant_ids(
    df: pd.DataFrame,
    participant_col: str = "participant_id",
    start: int = 0,
) -> list[str]:
    """
    Finds missing participant IDs.

    Supports IDs such as:
        ieb_03, ieb_04, ...
        id01, id05, ...
        1, 2, 4, 5, ...

    Parameters
    ----------
    start : int
        First ID to consider when searching for missing values.

    Returns
    -------
    list[str]
        Missing participant IDs.
    """

    ids = df[participant_col].dropna().astype(str).tolist()

    if not ids:
        return []

    # ------------------------------------------------------------------
    # Case 1: IDs are purely numeric
    # ------------------------------------------------------------------
    if all(re.fullmatch(r"\d+", pid) for pid in ids):

        width = max(len(pid) for pid in ids)
        existing = {int(pid) for pid in ids}

        return [
            f"{i:0{width}d}" if width > 1 else str(i)
            for i in range(start, max(existing) + 1)
            if i not in existing
        ]

    # ------------------------------------------------------------------
    # Case 2: IDs have a common prefix followed by digits
    # ------------------------------------------------------------------
    pattern = re.compile(r"^(.*?)(\d+)$")

    parsed = []

    for pid in ids:
        m = pattern.match(pid)
        if m is None:
            raise ValueError(
                f"Participant ID '{pid}' does not end with a numeric suffix."
            )

        prefix, number = m.groups()
        parsed.append((prefix, int(number), len(number)))

    prefixes = {p for p, _, _ in parsed}
    if len(prefixes) != 1:
        raise ValueError(f"Multiple prefixes found: {prefixes}")

    prefix = prefixes.pop()

    widths = {w for _, _, w in parsed}
    if len(widths) != 1:
        raise ValueError(f"Numeric suffixes have different widths: {widths}")

    width = widths.pop()

    existing = {n for _, n, _ in parsed}

    return [
        f"{prefix}{i:0{width}d}"
        for i in range(start, max(existing) + 1)
        if i not in existing
    ]


def convert_participant_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts participant IDs to a consistent format.
    It converts from #001 to hu_01, #002 to hu_02, etc.
    Returns
    -------
    pd.DataFrame
        Dataframe with converted participant IDs.
    """

    if "participant_id" not in df.columns:
        raise ValueError(
            f"Input features must contain a 'participant_id' column."
        )

    def convert_id(pid: str) -> str:
        if pid.startswith("#"):
            # Convert from #001 to hu_01
            return f"hu_{int(pid[1:]):02d}"
        return pid

    df["participant_id"] = df["participant_id"].astype(str).apply(convert_id)

    return df


def __extract_participants_from_features_file(dataframe, list_of_participant_ids) -> pd.DataFrame:
    return dataframe[dataframe["participant_id"].isin(list_of_participant_ids)]


def features_files_from_defined_participants_lists(datasetConfig: DatasetConfig) -> None:
    features_folder = datasetConfig.get_dataset_features_path()
    features_file_prefix = datasetConfig.get_value("FEATURES_FILE_PREFIX")
    preamble_train_test_splits = datasetConfig.get_value(
        "TRAIN_TEST_SPLIT_PREFIX")

    # folder where the manual splits are defined as txt files: splits_folder
    splits_folder = os.path.join(datasetConfig.get_value(
        "SIMULATIONS_INPUT_PATH"), "train_test_splits")

    features_file = os.path.join(
        features_folder, datasetConfig.get_features_file_name())
    # check whether the features_file exists
    if not os.path.exists(features_file):
        raise FileNotFoundError(
            f"Features file '{features_file}' does not exist. Please run the feature extraction script first.")
    features_file = os.path.normpath(features_file)
    print("Using input file", features_file)
    df = pd.read_csv(features_file)
    print("Input dataframe shape:", df.shape)

    df = format_metadata(df)

    # print("columns in the dataframe:", df['participant_id'][0:10])
    df = convert_participant_ids(df)
    # print("columns in the dataframe:", df['participant_id'][0:10])
    # exit(-1)

    # folder where the manual splits are: splits_folder
    # read the participants_train.txt file, taking in account that the first row is the header
    training_set_participants = pd.read_csv(os.path.join(
        splits_folder, preamble_train_test_splits + "_participants_train.txt"), header=0)
    test_set_participants = pd.read_csv(os.path.join(
        splits_folder, preamble_train_test_splits + "_participants_test.txt"), header=0)
    # check the size of the validation file to verify if it is empty
    validation_file_path = os.path.join(
        splits_folder, preamble_train_test_splits + "_participants_validation.txt")
    # check if the validation file exists and is not empty
    is_there_validation = False
    if os.path.exists(validation_file_path) and os.path.getsize(validation_file_path) > 0:
        validation_set_participants = pd.read_csv(
            validation_file_path, header=0)
        is_there_validation = True

    training_set = __extract_participants_from_features_file(
        df, training_set_participants["participant_id"])
    test_set = __extract_participants_from_features_file(
        df, test_set_participants["participant_id"])

    if is_there_validation:
        validation_set = __extract_participants_from_features_file(
            df, validation_set_participants["participant_id"])

    # balance the test set, and if requested, also balance training and validation sets
    # by removing excess rows from participants with more than the minimum number of rows
    if datasetConfig.get_value("DROP_TEST_EXAMPLES_FOR_PARTICIPANT_BALANCE", False):
        print(LOGIDENTIFIER + "Balancing test set:")
        test_set = create_balanced_set(test_set)
    if datasetConfig.get_value("DROP_TRAINING_EXAMPLES_FOR_PARTICIPANT_BALANCE", False):
        print(LOGIDENTIFIER + "Balancing training set:")
        training_set = create_balanced_set(training_set)
        if is_there_validation:
            print("Balancing validation set:")
            validation_set = create_balanced_set(validation_set)

    # save the sets to csv files:
    train_csv = datasetConfig.get_splitted_features_file_name("train")
    # features_folder, features_file_prefix + "_" + preamble_train_test_splits + "_train.csv")
    training_set.to_csv(train_csv, index=False)
    test_csv = datasetConfig.get_splitted_features_file_name("test")
    # features_folder, features_file_prefix + "_" + preamble_train_test_splits + "_test.csv")
    test_set.to_csv(test_csv, index=False)
    if is_there_validation:
        validation_csv = datasetConfig.get_splitted_features_file_name(
            "validation")
        # features_folder, features_file_prefix + "_" + preamble_train_test_splits + "_validation.csv")
        validation_set.to_csv(validation_csv, index=False)

    # convert all / and \ to the OS-specific separator in the output folder path
    print(
        f"Wrote training set to {os.path.normpath(train_csv)}")
    print(LOGIDENTIFIER + "Output training dataframe shape:", training_set.shape)
    print(
        f"Wrote test set to {os.path.normpath(test_csv)}")
    print(LOGIDENTIFIER + "Output test dataframe shape:", test_set.shape)
    if is_there_validation:
        print(
            f"Wrote validation set to {os.path.normpath(validation_csv)}")
        print(LOGIDENTIFIER + "Output validation dataframe shape:",
              validation_set.shape)


def fixed_split_for_dataset_ieb3(datasetConfig: DatasetConfig) -> None:
    print("Creating splits of participants into training, validation, and test sets. These files just list the participant IDs, not the actual data.")
    training_set = pd.DataFrame(
        [
            {"participant_id": "ieb_01"},
            {"participant_id": "ieb_02"},
            {"participant_id": "ieb_03"},
            {"participant_id": "ieb_04"},
            {"participant_id": "ieb_06"},
            {"participant_id": "ieb_08"},
        ]
    )
    test_set = pd.DataFrame(
        [
            {"participant_id": "ieb_05"},
            {"participant_id": "ieb_07"},
        ]
    )
    validation_set = pd.DataFrame(
        [
            {"participant_id": "ieb_05"},
            {"participant_id": "ieb_07"},
        ]
    )

    # Save the sets to csv files:
    output_folder = datasetConfig.get_dataset_machine_learning_path()
    # convert all / and \ to the OS-specific separator in the output folder path
    output_folder = os.path.normpath(output_folder)

    output_train_file = os.path.join(
        output_folder, OUTPUT_PREAMBLE_TRAIN_TEST_SPLITS_FILE_NAMES + "_participants_train.txt")
    training_set.to_csv(output_train_file, index=False)

    output_test_file = os.path.join(
        output_folder, OUTPUT_PREAMBLE_TRAIN_TEST_SPLITS_FILE_NAMES + "_participants_test.txt")
    test_set.to_csv(output_test_file, index=False)

    output_validation_file = os.path.join(
        output_folder, OUTPUT_PREAMBLE_TRAIN_TEST_SPLITS_FILE_NAMES + "_participants_validation.txt")
    validation_set.to_csv(output_validation_file, index=False)

    print(f"Wrote list of participants in training set to {output_train_file}")
    print(f"Wrote list of participants in test set to {output_test_file}")
    print(
        f"Wrote list of participants in validation set to {output_validation_file}")


def create_random_splits_and_write_participants_list(
    output_folder: str,
    df: pd.DataFrame,
    train_pct: float = 0.7,
    val_pct: float = 0.15,
    test_pct: float = 0.15,
    random_state: int = 42,
) -> None:
    """
    Splits participants into train/validation/test sets based on percentages.

    Args:
        datasetConfig: config object with output path
        df: dataframe containing a 'participant_id' column
        train_pct, val_pct, test_pct: must sum to 1.0
        random_state: for reproducibility
    """

    assert abs(train_pct + val_pct + test_pct -
               1.0) < 1e-6, "Percentages must sum to 1.0"

    # Get unique participants
    participants = df["participant_id"].dropna().unique()
    participants = pd.DataFrame({"participant_id": participants})

    # First split: train vs temp (val+test)
    train_df, temp_df = train_test_split(
        participants,
        test_size=(1 - train_pct),
        random_state=random_state,
        shuffle=True,
    )

    # Second split: validation vs test
    val_relative = val_pct / (val_pct + test_pct)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_relative),
        random_state=random_state,
        shuffle=True,
    )

    # Save
    # output_folder = datasetConfig.get_dataset_machine_learning_path()

    train_file = os.path.join(
        output_folder, OUTPUT_PREAMBLE_TRAIN_TEST_SPLITS_FILE_NAMES + "_participants_train.txt")
    val_file = os.path.join(
        output_folder, OUTPUT_PREAMBLE_TRAIN_TEST_SPLITS_FILE_NAMES + "_participants_validation.txt")
    test_file = os.path.join(
        output_folder, OUTPUT_PREAMBLE_TRAIN_TEST_SPLITS_FILE_NAMES + "_participants_test.txt")

    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)
    test_df.to_csv(test_file, index=False)

    print(f"Train: {len(train_df)} participants → {train_file}")
    print(f"Validation: {len(val_df)} participants → {val_file}")
    print(f"Test: {len(test_df)} participants → {test_file}")


def create_stratified_glc_splits_and_write_participants_list(
    output_folder: str,
    df: pd.DataFrame,
    train_pct: float = 0.7,
    val_pct: float = 0.15,
    test_pct: float = 0.15,
    random_state: int = 42,
    low_threshold: float = 70.0,
    high_threshold: float = 180.0,
    participant_col: str = "participant_id",
    glc_col: str = "GLC",
    aggregation: str = "median",
) -> None:
    """
    Splits participants into train/validation/test sets disjoint by participant_id,
    stratified by glucose class derived from GLC.

    Glucose classes:
        low    : GLC < low_threshold
        normal : low_threshold <= GLC <= high_threshold
        high   : GLC > high_threshold

    Since each participant may have many GLC samples, one participant-level GLC
    value is first computed using aggregation, e.g., median or mean.

    Parameters
    ----------
    output_folder : str
        Folder where participant lists will be written.
    df : pd.DataFrame
        Dataframe containing participant_id and GLC columns.
    train_pct, val_pct, test_pct : float
        Split percentages. Must sum to 1.
    random_state : int
        Random seed for reproducibility.
    low_threshold : float
        Threshold below which glucose is classified as low.
    high_threshold : float
        Threshold above which glucose is classified as high.
    participant_col : str
        Name of the participant ID column.
    glc_col : str
        Name of the glucose column.
    aggregation : str
        Participant-level GLC aggregation: "median" or "mean".
    """

    assert abs(train_pct + val_pct + test_pct - 1.0) < 1e-6, (
        "Percentages must sum to 1.0"
    )

    required_cols = {participant_col, glc_col}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    work_df = df[[participant_col, glc_col]].dropna().copy()

    if aggregation == "median":
        participant_glc = (
            work_df.groupby(participant_col, as_index=False)[glc_col].median()
        )
    elif aggregation == "mean":
        participant_glc = (
            work_df.groupby(participant_col, as_index=False)[glc_col].mean()
        )
    else:
        raise ValueError("aggregation must be 'median' or 'mean'")

    participant_glc["glc_class"] = np.select(
        [
            participant_glc[glc_col] < low_threshold,
            participant_glc[glc_col] > high_threshold,
        ],
        ["low", "high"],
        default="normal",
    )

    def _build_stratify_labels_or_none(
        labels: pd.Series,
        test_size: float,
        split_name: str,
    ) -> pd.Series | None:
        class_counts = labels.value_counts()
        n_classes = len(class_counts)
        n_samples = len(labels)

        # Stratified split requires at least 2 samples in every class.
        if n_classes == 0 or class_counts.min() < 2:
            print(
                f"Warning: '{split_name}' split is not stratified because at least one class has fewer than 2 participants."
            )
            return None

        n_test = int(np.ceil(test_size * n_samples))
        n_train = n_samples - n_test

        # Stratified split also needs enough samples in each side to host all classes.
        if n_train < n_classes or n_test < n_classes:
            print(
                f"Warning: '{split_name}' split is not stratified because split sizes are too small for {n_classes} classes (train={n_train}, test={n_test})."
            )
            return None

        return labels

    first_test_size = 1.0 - train_pct
    first_stratify = _build_stratify_labels_or_none(
        participant_glc["glc_class"],
        test_size=first_test_size,
        split_name="train-vs-temp",
    )

    # First split: train vs temp
    train_df, temp_df = train_test_split(
        participant_glc,
        test_size=first_test_size,
        random_state=random_state,
        shuffle=True,
        stratify=first_stratify,
    )

    # Second split: validation vs test
    val_relative = val_pct / (val_pct + test_pct)
    second_test_size = 1.0 - val_relative
    second_stratify = _build_stratify_labels_or_none(
        temp_df["glc_class"],
        test_size=second_test_size,
        split_name="validation-vs-test",
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=second_test_size,
        random_state=random_state,
        shuffle=True,
        stratify=second_stratify,
    )

    os.makedirs(output_folder, exist_ok=True)

    train_file = os.path.join(
        output_folder, OUTPUT_PREAMBLE_TRAIN_TEST_SPLITS_FILE_NAMES + "_participants_train.txt"
    )
    val_file = os.path.join(
        output_folder, OUTPUT_PREAMBLE_TRAIN_TEST_SPLITS_FILE_NAMES +
        "_participants_validation.txt"
    )
    test_file = os.path.join(
        output_folder, OUTPUT_PREAMBLE_TRAIN_TEST_SPLITS_FILE_NAMES + "_participants_test.txt"
    )

    # before saving, sort the dataframes by participant_id for consistency
    train_df = train_df.sort_values(by=participant_col)
    val_df = val_df.sort_values(by=participant_col)
    test_df = test_df.sort_values(by=participant_col)

    train_df[[participant_col]].to_csv(train_file, index=False)
    val_df[[participant_col]].to_csv(val_file, index=False)
    test_df[[participant_col]].to_csv(test_file, index=False)

    # create a combined dataframe with all participants and their assigned split
    combined_df = pd.concat([
        train_df.assign(split="train"),
        val_df.assign(split="validation"),
        test_df.assign(split="test"),
    ])
    missing_ids = find_missing_participant_ids(
        combined_df,
        participant_col=participant_col,
        start=0,
    )
    print("\nMissing participant IDs (if any):", missing_ids, "\n")

    # check whether the sets are disjoint and count the number of unique values:
    train_participants = set(train_df[participant_col])
    val_participants = set(val_df[participant_col])
    test_participants = set(test_df[participant_col])
    # check disjointness
    assert train_participants.isdisjoint(
        val_participants), "Train and validation sets are not disjoint"
    assert train_participants.isdisjoint(
        test_participants), "Train and test sets are not disjoint"
    assert val_participants.isdisjoint(
        test_participants), "Validation and test sets are not disjoint"
    # print the number of participants in each set and the file they were written to
    print(f"Train: {len(train_df)} participants → {train_file}")
    print(f"Validation: {len(val_df)} participants → {val_file}")
    print(f"Test: {len(test_df)} participants → {test_file}")
    # print total number of unique participants
    total_unique_participants = len(train_participants.union(
        val_participants).union(test_participants))
    print(f"\nTotal unique participants: {total_unique_participants}")

    print("\nClass distribution:")
    print("Train:")
    print(train_df["glc_class"].value_counts(normalize=True))
    print("Validation:")
    print(val_df["glc_class"].value_counts(normalize=True))
    print("Test:")
    print(test_df["glc_class"].value_counts(normalize=True))


def run_fixed_split_for_dataset_ieb3(datasetConfig):
    # datasetConfig = DatasetConfig(DATASET_CONFIG_FILE)
    fixed_split_for_dataset_ieb3(datasetConfig)
    output_folder = datasetConfig.get_dataset_machine_learning_path()
    print("\nNow, based on the list of participants, create the train/validation/test splits from the features file.")
    raise NotImplementedError(
        "Please implement the function to create splits from the features file based on the fixed participant lists.")
    # features_files_from_defined_participants_lists(output_folder)


def format_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures that the dataframe has a column named 'participant_id'.
    If it has a column named 'participant', it renames it to 'participant_id'.
    Raises an error if neither column is present.
    """
    if "session_id" not in df.columns:
        # add a column named "session_id" with all values set to 1
        df["session_id"] = 1

    if "datetime" not in df.columns:
        # add a column named "datetime" with all values set to "2023-01-01 00:00:00"
        df["datetime"] = "0000-01-01"

    # Check for columns that are not needed for machine learning and remove them
    # check whether GENDER is a column
    if "GENDER" in df.columns:
        columns_to_be_removed = ["AGE", "GENDER",
                                 "HEIGHT(cm)", "WEIGHT(kg)", "IP", "EP"]
        df = df.drop(columns=columns_to_be_removed)
    if "AGE" in df.columns:
        columns_to_be_removed = ["AGE"]
        df = df.drop(columns=columns_to_be_removed)

    if "participant_id" in df.columns:
        return df
    elif "SUBJECT_ID" in df.columns:
        df = df.rename(columns={"SUBJECT_ID": "participant_id"})
        return df
    elif "ID" in df.columns:
        df = df.rename(columns={"ID": "participant_id"})
        return df
    else:
        raise ValueError(
            "Dataframe must contain either 'participant_id' or 'participant' column.")


def run_split_with_stratified_glucose(datasetConfig):
    # datasetConfig = DatasetConfig(DATASET_CONFIG_FILE)

    output_folder = datasetConfig.get_dataset_machine_learning_path()

    INPUT_FEATURES_FILE = datasetConfig.get_features_file_name()

    print("Input file: ", INPUT_FEATURES_FILE)

    train_pct = 0.7
    val_pct = 0.15
    test_pct = 0.15
    random_state = 42
    low_threshold = LOW_GLC_THRESHOLD
    high_threshold = HIGH_GLC_THRESHOLD
    participant_col = "participant_id"
    glc_col = "GLC"
    aggregation = "median"

    # read pd.Dataframe from CSV file
    df = pd.read_csv(INPUT_FEATURES_FILE)
    df = format_metadata(df)

    print(
        f"Using low_threshold = {low_threshold}, high_threshold = {high_threshold} for stratification.")
    # find the 33% and 66% percentiles of the GLC column
    ideal_low_threshold = np.percentile(df[glc_col].dropna(), 33)
    ideal_high_threshold = np.percentile(df[glc_col].dropna(), 66)
    print(
        f"Suggested (ideal) thresholds: low_threshold = {ideal_low_threshold}, high_threshold = {ideal_high_threshold} for stratification.")

    create_stratified_glc_splits_and_write_participants_list(
        output_folder,
        df,
        train_pct,
        val_pct,
        test_pct,
        random_state,
        low_threshold,
        high_threshold,
        participant_col=participant_col,
        glc_col=glc_col,
        aggregation=aggregation,
    )

    print("\nNow, based on the list of participants, create the train/validation/test splits from the features file.")
    # features_files_from_defined_participants_lists(output_folder)
    raise NotImplementedError(
        "Please implement the function to create splits from the features file based on the stratified participant lists.")


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

    features_files_from_defined_participants_lists(datasetConfig)
