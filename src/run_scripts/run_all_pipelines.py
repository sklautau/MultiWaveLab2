'''
Script to run all pipelines in sequence, given a JSON configuration file.

Treat scripts as "executables" and call them with subprocess.
'''

import argparse
import argparse
import json
import subprocess
import sys
import pandas as pd
from datasets_util.naming_conventions import DatasetConfig

PYTHON_EXECUTABLE = "python"


class Tee:
    '''
    Code to be able to print to console and log file at the same time.
    '''

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            # s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def remove_missing_data(
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


def run_script(script_path: str, *args: str):
    # replace \ by / in script_path for Windows compatibility
    script_path = script_path.replace("\\", "/")

    command = [PYTHON_EXECUTABLE, script_path] + list(args)
    print(f"#### Running command: {' '.join(command)}")

    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as proc:

        for line in proc.stdout:
            print(line, end="")      # Goes through your Tee

        proc.wait()

        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, command)


def signal_processing(json_file: str):
    datasetConfig = DatasetConfig(json_file)
    modalities = datasetConfig.modalities
    # 2) calculate new signals
    if "ecg" in modalities:
        run_script(r'.\signal_processing\ecg.py', json_file)
    if "ppg" in modalities:
        run_script(r'.\signal_processing\ppg.py', json_file)
    if "bioimp" in modalities:
        run_script(r'.\signal_processing\bioimpedance.py', json_file)


def feature_extraction(json_file: str):
    # 3) Feature extraction for machine learning

    run_script(r'.\segments\create_segment_files.py', json_file)

    # the code will calculate all features per modality
    run_script(
        r'.\features\extract_features_for_all_modalities.py', json_file)


def fusion_and_split(json_file: str):
    # fuse the features, even in case they have a different
    # number of vectors, aiming at creating a large number of rows in
    # the output features dataframe
    run_script(
        r'.\features\fusion_of_multimodality_features.py', json_file)

    # use a manual train/test split based on pre-defined file lists
    run_script(r'.\machinelearning\train_test_splits.py', json_file)


def feature_selection(json_file: str):
    # 4) Feature selection for machine learning
    run_script(r'.\features\feature_selection.py', json_file)

    # Legacy code for LOSO
    # run_script(r'.\machinelearning\feature_selection_for_regression.py', '0', '20')
    # Legacy code for clustering and visualization of feature selection results
    # run_script(r'.\machinelearning\dimensionality_reduction.py')


def machine_learning_regression(json_file: str):
    # 5) Evaluate using nested and grouped cross-validation, with hyperparameter optimization
    run_script(r'.\machinelearning\regression_evaluation.py', json_file)


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

    json_file = args.json_file
    datasetConfig = DatasetConfig(json_file)
    log_file_name = datasetConfig.get_output_log_file_name()
    log = open(log_file_name, "w")
    sys.stdout = Tee(sys.stdout, log)
    print("Will log in file", log_file_name)

    signal_processing(json_file)
    feature_extraction(json_file)
    fusion_and_split(json_file)
    feature_selection(json_file)
    machine_learning_regression(json_file)
