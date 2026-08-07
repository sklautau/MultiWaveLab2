'''
This is valid only for IEB1 dataset, which has a specific folder structure and naming convention.

Get files from:
..\\ufsc-bgl-based\\Dataset_IEB\\sqi_signals
named
#001_sqi_signal.csv
#002_sqi_signal.csv
...
and combine them into a single dataframe with all columns

and also get single file
..\\ufsc-bgl-based\\Dataset_IEB\\window_debug_report_with_sqi_signal.csv

'''
from signal_processing.ppg import extract_ppg_features, ppg_bandpass_waveform_processing
from features.estimate_sqi_using_external_libraries import run_all_sqi_estimators
from datasets_util.waveform_files import read_sigmf_file
from datasets_util.util_visualize_plots import plot_signal_with_sqi
from datasets_util.util_visualize_plots import plot_rmse_matrix
from datasets_util.util_various_methods import pairwise_mse
from datasets_util.util_various_methods import accumulate_mse_dicts
from datasets_util.naming_conventions import DatasetConfig
import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

INPUT_FOLDER = "..\\ufsc-bgl-based\\Dataset_IEB\\sqi_signals"
FILE_NAME_SUFFIX = "_sqi_signal.csv"

OTHER_FILE_NAME = "..\\ufsc-bgl-based\\Dataset_IEB\\window_debug_report_with_sqi_signal.csv"

# Ensure project root is available when script is run as a standalone file.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_binary_vectors(csv_file):
    """
    Load a CSV containing columns of binary vectors with different lengths.

    Returns
    -------
    dict
        Dictionary where the keys are the column names and the values
        are NumPy arrays containing only the original vector elements.
    """
    df = pd.read_csv(csv_file)

    vectors = {}

    for col in df.columns:
        # Remove NaNs introduced by pandas
        vec = df[col].dropna().astype(np.uint8).to_numpy()
        vectors[col] = vec

    return vectors


def _plot_two_ppgs(ppg1: np.ndarray, ppg2: np.ndarray, fs: int, note: str = "") -> None:
    """Plot two PPG signals for visual comparison."""
    import matplotlib.pyplot as plt

    time_axis1 = np.arange(len(ppg1)) / fs
    time_axis2 = np.arange(len(ppg2)) / fs
    plt.figure(figsize=(12, 6))
    plt.plot(time_axis1, ppg1, label="MultiWavLab", alpha=0.7)
    plt.plot(time_axis2, ppg2, label="UFSC/Luis", alpha=0.7)
    plt.title(f"Comparison of Two PPG Signals {note}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()
    plt.show()


def _results_to_dataframe(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert SQI estimator outputs to a sample-wise DataFrame."""
    sqi_columns: dict[str, np.ndarray] = {}

    for idx, result in enumerate(results, start=1):
        name = str(result.get("name", f"estimator_{idx}"))
        column_name = f"sqi_{name.lower().replace(' ', '_').replace('-', '_')}"
        sqi_columns[column_name] = np.asarray(
            result.get("sqi_for_each_signal_sample", []), dtype=float
        )

    return pd.DataFrame(sqi_columns)


def _read_ppg_for_comparison(
    dataset_config: DatasetConfig,
    file_id: str,
    required_ppg_fs: int,
    pipeline: str = "filtered"
) -> tuple[np.ndarray, str]:
    """Load preferred PPG input for SQI comparison.

    Prefer already-generated filtered waveform. If it does not exist, read raw
    SigMF and run the same filtering used in the current PPG pipeline.
    """
    if pipeline == "filtered":
        filtered_path = dataset_config.get_gen_complete_path(file_id, pipeline)
        if os.path.exists(filtered_path):
            signal, metadata = read_sigmf_file(filtered_path)
            fs = metadata["global"]["core:sample_rate"]
            if fs != required_ppg_fs:
                raise ValueError(
                    f"Expected {required_ppg_fs} Hz but got {fs} Hz in {filtered_path}"
                )
            return np.asarray(signal, dtype=float), filtered_path

        raw_path = dataset_config.get_raw_complete_path(file_id)
        filtered_signal, _, _ = ppg_bandpass_waveform_processing(
            raw_path, required_ppg_fs)
        return np.asarray(filtered_signal, dtype=float), raw_path
    elif pipeline == "raw":
        raw_path = dataset_config.get_raw_complete_path(file_id)
        # read the signal from file
        raw_signal = read_sigmf_file(raw_path)[0]
        return np.asarray(raw_signal, dtype=float), raw_path
    else:
        raise ValueError(f"Unknown pipeline: {pipeline}")


def add_new_columns_with_sqis_sum(df: pd.DataFrame, output_filename: str) -> pd.DataFrame:
    """Add new columns to the DataFrame with the sum of SQI values for each row."""
    sqi_columns = [col for col in df.columns if col.startswith("sqi_")]
    df["sqi_sum"] = df[sqi_columns].sum(axis=1)

    # save output_filename
    df.to_csv(output_filename, index=False)
    print("Wrote SQI DataFrame with sum of SQIs to:", output_filename)
    return df


def calculate_and_save_sqis(dataset_config_file: str,
                            should_show: bool = False
                            ) -> pd.DataFrame:
    dataset_config = DatasetConfig(dataset_config_file)
    required_ppg_fs = dataset_config.get_ppg_fs()

    df = dataset_config.get_dataset_info_dataframe()
    df = df[df["modality"].str.contains("ppg", na=False)]

    print("Columns in the dataframe:", df.columns.tolist())

    print(f"Found {len(df)} PPG files to process (fs={required_ppg_fs} Hz)")

    # ADD the SQI from the new file
    dictionary_with_other_sqi = load_binary_vectors(OTHER_FILE_NAME)
    # check the number of rows in each column of the dictionary and print them
    for col in dictionary_with_other_sqi:
        print(f"Column {col} has {len(dictionary_with_other_sqi[col])} rows")

    # Initialize per-file summary rows with:
    # file_id, participant_id, and average(sum of all SQI waveforms per sample).
    total_sqi_per_subject = list()

    for idx, row in enumerate(df.itertuples(index=False), start=1):

        # if idx < 82:
        #    continue
        file_id = str(row.file_id)
        participant_id = str(getattr(row, "participant_id", "unknown"))
        print(f"\n[{idx}/{len(df)}] file_id={file_id} participant={participant_id}")

        glc = str(row.GLC)

        # convert file_id08 to file_id8, file_id09 to file_id9, etc.
        # if file_id.startswith("file_id0"):
        #    file_id = "file_id" + file_id[8:]
        #    print(f"  Converted file_id to {file_id}")

        try:
            ppg_signal, signal_path = _read_ppg_for_comparison(
                dataset_config, file_id, required_ppg_fs, pipeline="raw"
            )
            print(f"  Input signal: {signal_path}")
            print(f"  Samples: {len(ppg_signal)}, shape = {ppg_signal.shape}")
        except Exception as exc:
            print(
                f"  ERROR reading/processing signal for file_id={file_id}: {exc}")
            continue

        # extract features
        if False:
            features = extract_ppg_features(ppg_signal, required_ppg_fs)
            print(features)
            exit(1)

        results = run_all_sqi_estimators(ppg_signal, required_ppg_fs)
        if len(results) < 2:
            print(
                "  WARNING: fewer than 2 valid SQI estimators for this file, skipping MSE")
            continue

        # open Get files from:
        # ..\\ufsc-bgl-based\\Dataset_IEB\\sqi_signals
        # named
        #   #001_sqi_signal.csv
        #   #002_sqi_signal.csv
        #   ...
        # and combine them into a single dataframe with all columns
        # extract 3 algarism number from file_id, e.g. file_id01 -> 001, file_id10 -> 010, file_id100 -> 100
        print(f"  file_id={file_id}")
        number_id = file_id[7:]
        # convert to str with leading zeros to make it 3 digits
        number_id = str(int(number_id)).zfill(3)
        sqi_csv_path = os.path.join(
            INPUT_FOLDER, f"#{number_id}{FILE_NAME_SUFFIX}")
        if not os.path.exists(sqi_csv_path):
            raise ValueError(
                f"  WARNING: SQI CSV file not found: {sqi_csv_path}")

        other_ppg = np.asarray(pd.read_csv(sqi_csv_path)["ppg"], dtype=float)
        if len(other_ppg) != len(ppg_signal):
            print(ppg_signal[-10:])
            print(other_ppg[-10:])
            # raise Exception(
            print(
                f"  WARNING: length mismatch between PPG signal and CSV ppg column: "
                f"len(ppg_signal)={len(ppg_signal)} vs len(other_ppg)={len(other_ppg)}"
            )
            _plot_two_ppgs(ppg_signal, other_ppg, required_ppg_fs,
                           note=f"File ID: {file_id}")

        # concatenate estimator SQIs with CSV columns
        sqi_df = pd.read_csv(sqi_csv_path)
        results_df = _results_to_dataframe(results)

        # ADD the SQI from the new file
        # filter other_df to only include rows with the current file_id
        this_other_sqi = dictionary_with_other_sqi[file_id]

        if len(this_other_sqi) != len(other_ppg):
            raise Exception("file_id=", file_id, "with len=", len(
                this_other_sqi), "while it should be=", len(other_ppg))

        # add (concatenate) this_other_sqi into sqi_df
        sqi_df["sqi_ufsc_ieb1_validity"] = this_other_sqi

        if len(results_df) != len(sqi_df):
            print(
                f"  WARNING: length mismatch results={len(results_df)} vs csv={len(sqi_df)}; "
                "truncating both to the common minimum length"
            )
            common_len = min(len(results_df), len(sqi_df))
            results_df = results_df.iloc[:common_len].reset_index(drop=True)
            sqi_df = sqi_df.iloc[:common_len].reset_index(drop=True)

        combined_sqi_df = pd.concat(
            [results_df.reset_index(drop=True), sqi_df.reset_index(drop=True)],
            axis=1,
        )
        # print(f"  Combined SQI dataframe shape: {combined_sqi_df.shape}")
        # print("All columns:", combined_sqi_df.columns.tolist())

        # find the number of unique values in column sqi_neurokit2___entropy
        # unique_values = combined_sqi_df['sqi_neurokit2___entropy'].nunique()
        # print(
        #    f"  Number of unique values in sqi_neurokit2___entropy: {unique_values}")

        # rename columns: 'validity' to 'sqi_ieb1_validity' and 'total_sqi' to 'sqi_ieb1_total'
        combined_sqi_df = combined_sqi_df.rename(columns={
            'validity': 'IEB1_pulse_validity',
            'total_sqi': 'IEB1_score',
            'window_validity': 'IEB1_window_validity'
        })

        columns_to_drop = ['sample_idx', 'ppg', 'sqi_component',
                           'skewness_component', 'ipa_component',
                           'peak_pos_component', 'sqi_neurokit2___entropy',
                           # 'sqi_neurokit2___templatematch',
                           # 'sqi_neurokit2___ho2025', 'IEB1_score', 'IEB1_window_validity'
                           ]
        combined_sqi_df = combined_sqi_df.drop(columns=columns_to_drop)
        # print("New columns:", combined_sqi_df.columns.tolist())

        results_for_plot = [
            {
                "name": column,
                "sqi_per_pulse": np.array([]),
                "sqi_for_each_signal_sample": np.asarray(
                    combined_sqi_df[column], dtype=float
                ),
            }
            for column in combined_sqi_df.columns
        ]

        if len(results_for_plot) == 0:
            print(
                f"  WARNING: no SQI waveforms available for file_id={file_id}, skipping")
            continue

        # Sum all waveform values across all SQI series and normalize by
        # number of samples in the file to get the requested average.
        n_samples = len(results_for_plot[0]["sqi_for_each_signal_sample"])
        if n_samples == 0:
            print(
                f"  WARNING: empty SQI waveform for file_id={file_id}, skipping")
            continue

        waveform_sum = 0.0
        for waveform_dict in results_for_plot:
            waveform = np.asarray(
                waveform_dict.get("sqi_for_each_signal_sample", []), dtype=float
            )
            if len(waveform) != n_samples:
                print(
                    f"  WARNING: inconsistent waveform length for file_id={file_id}; "
                    f"expected {n_samples}, got {len(waveform)}. Skipping file."
                )
                waveform_sum = None
                break
            waveform_sum += float(np.sum(waveform))

        if waveform_sum is None:
            continue

        total_sqi_per_subject.append(
            {
                "file_id": file_id,
                "participant_id": participant_id,
                "GLC": glc,
                "waveforms_average": waveform_sum / n_samples,
            }
        )

        if should_show:
            note = f"file_id={file_id}"
            plot_signal_with_sqi(ppg_signal, results_for_plot,
                                 required_ppg_fs, note=note)

    return pd.DataFrame(
        total_sqi_per_subject,
        columns=["file_id", "participant_id", "GLC", "waveforms_average"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare SQI estimators for IEB1 PPG files using DatasetConfig + SigMF."
    )
    parser.add_argument(
        "--dataset-config",
        default="ieb1_multimodal_dataset_folders.json",
        help="Path to dataset config JSON (default: ieb1_multimodal_dataset_folders.json)",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Plot SQI curves and per-file RMSE matrices",
    )
    args = parser.parse_args()

    combined_df = calculate_and_save_sqis(
        dataset_config_file=args.dataset_config,
        should_show=args.show_plots
    )

    dataset_config = DatasetConfig(args.dataset_config)
    output_folder = dataset_config.get_dataset_machine_learning_path()
    output_path = os.path.join(
        output_folder, "sqi_waveforms_average_per_file.csv")
    combined_df.to_csv(output_path, index=False)
    print("Wrote SQI per-file summary to:", output_path)
