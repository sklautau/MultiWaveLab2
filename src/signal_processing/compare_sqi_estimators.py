'''
Compare SQI estimators for IEB1 PPG files using DatasetConfig + SigMF.
It assumes the text file ieb1_best_sqi_files.txt exists in the dataset machine learning path, with contents:
hu_89
hu_52
hu_90
...
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

# Ensure project root is available when script is run as a standalone file.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _read_ppg_for_comparison(
    dataset_config: DatasetConfig,
    file_id: str,
    required_ppg_fs: int,
) -> tuple[np.ndarray, str]:
    """Load preferred PPG input for SQI comparison.

    Prefer already-generated filtered waveform. If it does not exist, read raw
    SigMF and run the same filtering used in the current PPG pipeline.
    """
    filtered_path = dataset_config.get_gen_complete_path(file_id, "filtered")
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


def compare_sqi_for_ieb1_dataset(
    dataset_config_file: str,
    should_show: bool = False,
    limit_files: int = -1,
) -> None:
    """Compare SQI estimators over all IEB1 PPG files from the dataset config."""
    dataset_config = DatasetConfig(dataset_config_file)
    required_ppg_fs = dataset_config.get_ppg_fs()

    df = dataset_config.get_dataset_info_dataframe()
    df = df[df["modality"].str.contains("ppg", na=False)]

    if limit_files > 0:
        df = df.head(limit_files)

    print(f"Found {len(df)} PPG files to process (fs={required_ppg_fs} Hz)")

    all_files_mse: list[dict] = []
    summary_rows: list[dict[str, Any]] = []

    for idx, row in enumerate(df.itertuples(index=False), start=1):

        file_id = str(row.file_id)
        participant_id = str(getattr(row, "participant_id", "unknown"))
        print(f"\n[{idx}/{len(df)}] file_id={file_id} participant={participant_id}")

        try:
            ppg_signal, signal_path = _read_ppg_for_comparison(
                dataset_config, file_id, required_ppg_fs
            )
            print(f"  Input signal: {signal_path}")
            print(f"  Samples: {len(ppg_signal)}")
        except Exception as exc:
            print(
                f"  ERROR reading/processing signal for file_id={file_id}: {exc}")
            continue

        # print("Number of samples in PPG signal:", len(ppg_signal))

        results = run_all_sqi_estimators(ppg_signal, required_ppg_fs)
        if len(results) < 2:
            print(
                "  WARNING: fewer than 2 valid SQI estimators for this file, skipping MSE")
            continue

        mse_results = pairwise_mse(
            results, key_name="sqi_for_each_signal_sample")
        all_files_mse.append(mse_results)

        summary: dict[str, Any] = {
            "file_id": file_id,
            "participant_id": participant_id,
            "input_signal_path": signal_path,
            "num_samples": len(ppg_signal),
        }
        for estimator_result in results:
            name = estimator_result["name"]
            key = name.lower().replace(" ", "_").replace(
                "-", "").replace("(", "").replace(")", "")
            sqi_values = np.asarray(
                estimator_result["sqi_for_each_signal_sample"], dtype=float)
            summary[f"{key}_mean"] = float(np.nanmean(sqi_values))
            summary[f"{key}_std"] = float(np.nanstd(sqi_values))
        summary_rows.append(summary)

        print("  MSE between estimators:", mse_results)

        if should_show:
            note = f"file_id={file_id} participant={participant_id}"
            plot_signal_with_sqi(ppg_signal, results,
                                 required_ppg_fs, note=note)
            plot_rmse_matrix(mse_results, notes=note)

    if not all_files_mse:
        print("No valid PPG files were processed. Nothing to aggregate.")
        return

    avg_mse = accumulate_mse_dicts(all_files_mse)
    print("\nAverage MSE between SQI estimators across files:")
    print(avg_mse)
    plot_rmse_matrix(avg_mse, notes="Average SQI across IEB1 PPG files")

    summary_df = pd.DataFrame(summary_rows)
    output_dir = dataset_config.get_dataset_machine_learning_path()
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir, "sqi_estimators_comparison_per_file.csv")
    summary_df.to_csv(output_file, index=False)
    print(f"Saved per-file SQI summary to: {output_file}")


def show_sqi_waveforms_based_on_id_list(dataset_config_file: str,
                                        should_show: bool = False,
                                        limit_files: int = -1,
                                        ) -> None:
    '''
    Given the text file ieb1_best_sqi_files.txt, with contents:
    hu_89
    hu_52
    hu_90
    ...
    read it as a dataframe, and for each file_id, read the
    corresponding PPG signal and compute SQI estimators in the
    order provide by this file.
    Then, use plot_signal_with_sqi() to plot the corresponding SQI signals.
    '''
    dataset_config = DatasetConfig(dataset_config_file)
    required_ppg_fs = dataset_config.get_ppg_fs()
    # read the file ieb1_best_sqi_files.txt
    input_dir = dataset_config.get_dataset_machine_learning_path()
    file_list_path = os.path.join(
        input_dir, "ieb1_best_sqi_files.txt")
    if not os.path.exists(file_list_path):
        print(
            f"File {file_list_path} does not exist. Please create it with the list of file_ids to process.")
        return

    df = pd.read_csv(file_list_path, header=None, names=["file_id"])

    # invert the order in file_list_path to have the worst SQI files first
    # df = df.iloc[::-1]

    if limit_files > 0:
        df = df.head(limit_files)
    # go over the file_ids in the order provided by the file
    for idx, row in enumerate(df.itertuples(index=False), start=1):
        file_id = str(row.file_id)

        # if file_id != "file_id89" and file_id != "file_id39":
        #    continue

        print(f"\n[{idx}/{len(df)}] file_id={file_id}")

        # convert file_id08 to file_id8, file_id09 to file_id9, etc.
        if file_id.startswith("file_id0"):
            file_id = "file_id" + file_id[8:]
            print(f"  Converted file_id to {file_id}")

        try:
            ppg_signal, signal_path = _read_ppg_for_comparison(
                dataset_config, file_id, required_ppg_fs
            )
            print(f"  Input signal: {signal_path}")
            print(f"  Samples: {len(ppg_signal)}, shape = {ppg_signal.shape}")
        except Exception as exc:
            print(
                f"  ERROR reading/processing signal for file_id={file_id}: {exc}")
            continue

        # AK
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

        if should_show:
            note = f"file_id={file_id}"
            plot_signal_with_sqi(ppg_signal, results,
                                 required_ppg_fs, note=note)


def main_show_sqis_in_order():
    parser = argparse.ArgumentParser(
        description="Compare SQI estimators for IEB1 PPG files using DatasetConfig + SigMF."
    )
    parser.add_argument(
        "--dataset-config",
        default="../MultiWaveLab-Inputs/input_ieb1/exp7.json",
        help="Path to dataset config JSON (default: ieb1_multimodal_dataset_folders.json)",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Plot SQI curves and per-file RMSE matrices",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=-1,
        help="Process only the first N PPG files (default: all)",
    )
    args = parser.parse_args()

    show_sqi_waveforms_based_on_id_list(
        dataset_config_file=args.dataset_config,
        should_show=args.show_plots,
        limit_files=args.limit_files,
    )


def main_compare_sqi_estimators():
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
    parser.add_argument(
        "--limit-files",
        type=int,
        default=-1,
        help="Process only the first N PPG files (default: all)",
    )
    args = parser.parse_args()

    compare_sqi_for_ieb1_dataset(
        dataset_config_file=args.dataset_config,
        should_show=args.show_plots,
        limit_files=args.limit_files,
    )


if __name__ == "__main__":
    # main_compare_sqi_estimators()
    main_show_sqis_in_order()
