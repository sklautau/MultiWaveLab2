'''Create the All-PPGs dataset by merging the three IEB datasets and copying the PPG SIGMF files.'''
from pathlib import Path
import pandas as pd
import shutil
from datasets_util.waveform_files import read_sigmf_file, save_sigmf_signal_rf32_le
from scipy.signal import resample


def resample_signal(data, original_fs, target_fs):
    """
    Resample the signal to the target sampling frequency.

    Parameters:
        data: numpy array of shape (n_samples,)
        original_fs: original sampling frequency
        target_fs: target sampling frequency

    Returns:
        resampled_data: numpy array of shape (new_n_samples,)
    """

    n_samples = len(data)
    duration = n_samples / original_fs
    new_n_samples = int(duration * target_fs)

    resampled_data = resample(data, new_n_samples)

    return resampled_data


def copy_ppg_sigmf_files(
    merged_csv: str,
    original_data_root: str = "../original data",
    output_dataset: str = "../original data/all_datasets",
):
    """
    Copy all PPG SIGMF files referenced in the merged CSV.

    For each row, copies both
        *.sigmf-data
        *.sigmf-meta

    while preserving the relative directory structure.
    """

    dataset_roots = {
        "ieb1": Path(original_data_root) / "dataset_ieb_1",
        "ieb2": Path(original_data_root) / "dataset_ieb_2",
        "ieb3": Path(original_data_root) / "dataset_ieb_3",
    }

    output_root = Path(output_dataset)
    output_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(merged_csv)

    copied = 0
    missing = 0
    resampled = 0

    for _, row in df.iterrows():

        source_root = dataset_roots[row["dataset"]]

        relative_path = Path(row["relative_path"])

        src_data = source_root / "raw" / relative_path
        src_meta = src_data.with_suffix(".sigmf-meta")

        dst_data = output_root / "raw" / relative_path
        dst_meta = dst_data.with_suffix(".sigmf-meta")

        dst_data.parent.mkdir(parents=True, exist_ok=True)

        if row["dataset"] == "ieb1":
            if not src_data.exists() or not src_meta.exists():
                if not src_data.exists():
                    print(f"WARNING: missing {src_data}")
                    missing += 1
                if not src_meta.exists():
                    print(f"WARNING: missing {src_meta}")
                    missing += 1
                continue

            data, input_metadata = read_sigmf_file(str(src_meta))
            original_fs = input_metadata["global"]["core:sample_rate"]
            # check whether the original_fs is 60 Hz, if not, raise an error
            if original_fs != 60:
                raise ValueError(
                    f"Expected original sampling frequency of 60 Hz for dataset ieb1, but got {original_fs} Hz."
                )

            target_fs = 500
            resampled_data = resample_signal(data, original_fs, target_fs)

            # update the metadata before saving so the SigMF helper writes the new sample rate too
            input_metadata["global"]["core:sample_rate"] = target_fs
            save_sigmf_signal_rf32_le(resampled_data, input_metadata, dst_data)
            copied += 1
            resampled += 1
            continue

        if src_data.exists() and src_meta.exists():
            data, input_metadata = read_sigmf_file(str(src_meta))
            # invert the amplitude of IEB2 and IEB3 datasets to match the expected polarity of the PPG signals
            inverted_waveform = data * -1.0
            save_sigmf_signal_rf32_le(
                inverted_waveform, input_metadata, dst_data)
            # Copy the metadata as is
            shutil.copy2(src_meta, dst_meta)
            copied += 1
        else:
            if not src_data.exists():
                print(f"WARNING: missing {src_data}")
                missing += 1
            if not src_meta.exists():
                print(f"WARNING: missing {src_meta}")
                missing += 1

    print()
    print(f"Copied {copied} files.")
    print(f"From copied files, {resampled} were resampled.")
    print(f"Missing {missing} files.")


def merge_ppg_datasets(output_csv: str = "merged_ppg_dataset.csv") -> pd.DataFrame:
    """
    Merge the PPG entries from the three IEB datasets into a single CSV.

    Output columns:
        file_id
        participant_id
        relative_path
        modality
        session_id
        datetime
        GLC
        dataset
    """

    datasets = [
        ("ieb1", "../original_data/dataset_ieb_1/ieb_1_dataset.csv"),
        ("ieb2", "../original_data/dataset_ieb_2/ieb_2_dataset.csv"),
        ("ieb3", "../original_data/dataset_ieb_3/ieb_3_dataset.csv")
    ]

    columns = [
        "file_id",
        "participant_id",
        "relative_path",
        "modality",
        "session_id",
        "datetime",
        "GLC",
        "dataset",
    ]

    merged = []

    for dataset_name, csv_file in datasets:
        print(f"Reading {csv_file}")

        df = pd.read_csv(csv_file)

        # Keep only PPG entries
        df = df[df["modality"].str.lower() == "ppg"].copy()

        # Keep only the desired columns
        df = df[
            [
                "participant_id",
                "relative_path",
                "modality",
                "session_id",
                "datetime",
                "GLC",
            ]
        ].copy()

        # Add dataset identifier
        df["dataset"] = dataset_name

        merged.append(df)

    merged_df = pd.concat(merged, ignore_index=True)

    # Recreate file_id sequentially
    merged_df.insert(
        0,
        "file_id",
        [f"file_id{i}" for i in range(1, len(merged_df) + 1)],
    )

    merged_df = merged_df[columns]

    print("Original number of rows in merged dataset:", len(merged_df))
    # remove some problematic rows
    # if participant_id=ieb_2 and session_id=S5 and datetime="2025-09-09_10-38", remove
    merged_df = merged_df[~((merged_df["participant_id"] == "ieb_02") & (
        merged_df["session_id"] == "S5") & (merged_df["datetime"] == "2025-09-09_10-38"))]
    print("Number of rows in merged dataset after removing problematic rows:", len(merged_df))

    merged_df.to_csv(output_csv, index=False)

    print(f"Created {output_csv}")
    print(f"Total PPG records: {len(merged_df)}")

    return merged_df


if __name__ == "__main__":
    DATASET_ROOT_OUTPUT = Path("../original_data/all_ppgs_dataset")
    csv_file_name = str(DATASET_ROOT_OUTPUT / "all_ppgs_dataset.csv")

    # create the output dataset folder if it doesn't exist
    DATASET_ROOT_OUTPUT.mkdir(parents=True, exist_ok=True)

    # create merged CSV file
    merge_ppg_datasets(csv_file_name)

    copy_ppg_sigmf_files(
        csv_file_name,
        original_data_root="../original_data",
        output_dataset=str(DATASET_ROOT_OUTPUT)
    )
