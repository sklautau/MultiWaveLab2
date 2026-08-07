'''
Utilities to read and write SigMF files,
and CSV files.
'''

from sigmf import sigmffile
import numpy as np
from pathlib import Path
import os
import pandas as pd

OVERWRITE_EXISTING_FILES = True  # Set to True to allow overwriting existing files


def load_signal(path: str) -> np.ndarray:
    """
    Read your .sigmf-data or CSV and return numpy array
    @TODO: provide support to checking the sampling frequency from metadata.
    """
    # read csv or sigmf-data file and return signal as numpy array
    if path.endswith(".csv"):
        df = pd.read_csv(path)
        return np.asarray(df['mag_filtered']).flatten()
    elif path.endswith(".sigmf-meta") or path.endswith(".sigmf-data"):
        # implement reading .sigmf-data file and return signal as numpy array
        # you can use read_sigmf_file from datasets_util.waveforms
        signal, metadata = read_sigmf_file(path)
        return np.asarray(signal)
    else:
        raise ValueError(f"Unsupported file format: {path}")


def load_sigmf_file(dataset_file, input_suffix, input_folder, file_id):
    # open dataframe
    dataset_dataframe = pd.read_csv(dataset_file)

    # get the row of the dataframe
    row = dataset_dataframe.loc[dataset_dataframe['file_id'] == file_id]

    relative_path = row['relative_path'].values[0]
    sigmf_input_filename = path_replace_root_folder_and_filename(
        relative_path, "./raw/", input_folder, input_suffix)
    # sigmf_input_filename = os.path.join(input_folder, relative_path)
    print("KKKKKKKK", sigmf_input_filename)
    exit(1)

    ppg_path = Path(sigmf_input_filename)
    # Ensure we point to the metadata file
    if ppg_path.suffix == ".sigmf-data":
        meta_path = ppg_path.with_suffix(".sigmf-meta")
    elif ppg_path.suffix == ".sigmf-meta":
        meta_path = ppg_path
    else:
        raise ValueError(f"Unsupported file type: {sigmf_input_filename}")

    # Load SigMF object
    sig = sigmffile.fromfile(str(meta_path))

    # Read full signal
    data = sig.read_samples()

    # Read complete (full) metadata as a dict of dicts
    input_metadata = sig._metadata  # full metadata
    # input_metadata is a dic of dicts metadata that has the schema:
    # {
    #        "global": {...},        # REQUIRED
    #        "captures": [...],      # REQUIRED
    #        "annotations": [...]    # optional but recommended
    # }

    # Ensure numpy array (SigMF already returns np.ndarray)
    return np.asarray(data), input_metadata


def read_sigmf_file(sigmf_input_filename: str) -> tuple[np.ndarray, dict]:

    ppg_path = Path(sigmf_input_filename)
    # Ensure we point to the metadata file
    if ppg_path.suffix == ".sigmf-data":
        meta_path = ppg_path.with_suffix(".sigmf-meta")
    elif ppg_path.suffix == ".sigmf-meta":
        meta_path = ppg_path
    else:
        raise ValueError(f"Unsupported file type: {sigmf_input_filename}")

    # Load SigMF object
    sig = sigmffile.fromfile(str(meta_path))

    # Read full signal
    data = sig.read_samples()

    # Read complete (full) metadata as a dict of dicts
    input_metadata = sig._metadata  # full metadata
    # input_metadata is a dic of dicts metadata that has the schema:
    # {
    #        "global": {...},        # REQUIRED
    #        "captures": [...],      # REQUIRED
    #        "annotations": [...]    # optional but recommended
    # }

    # Ensure numpy array (SigMF already returns np.ndarray)
    return np.asarray(data), input_metadata


def save_sigmf_signal_rf32_le(signal, metadata, basepath, description=None):
    # --------------------------------------------------
    # Save filtered signal as SigMF (using API)
    # --------------------------------------------------
    basepath = Path(basepath)
    # print(f"CCCCCC  Saving signal to basepath: {basepath}")
    data_file = basepath.with_suffix(".sigmf-data")
    meta_file = basepath.with_suffix(".sigmf-meta")

    # Ensure float32
    signal = np.asarray(signal, dtype=np.float32)
    signal.tofile(str(data_file))  # write waveform data as binary file

    # manipulate metadata to update fields for the filtered signal
    if description is not None:
        metadata["global"][
            "core:description"] = description  # update description in metadata

    # create a SigMFFile object from dic of dicts metadata that has the schema:
    # {
    #        "global": {...},        # REQUIRED
    #        "captures": [...],      # REQUIRED
    #        "annotations": [...]    # optional but recommended
    # }
    sigmffile_with_metadata = sigmffile.SigMFFile(metadata=metadata)
    sigmffile_with_metadata._metadata["global"].pop("core:sha512", None)

    # write the metadata to a file (the API will handle the JSON formatting and schema validation)
    sigmffile_with_metadata.tofile(
        str(meta_file), overwrite=OVERWRITE_EXISTING_FILES)

    return str(data_file), str(meta_file)
