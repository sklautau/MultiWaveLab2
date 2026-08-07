# MultiWaveLab

## Installing MultiWaveLab

### On Windows
Using Windows CMD, clone this repository. Then create a virtual environment with Python 3.10.13. Use pip to install the required packages from file ```requirements.txt```. We will be using two biomedical signal processing libraries: PyPPG and NeuroKit2. PyPPG has dependencies on some outdated packages, such as Numpy 1.x. To avoid conflicts with NeuroKit2, which is more actively maintained and has more frequent updates, we must install PyPPG with no dependencies. Therefore, after installing the requirements, run ```pip install pyPPG --no-deps```.  

### On Linux
Using Ubuntu Bash, first install Miniconda or Anaconda and then clone this repository. With Conda, the installation is the same as the one for Windows. Create a virtual environment with Python 3.10.13. Then use pip to install the required packages from file ```requirements.txt```. After installing the requirements, run ```pip install pyPPG --no-deps```.

##### Miniconda installation example
````bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
````

### MultiWaveLab full installation using Conda:

```bash
conda create -n pyppg_env python=3.10.13
conda activate pyppg_env
pip install -r requirements.txt
pip install pyPPG --no-deps
```

## Relying on SigML and Pandas
This repository  provides Python for dealing with files using Pandas DataFrame and SigMF API for data storage in machine learning applications.
The goal of this code is to provide a convenient way of dealing with waveforms (stored in binary files) and the labeling of associated events (stored in files representing Pandas Dataframes).

When implementing machine learning models, it is often straightforward to work with images that are completely independent of one another. However, when multiple images are associated with a single individual (for example, MRI scans from the same patient), care must be taken to prevent data leakage when partitioning the data into training and test sets. This issue becomes more complex when dealing with time-domain waveforms such as speech, ECG, EEG, and similar signals, which are associated to a specific person and require the labeling of key events. This repository assumes time-domain waveforms obtained through periodic sampling and provides code to efficiently store both the signals and their associated labels. It also provides code to help organizing the files into folders, and quickly getting the paths for reading and writing them. Besides Pandas DataFrame and SigMF, it also uses the NeuroKit2 definitions, which is a library for neurophysiological data analysis that relies on Pandas.

## Files and their locations: design guidelines

We adopt both the SIGMF API and Pandas, to get the better of each world: storing binary files in an efficient and portable way, manipulating Pandas' Dataframes and preparing datasets for ML.

The mains assumptions are:
1) Each SigMF binary file with waveform samples has its corresponding JSON metadata file. We use this metadata file to register sampling frequency, quantization step and other things related to that specific waveform itself. SIGMF provides support for annotations (labeling) but for most labeling events we will use Pandas Dataframes stored in text CSV files. For instance, For a given dataset `myecg`, we will have always a pair of SiGMF files for a given waveform file:
```
/original_data/myecg_data/raw/user1/fileA.sigmf-data
/original_data/myecg_data/raw/user1/fileA.sigmf-meta
...
/original_data/myecg_data/raw/user5/fileX.sigmf-data
/original_data/myecg_data/raw/user5/fileX.sigmf-meta
```
2) Each dataset is stored in root folder DATASET_ROOT_PATH. Within this root folder, a CSV dataframe file with extension csv and suffix _dataset (e.g. `myecg_dataset.csv`) has columns `file_id` and `relative_path`. The concatenation (via, e.g., os.path.join()) of DATASET_ROOT_PATH and relative paths leads to the waveform (not the metadata) file path corresponding to the unique file_id in the dataset (listed e.g. in `myecg_dataset.csv`). The CSV can be augmented with extra columns, providing extra information about each file in the dataset. For instance, besides a file_id, columns can include patient_id, disease, etc. Recall that the sample rate and other information about a specific waveform file is stored in the corresponding SigMF metadata file.
3) All the original data in DATASET_ROOT_PATH must be placed in a folder called "raw". An example of contents of a DATASET_ROOT_PATH=/original_data/myecg_data is:
```
/original_data/myecg_data
04/18/2026  10:45 PM            15,614 myecg_dataset.csv
04/18/2026  10:43 PM    <DIR>          raw
```
And inside the raw folder, one can find any structure of nested subfolders, such as:
```
/original_data/myecg_data/raw/user1/
/original_data/myecg_data/raw/user2/
/original_data/myecg_data/raw/groupA/user1/
/original_data/myecg_data/raw/groupA/user2/
/original_data/myecg_data/raw/groupD/user192/
...
```
4) We will deal with four major categories of information and respective files: a) waveforms, b) segments, c) events and d) ML features.
The a) waveforms are periodically sampled signals, similar to the original (raw) files that are assumed to be waveforms. But waveforms can also be generated by ourselves via filtering, etc., with the same or different sample rate when compared to the raw signals.
The b) segments are variable-duration sequences of waveform samples, and are denoted by their starting sample and duration (as in NeuroKit2), and associated to their corresponding waveform file via the unique file_id. The c) events are markers and other information extracted from waveforms or segments, such as the peaks of ECG pulses. Finally, d) ML features (or simply features) are used to compose examples as pairs (X, y) of input vector X and label y for supervised learning or simply X for unsupervised ML.
5) As mentioned, waveforms are stored as SigMF files, while the other 3 categories are stored as text CSV representing Dataframes. An exception can be made when feature files become too large and are stored in binary format.
6) All the waveforms we generate from a given dataset are stored in a subfolder of the GENERATED_WAVEFORMS_PATH, keeping the same relative path as in DATASET_ROOT_PATH. Any specific waveform processing that generates must have a unique name (e.g., lowpassfiltered_ecg), called WAVEFORM_ID, which is used as the name of the subfolder under GENERATED_WAVEFORMS_PATH, to compose GENERATED_WAVEFORMS_PATH/WAVEFORM_ID. And WAVEFORM_ID is also used as the suffix of all files under this subfolder, to facilitate identification in case a file is shared without the complete path structure. For instance, assume the research is comparing two alternative filtering strategies, identified as lowpassfiltered_ecg and bandpassfiltered_ecg. The GENERATED_WAVEFORMS_PATH=/my_ecg_generated_data/ could be:
```
/my_ecg_generated_data/lowpassfiltered_ecg/user1/
/my_ecg_generated_data/lowpassfiltered_ecg/user2/
/my_ecg_generated_data/lowpassfiltered_ecg/groupA/user1/
/my_ecg_generated_data/lowpassfiltered_ecg/groupA/user2/
/my_ecg_generated_data/lowpassfiltered_ecg/groupD/user192/
...
/my_ecg_generated_data/bandpassfiltered_ecg/user1/
/my_ecg_generated_data/bandpassfiltered_ecg/user2/
/my_ecg_generated_data/bandpassfiltered_ecg/groupA/user1/
/my_ecg_generated_data/bandpassfiltered_ecg/groupA/user2/
/my_ecg_generated_data/bandpassfiltered_ecg/groupD/user192/
...
```
And the suffix (filtered_ecg in the case above) is appended to all files. For example, if the original name is `test1.sigmf-data` and `WAVEFORM_ID=lowpassfiltered_ecg`, the new name is `test1_lowpassfiltered_ecg.sigmf-data`.

7) To go quicker in a loop over the whole data, each GENERATED_WAVEFORMS_PATH/WAVEFORM_ID folder can have its own _dataset.csv file with updated file names. Alternatively, this information can be generated on-the-fly. As for the waveform files, this _dataset.csv file is created by using WAVEFORM_ID as a suffix for the original file name. For instance, the original `myecg_dataset.csv` in DATASET_ROOT_PATH would be called `myecg_lowpassfiltered_ecg_dataset.csv` in `/my_ecg_generated_data/lowpassfiltered_ecg/`.

8) As mentioned, in a given dataset, all waveform files must have a unique ID called `file_id`. This ID simplifies the CSV Dataframe files for segments and events, which do need to have a nested structure. For instance, when one file ID needs to be associated to several events (several rows in the CSV Dataframe file), the rows will use a column called ``label'' (adopting the same nomenclature as NeuroKit2) to distinguish the events of a given `file_id`. The `file_id` in a given file stored in DATASET_ROOT_PATH will be repeated in all its "children" files in GENERATED_WAVEFORMS_PATH.

9) All files with segments go in the SEGMENTS_PATH, while the events go in the EVENTS_PATH.
10) A dataset is not described by DATASET_ROOT_PATH, but by a _dataset.csv file called DATASET_FILE. For instance, a folder may contain several different signals (ECG, EEG, etc.), and the user can filter only ECG files and create a _dataset.csv file with ECG only (the other signals will be ignored). Hence, DATASET_ROOT_PATH can be obtained as the folder of DATASET_FILE (e.g., with `DATASET_ROOT_PATH=os.path.dirname(DATASET_FILE)`).
11) The distinct sets (train, test, validation, etc.), have a file name with extension txt.

## JSON configuration file

An example:

{
  "DATASET_FILE": "..\\original_data\\dataset_ieb_1\\ieb_1_dataset.csv",
  "GENERATED_WAVEFORMS_PATH": "..\\output_ieb1\\waveforms",
  "SEGMENTS_PATH": "..\\output_ieb1\\segments",
  "EVENTS_PATH": "..\\output_ieb1\\events",
  "FEATURES_PATH": "..\\output_ieb1\\features",
  "MACHINE_LEARNING_PATH": "..\\output_ieb1\\ml",
  "PPG_FS": 60
}

## Software to loop over files and process information

12) The provided software has a class `DatasetConfig` that can be imported with
```
from datasets_util.naming_conventions import DatasetConfig
```
and facilitates organizing the files in sensible paths. In this code, "raw" indicates the original data, while "gen" (generation) indicates a folder in the GENERATED_WAVEFORMS_PATH. The following methods allow to obtain the relative path:
- get_raw_relative_path(file_id)
- get_gen_relative_path(file_id, waveform_id)
- get_relative_path(file_id)

while the following provide the complete (which may be a relative path, eventually not a full path):
- get_raw_complete_path(file_id)
- get_gen_complete_path(file_id,waveform_id)

The method get_raw_complete_path() uses DATASET_ROOT_PATH / "raw" / relative_path to concatenate with the relative path, while get_gen_complete_path() uses GENERATED_WAVEFORMS_PATH / relative_path (without the "raw" subfolder).

13) Creating new waveforms follow a well-defined pipeline. The method ppg_WAVEFORM_ID_file_processing implements the processing of a single PPG file, while the method ppg_processing_pipeline goes over all files. For instance, for WAVEFORM_ID=filtering and quality:
```
def ppg_bandpass_file_processing(ppg_filename: str) -> tuple[np.ndarray, np.ndarray, dict]:
def ppg_quality_file_processing(ppg_filename: str) -> tuple[np.ndarray, np.ndarray, dict]:

These methods are called, respectively, by:
    ppg_processing_pipeline(
        dataset_config_file, input_waveform_id="raw", output_waveform_id="bandpass")
    ppg_processing_pipeline(
        dataset_config_file, input_waveform_id="bandpass", output_waveform_id="quality")
```

14) The methods that extract features generate one feature vector
per segment.

15) Segments are generic. But there are 3 main types here:
a) wave: complete waveform
b) window: fixed-duration segments, also called windows
c) pulse: short segments representing a single pulse
d) generic: variable-duration segments

The module segments.py defines a segment, which is composed of:
segment_id,file_id,modality,start_sample,duration,quality_indicator
seg_id0,file_id2,ppg,0,80840,0.993
seg_id1,file_id2,ppg,81183,14800,0.986
...

We organize CSV files into 2 categories:
all: segments: does not filter nor use a segmenter (by SQI, etc)
selected: _segments: just selected segments

What defines a segmenter?
- type: window, pulse, generic
- input_waveform_id ==> THIS DOES NOT MATTER, BECAUSE SQI SUFFICES
- sqi_id or sqi_method - input_waveform_id
- sqi_threshold
- mean or median
- output file name

..
seg_id282,file_id1,ppg,17609,600,0.997990608215332
seg_id283,file_id1,ppg,17669,600,0.997990608215332
seg_id0,file_id2,ppg,0,600,0.9974362850189209
seg_id1,file_id2,ppg,60,600,0.9974362850189209
seg_id2,file_id2,ppg,120,600,0.9974362850189209
seg_id3,file_id2,ppg,180,600,0.997436285018920
..

The files in segmentation folder are:
select_segments.py
features_best_segments.py
save_good_segments.py
segments_to_fixed_duration_windows.py


# Links

## Installation of SigMF in Python (there is also code for C++ and others)
* https://pypi.org/project/SigMF/

## Source code of SigMF in Python
* https://github.com/sigmf/sigmf-python

## SigMF API documentation (spreaded out at different sites):
* https://github.com/sigmf/SigMF
* https://sigmf.org/index.html
* https://sigmf.readthedocs.io/en/latest/quickstart.html

## Dataframes and Pandas
* https://pandas.pydata.org/
* https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html

## NeuroKit
* https://neuropsychology.github.io/NeuroKit/

