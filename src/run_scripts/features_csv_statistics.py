'''
From command line, read a csv pandas file and
provide statistics about the number of features for each modality (ECG, PPG, Bioimpedance),
such as:
Number of ECG features: 0
Number of PPG features: 20
Number of Bioimpedance features: 0
'''


import argparse
import pandas as pd

segments_metadata = ['segment_id',
                     'file_id',
                     'modality',
                     'start_sample',
                     'duration',
                     'quality_indicator'
                     ]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CSV statistics."
    )

    parser.add_argument(
        "features_file",
        metavar="FEATURES_FILE",
        help="Path to the input features CSV file (e.g.  ../outputs/output_ieb1/features/features14_train_selected_features.csv)."
    )
    # parse command line
    args = parser.parse_args()

    features_file = args.features_file

    # read the features file
    features_df = pd.read_csv(features_file)

    # remove the columns that have suffixes as the segments_metadata
    # add prefix ecg to the columns that have suffixes as the segments_metadata
    for col in features_df.columns:
        for suffix in segments_metadata:
            if col.endswith(suffix):
                # remove the column
                features_df = features_df.drop(columns=[col])

    # count the number columns starting with ecg_, ppg_, bioimpedance_
    ecg_columns = [
        col for col in features_df.columns if col.startswith("ecg_")]
    ppg_columns = [
        col for col in features_df.columns if col.startswith("ppg_")]
    bioimpedance_columns = [
        col for col in features_df.columns if col.startswith("bio")]

    print(f"Number of ECG features: {len(ecg_columns)}")
    print(f"Number of PPG features: {len(ppg_columns)}")
    print(f"Number of Bioimpedance features: {len(bioimpedance_columns)}")
