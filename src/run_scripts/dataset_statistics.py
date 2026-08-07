import argparse
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CSV statistics from files such as ieb_2_dataset.csv."
    )

    parser.add_argument(
        "dataset_file",
        metavar="DATASET_FILE",
        help="Path to the input dataset CSV file (e.g. ../original_data/dataset_ieb_2/ieb_2_dataset.csv)."
    )
    # parse command line
    args = parser.parse_args()

    dataset_file = args.dataset_file

    # read the dataset file
    dataset_df = pd.read_csv(dataset_file)

    # group by participant_id,session_id,datetime and count the number of rows per group
    grouped_df = dataset_df.groupby(
        ['participant_id', 'session_id', 'datetime']).size().reset_index(name='count')

    # print the grouped data
    print(grouped_df)

    # for each group, read the modality column and
    # count the number of ecg, ppg and bioimpedance modalities per group

    print(
        f"Participant  Session:  Datetime:  ECG:  PPG:  Bioimpedance:  GLC")
    all_glucose_values = []
    for _, group in grouped_df.iterrows():
        participant_id = group['participant_id']
        session_id = group['session_id']
        datetime = group['datetime']

        # filter the dataset_df for the current group
        filtered_df = dataset_df[(dataset_df['participant_id'] == participant_id) &
                                 (dataset_df['session_id'] == session_id) &
                                 (dataset_df['datetime'] == datetime)]

        # count the number of ecg, ppg and bioimpedance modalities
        ecg_count = (filtered_df['modality'] == 'ecg').sum()
        ppg_count = (filtered_df['modality'] == 'ppg').sum()
        bioimpedance_count = (
            filtered_df['modality'] == 'bioimp').sum()

        # for this group, collect the GLC (glucose) value and check whether
        # it is the same for all files in the group, if not, print a warning
        glc_values = filtered_df['GLC'].unique()
        if len(glc_values) > 1:
            print(
                f"Warning: Participant: {participant_id}, Session: {session_id}, Datetime: {datetime}, GLC values: {glc_values}")
        glc = glc_values[0] if len(glc_values) > 0 else None
        all_glucose_values.append(glc)
        print(
            f"{participant_id} & {session_id} & {datetime} & {ecg_count} & {ppg_count} & {bioimpedance_count} & {glc} \\\\")

    print("\nNumber of groups (GLC measurements):", len(grouped_df))

    # write all_glucose_values to a csv file
    glucose_df = pd.DataFrame(all_glucose_values, columns=['GLC'])
    glucose_df.to_csv('glucose_values.csv', index=False)
    print("Saved glucose values to glucose_values.csv")
