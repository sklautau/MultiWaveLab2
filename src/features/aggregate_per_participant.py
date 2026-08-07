'''
Aggregate feature vectors per participant or session ID.
Instead of using an average pulse, take average of feature vectors,
for the so-called "average-feature" features.
For IEB3 dataset:
    - In this case, each session ID will have a single feature vector,
which is the average of all feature vectors for that session ID.
For IEB1 dataset:
    - Each participant ID will have a single feature vector.
'''

import pandas as pd
import os

TASK = 1  # 1) "tcc_guilherme" or 2) "dissertation_luis"

if TASK == 1:
    INPUT_FILE_NAME = r"..\output_ieb1\ml\ufsc_tcc_ppg_features.csv"
    TARGET_COLUMN = 'GLC'
    METADATA_COLUMNS = [line.strip() for line in open(
        "../input_ieb1/metadata/guilherme_metadata_columns.txt").read().splitlines()[1:]]
    ID_COLUMN = "ID"
    # Specify the column used to group by and generate a single feature vector per unique ID. This should be the column that identifies each participant or session.
    # Change this to the appropriate column name for your dataset
    AGGREGATION_COLUMN = ID_COLUMN

elif TASK == 2:
    INPUT_FILE_NAME = r"..\output_ieb1\ml\ufsc_dissertation_ppg_features.csv"
    TARGET_COLUMN = 'GLC'
    METADATA_COLUMNS = [line.strip() for line in open(
        "../input_ieb1/metadata/luis_metadata_columns.txt").read().splitlines()[1:]]
    ID_COLUMN = "SUBJECT_ID"
    # Specify the column used to group by and generate a single feature vector per unique ID. This should be the column that identifies each participant or session.
    # Change this to the appropriate column name for your dataset
    AGGREGATION_COLUMN = ID_COLUMN


def statiscal_summary(df: pd.DataFrame) -> None:
    """
    Prints a statistical summary of the DataFrame, including the number of rows, columns, and data types.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    """
    print(f"DataFrame shape: {df.shape}")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    print("Data types:")
    print(df.dtypes)
    print("Statistical summary:")
    print(df.describe())


def convert_id_into_numeric(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """
    Converts the specified ID column into numeric values.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the ID column.
    id_col (str): The name of the column to convert to numeric.

    Returns:
    pd.DataFrame: A new DataFrame with the specified ID column converted to numeric values.
    """
    # check whether the items in specified ID column start with #
    # and remove the # char if so
    if df[id_col].astype(str).str.startswith('#').any():
        df[id_col] = df[id_col].astype(str).str.lstrip('#')

    df[id_col] = pd.to_numeric(df[id_col])
    return df


def list_numeric_columns(df: pd.DataFrame) -> list:
    """
    Returns a list of numeric columns in the DataFrame
    and also the list of non-numeric columns.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    list: A list of column names that are numeric and a list of column names that are non-numeric.
    """
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    non_numeric_columns = df.select_dtypes(exclude=['number']).columns.tolist()
    return numeric_columns, non_numeric_columns


def take_average_feature_vectors(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """
    Takes the average of feature vectors for each unique ID in the specified column.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing feature vectors and IDs.
    id_col (str): The name of the column containing unique IDs (e.g., 'session_id' or 'participant_id').

    Returns:
    pd.DataFrame: A new DataFrame with one row per unique ID, containing the average feature vector.
    """
    # first check the hypothesis that all rows grouped by the specified ID column have the same GLC value
    if not df.groupby(id_col)[TARGET_COLUMN].apply(lambda x: x.nunique() == 1).all():
        raise ValueError(
            "Rows grouped by the specified ID column do not have the same GLC value")

    # Group by the specified ID column and calculate the mean for each group
    averaged_df = df.groupby(id_col).mean().reset_index()

    return averaged_df


if __name__ == "__main__":
    # Load the dataset
    df = pd.read_csv(INPUT_FILE_NAME)
    print(f"Loaded dataset {INPUT_FILE_NAME} with shape: {df.shape}")

    # Drop metadata columns to focus on feature vectors
    df = df.drop(columns=METADATA_COLUMNS)

    df = convert_id_into_numeric(df, ID_COLUMN)

    numeric_columns, non_numeric_columns = list_numeric_columns(df)
    print(f"Numeric columns: {numeric_columns}")
    print(f"Non-numeric columns: {non_numeric_columns}")

    # Take the average of feature vectors
    averaged_df = take_average_feature_vectors(df, AGGREGATION_COLUMN)

    # Rename column ID to participant_id
    averaged_df = averaged_df.rename(columns={ID_COLUMN: "participant_id"})

    # move the participant_id column to the first position
    cols = averaged_df.columns.tolist()
    cols.insert(0, cols.pop(cols.index("participant_id")))
    averaged_df = averaged_df[cols]

    # move the GLC to the last position
    cols = averaged_df.columns.tolist()
    cols.append(cols.pop(cols.index(TARGET_COLUMN)))
    averaged_df = averaged_df[cols]

    statiscal_summary(averaged_df)

    # Save the averaged DataFrame to a new CSV file
    # extract filename without extension from INPUT_FILE_NAME
    filename_without_extension = os.path.basename(
        INPUT_FILE_NAME).split('.')[0]
    output_file_name = filename_without_extension + "_aggregated.csv"
    output_file_path = os.path.join(
        os.path.dirname(INPUT_FILE_NAME), output_file_name)
    # Save the averaged DataFrame to a new CSV file
    # check whether the file already exists, if so, delete it
    if os.path.exists(output_file_path):
        print(
            f"Warning: output file {output_file_path} already existed and was overwritten!")
    averaged_df.to_csv(output_file_path, index=False)

    print(f"Averaged feature vectors saved to {output_file_path}")
