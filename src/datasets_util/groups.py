'''Provides utilities for working with groups of
columns in metadata organized as Pandas Dataframe.'''

from typing import Union

import pandas as pd

from segments.segments_core import SegmentManager, best_segment


class ColumnsGroup:
    def __init__(self, dataframe: pd.DataFrame, columns_to_be_grouped: list):
        self.columns = columns_to_be_grouped

        # check whether all columns to be grouped are present in the dataframe
        missing_cols = [
            col for col in columns_to_be_grouped if col not in dataframe.columns]
        if missing_cols:
            raise ValueError(f"Columns {missing_cols} not found in dataframe")

        # pandas.api.typing.DataFrameGroupBy instance
        self.df_groupby = dataframe.groupby(columns_to_be_grouped)

    def get_group(self, group_key: Union[tuple, list]) -> pd.DataFrame:
        """
        Get the DataFrame for a specific group key.

        Parameters
        ----------
        group_key : tuple or list
            A tuple of values corresponding to the columns in self.columns, in the same order.

        Returns
        -------
        pd.DataFrame
            The DataFrame containing all rows that match the group key.
        """
        if not isinstance(group_key, (tuple, list)):
            raise ValueError(
                f"group_key must be a tuple or list of values corresponding to columns {self.columns}")
        # normalize list keys to tuples for group lookup
        if isinstance(group_key, list):
            group_key = tuple(group_key)
        if group_key not in self.df_groupby.groups:
            raise KeyError(
                f"Group key {group_key} not found in groups. Available groups: {list(self.df_groupby.groups.keys())}")
        return self.df_groupby.get_group(group_key)


def select_best_in_group(group: pd.DataFrame, segments: pd.DataFrame, required_modalities: set = {"ecg", "ppg", "bioimp"}) -> pd.DataFrame:
    """
    Select the best quality segment for each required modality from a group.

    Parameters
    ----------
    group : pd.DataFrame
        DataFrame with rows for a single participant/session/datetime combination
    segments : pd.DataFrame
        DataFrame with segment information including file_id, modality, and quality_indicator
    required_modalities : set
        Set of modalities to select (default: ecg, ppg, bioimp)

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per modality (the best quality segment for each)
    """
    # Check whether all modalities are present in this group
    modalities_in_group = set(group["modality"].unique())
    if not required_modalities.issubset(modalities_in_group):
        raise ValueError(
            f"Missing modalities in group: {required_modalities - modalities_in_group}")

    # Check whether all records have the same participant_id
    participant_ids = group["participant_id"].unique()
    if len(participant_ids) > 1:
        raise ValueError(
            f"Multiple participant_ids in group: {participant_ids}. Expected only one participant_id per group.")
    participant_id = participant_ids[0]

    # Select the best quality segment for each modality
    # Consider that there may be multiple records for the same modality in this group, corresponding to different files (file_id)
    best_rows = []
    for modality in required_modalities:
        # Get all the file_id's for this modality in this group
        file_ids = group[group["modality"] == modality]["file_id"].unique()
        largest_quality = -float('inf')
        segment_with_larger_quality = None  # initialize best_segment to None
        best_file_id = None
        for file_id in file_ids:
            if file_id not in segments["file_id"].values:
                raise ValueError(
                    f"file_id {file_id} from group not found in segments dataframe")
            # Get segments for this file_id
            best_segment_for_modality_fileid, quality = best_segment(
                segments, file_id, modality)
            if quality > largest_quality:
                largest_quality = quality
                segment_with_larger_quality = best_segment_for_modality_fileid
                best_file_id = file_id

        if segment_with_larger_quality is None:
            raise ValueError(
                f"No segments found for modality {modality} in file_ids {file_ids}")

        # Get the corresponding row from group with this file_id
        original_row_for_best_fileid = group[group["file_id"] == best_file_id]
        if original_row_for_best_fileid.empty:
            raise ValueError(
                f"No original row found in group for best file_id {best_file_id}")

        # Keep a single representative record for this file_id/modality pair
        original_row_for_best_fileid = original_row_for_best_fileid.iloc[[
            0]].reset_index(drop=True)

        # Drop duplicate file_id/modality columns from segment info before concatenation
        segment_info = pd.DataFrame([segment_with_larger_quality]).drop(
            columns=[col for col in ["file_id", "modality"]
                     if col in segment_with_larger_quality],
            errors="ignore"
        )

        best_row = pd.concat(
            [original_row_for_best_fileid, segment_info], axis=1)

        best_rows.append(best_row)

    # convert best_rows into a single DataFrame
    best_rows_df = pd.concat(best_rows, ignore_index=True)

    return best_rows_df


if __name__ == "__main__":
    # Example usage
    original_df = pd.DataFrame({
        'pid': [1, 1, 1, 2, 2, 3],
        'sid': [1, 1, 1, 1, 2, 1],
        'datetime': ['2024-01-01', '2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02', '2024-01-03'],
        'value1': [-1, 10, 20, 30, 40, 50],
        'value2': [1, 2, 3, 4, 5, 6]
    })
    print("Original DataFrame:")
    print(original_df.head())
    columns_to_be_grouped = ['pid', 'sid', 'datetime']
    columnsGroup = ColumnsGroup(original_df, columns_to_be_grouped)
    print("\nGrouped DataFrame:")
    print(columnsGroup.df_groupby.head())

    print("\nIterating over groups:")
    for (pid, sid, dt), group in columnsGroup.df_groupby:
        # group is a DataFrame containing all rows for this participant/session/datetime combination
        print("pid:", pid, "sid:", sid, "dt:", dt, "\ngroup:\n", group, "\n")
        # show number of records in this group
        print(f"  Number of records in this group: {len(group)}")

    # show the number of different groups
    print(f"\nNumber of different groups: {columnsGroup.df_groupby.ngroups}")

    # show in another way:
    groups_list = [(key, group) for key, group in columnsGroup.df_groupby]
    print(f"\nList of group keys and group dataframes: {groups_list}")

    # get the Dataframe for a specific group, e.g. pid=1, sid=1, datetime='2024-01-01'
    specific_group = columnsGroup.get_group([1, 1, '2024-01-01'])
    print("\nDataFrame for group pid=1, sid=1, datetime='2024-01-01':")
    print(specific_group)
