'''
From a CSV file such as
segment_id,file_id,modality,start_sample,duration,quality_indicator
seg_id0,file_id1,ppg,201,2891,0.9973746
seg_id1,file_id1,ppg,3192,11042,0.9981615
seg_id2,file_id1,ppg,14291,3980,0.9972514
seg_id3,file_id2,ppg,0,925,0.9933409
seg_id4,file_id2,ppg,1695,669,0.9937955
...

We here use a heuristic to select a single segment per file_id, based
on the quality_indicator and duration.
The selected segments are saved to a new CSV file.
'''
import pandas as pd


def select_best_segments(input_file: str, output_file: str):

    # Load the CSV file into a DataFrame
    df = pd.read_csv(input_file)

    # Sort so the top row per file has highest quality, then longest duration.
    best_segments = (
        df.sort_values(
            by=["file_id", "quality_indicator", "duration"],
            ascending=[True, False, False],
        )
        .groupby("file_id", as_index=False)
        .head(1)
    )

    # Save the selected segments to a new CSV file
    best_segments.to_csv(output_file, index=False)


if __name__ == "__main__":
    input_file = r"..\output_ieb1\segments\all_segments_with_quality.csv"
    output_file = r"..\output_ieb1\segments\best_segments_per_file.csv"
    print(f"Input file: {input_file}")
    select_best_segments(input_file, output_file)
    print(f"Output file: {output_file}")
