"""
Generate PDF plots of average segment duration (seconds) per participant.

The script relies on DatasetConfig + SegmentManager to load dataset metadata
and the segments CSV, then aggregates duration by participant_id for a
selected modality.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

from datasets_util.naming_conventions import DatasetConfig
from segments.segments_core import SegmentManager, Segmenter


AXIS_LABEL_FONTSIZE = 16
AXIS_TICK_FONTSIZE = 16
TITLE_FONTSIZE = 16


# Ensure project root is in path for imports when running this script directly.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _get_sample_rate_hz(dataset_config: DatasetConfig, modality: str) -> float:
    modality = str(modality).strip().lower()
    if modality == "ppg":
        return float(dataset_config.get_ppg_fs())
    if modality == "ecg":
        return float(dataset_config.get_ecg_fs())
    raise ValueError(
        f"Unsupported modality '{modality}'. Expected one of: ppg, ecg."
    )


def _build_file_to_participant_map(dataset_config: DatasetConfig) -> Dict[str, str]:
    dataset_info = dataset_config.get_dataset_info_dataframe().copy()
    required = {"file_id", "participant_id"}
    missing = sorted(required - set(dataset_info.columns))
    if missing:
        raise ValueError(
            f"Dataset info is missing required columns: {missing}")

    mapping = (
        dataset_info[["file_id", "participant_id"]]
        .drop_duplicates(subset=["file_id"])
        .set_index("file_id")["participant_id"]
        .astype(str)
        .to_dict()
    )
    return mapping


def _aggregate_total_seconds_per_participant(
        dataset_config: DatasetConfig,
        segments_df: pd.DataFrame,
    modality: str,
    aggregation_mode: str,
) -> pd.DataFrame:
    required = {"file_id", "modality", "duration"}
    missing = sorted(required - set(segments_df.columns))
    if missing:
        raise ValueError(
            f"Segments file is missing required columns: {missing}")

    file_to_participant = _build_file_to_participant_map(dataset_config)

    df = segments_df.copy()
    df["file_id"] = df["file_id"].astype(str)
    df["modality"] = df["modality"].astype(str).str.lower()
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
    df = df.dropna(subset=["duration"])
    df = df[df["duration"] > 0]

    selected_modality = str(modality).strip().lower()
    supported_modalities = {"ppg", "ecg"}
    if selected_modality not in supported_modalities:
        raise ValueError(
            f"Unsupported modality '{selected_modality}'. Expected one of: ppg, ecg."
        )

    # Keep only rows from the selected modality.
    df = df[df["modality"] == selected_modality].copy()
    if df.empty:
        raise ValueError(
            f"No rows with modality '{selected_modality}' were found."
        )

    selected_fs_hz = _get_sample_rate_hz(dataset_config, selected_modality)
    df["fs_hz"] = selected_fs_hz
    df["duration_seconds"] = df["duration"] / df["fs_hz"]

    df["participant_id"] = df["file_id"].map(file_to_participant)
    missing_participant = df["participant_id"].isna().sum()
    if missing_participant > 0:
        raise ValueError(
            f"Could not map participant_id for {missing_participant} segment rows. "
            "Check if file_id values exist in DATASET_FILE."
        )

    # Aggregate per file first to make participant-level aggregation explicit.
    file_level = (
        df.groupby(["participant_id", "file_id"],
                   as_index=False)["duration_seconds"]
        .sum()
    )

    selected_mode = str(aggregation_mode).strip().lower()
    if selected_mode == "average":
        out = (
            file_level.groupby("participant_id", as_index=False)
            .agg(
                duration_seconds=("duration_seconds", "mean"),
                n_files=("file_id", "nunique"),
            )
            .sort_values("participant_id")
        )
    elif selected_mode == "total":
        out = (
            file_level.groupby("participant_id", as_index=False)
            .agg(
                duration_seconds=("duration_seconds", "sum"),
                n_files=("file_id", "nunique"),
            )
            .sort_values("participant_id")
        )
    else:
        raise ValueError(
            f"Unsupported aggregation mode '{selected_mode}'. Expected one of: average, total."
        )
    return out


def _plot_participant_duration_pdf(
        participant_df: pd.DataFrame,
        output_pdf: Path,
        title: str,
    x_label: str,
    show_plot: bool = False,
) -> None:
    n_participants = len(participant_df)
    if n_participants == 0:
        raise ValueError("No participant durations to plot.")

    # One participant per row -> horizontal bars, with height scaled by count.
    fig_height = max(10.0, 0.28 * n_participants)
    fig, ax = plt.subplots(figsize=(14, fig_height))

    y_labels = participant_df["participant_id"].astype(str).tolist()
    x_values = participant_df["duration_seconds"].astype(float).tolist()

    ax.barh(y_labels, x_values, color="tab:blue", alpha=0.9)
    ax.set_xlabel(x_label, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Participant ID", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.tick_params(axis="both", labelsize=AXIS_TICK_FONTSIZE)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()

    plt.tight_layout()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_pdf, format="pdf")
    if show_plot:
        plt.show()
    plt.close(fig)


def _plot_participant_duration_cdf_pdf(
        participant_df: pd.DataFrame,
        output_pdf: Path,
        title: str,
    x_label: str,
    show_plot: bool = False,
) -> None:
    if len(participant_df) == 0:
        raise ValueError("No participant durations to plot CDF.")

    # Empirical CDF over participant total durations.
    sorted_durations = (
        participant_df["duration_seconds"]
        .astype(float)
        .sort_values()
        .to_numpy()
    )
    n = len(sorted_durations)
    y = (pd.Series(range(1, n + 1), dtype=float) / float(n)).to_numpy()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.step(sorted_durations, y, where="post", color="tab:orange", linewidth=2)
    ax.set_xlabel(x_label, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Cumulative Probability", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.tick_params(axis="both", labelsize=AXIS_TICK_FONTSIZE)
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_pdf, format="pdf")
    if show_plot:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create PDF plots of participant_id vs average segment duration "
            "(seconds) from a segments CSV for a selected modality."
        )
    )
    parser.add_argument(
        "json_file",
        help="Path to dataset config JSON file used by DatasetConfig.",
    )
    parser.add_argument(
        "--modality",
        choices=["ppg", "ecg"],
        default="ppg",
        help=(
            "Modality to process. Use ppg or ecg. "
            "Default: ppg."
        ),
    )
    parser.add_argument(
        "--aggregation-mode",
        choices=["average", "total"],
        default="total",
        help=(
            "Aggregation mode across files of the same participant. "
            "'average' = mean of file-level durations; "
            "'total' = sum of file-level durations. "
            "Default: total."
        ),
    )
    parser.add_argument(
        "--output-pdf",
        default=None,
        help=(
            "Output PDF path. Default: "
            "<segments_folder>/<segments_stem>_participant_duration_seconds_<modality>.pdf"
        ),
    )
    parser.add_argument(
        "--output-cdf-pdf",
        default=None,
        help=(
            "Output CDF PDF path. Default: "
            "<segments_folder>/<segments_stem>_participant_duration_seconds_<modality>_cdf.pdf"
        ),
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help=(
            "Output CSV path with aggregated durations. "
            "Default: "
            "<segments_folder>/<segments_stem>_participant_duration_seconds_<modality>.csv"
        ),
    )
    parser.add_argument(
        "--sort-by-duration",
        action="store_true",
        help=(
            "Sort participants by total duration (descending) in both plot and CSV. "
            "Default behavior sorts by participant_id."
        ),
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display plots interactively in addition to saving PDF files.",
    )
    args = parser.parse_args()
    modality = str(args.modality).strip().lower()
    aggregation_mode = str(args.aggregation_mode).strip().lower()

    dataset_config = DatasetConfig(args.json_file)
    # segmenter_file_name = dataset_config.get_segmenter_file_name()
    segments_file = dataset_config.get_segments_file_name()
    # segments_file = Segmenter(
    #    segmenter_file_name, args.json_file).get_segments_file_name()
    segment_manager = SegmentManager(dataset_config, segments_file)
    segments_df = segment_manager.get_segments_dataframe()
    seg_stem = Path(segments_file).stem

    participant_df = _aggregate_total_seconds_per_participant(
        dataset_config,
        segments_df,
        modality,
        aggregation_mode,
    )
    participant_df["modality"] = modality
    participant_df["aggregation_mode"] = aggregation_mode

    if args.sort_by_duration:
        participant_df = participant_df.sort_values(
            "duration_seconds", ascending=False
        )
    else:
        participant_df = participant_df.sort_values("participant_id")

    if args.output_pdf is None:
        output_pdf = Path(dataset_config.get_dataset_segments_path()) / (
            f"{seg_stem}_participant_duration_seconds_{modality}.pdf"
        )
    else:
        output_pdf = Path(args.output_pdf)

    if args.output_csv is None:
        output_csv = Path(dataset_config.get_dataset_segments_path()) / (
            f"{seg_stem}_participant_duration_seconds_{modality}.csv"
        )
    else:
        output_csv = Path(args.output_csv)

    if args.output_cdf_pdf is None:
        output_cdf_pdf = Path(dataset_config.get_dataset_segments_path()) / (
            f"{seg_stem}_participant_duration_seconds_{modality}_cdf.pdf"
        )
    else:
        output_cdf_pdf = Path(args.output_cdf_pdf)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    participant_df.to_csv(output_csv, index=False)

    _plot_participant_duration_pdf(
        participant_df,
        output_pdf,
        title=(
            f"{aggregation_mode.capitalize()} Segment Duration per Participant "
            f"({modality.upper()})"
        ),
        x_label=f"{aggregation_mode.capitalize()} Duration (seconds)",
        show_plot=args.show_plots,
    )

    _plot_participant_duration_cdf_pdf(
        participant_df,
        output_cdf_pdf,
        title=(
            f"CDF of {aggregation_mode.capitalize()} Segment Duration per Participant "
            f"({modality.upper()})"
        ),
        x_label=f"{aggregation_mode.capitalize()} Duration per Participant (seconds)",
        show_plot=args.show_plots,
    )

    average_duration_seconds = participant_df["duration_seconds"].mean()

    print(f"Saved PDF: {output_pdf}")
    print(f"Saved CDF PDF: {output_cdf_pdf}")
    print(f"Saved CSV: {output_csv}")
    print(f"Modality: {modality.upper()}")
    if aggregation_mode == "average":
        print("Aggregation: per-file sum, then per-participant average")
    else:
        print("Aggregation: per-file sum, then per-participant total")
    print(
        f"Average duration per participant (seconds): {average_duration_seconds:.3f}")
    print(f"Participants plotted: {len(participant_df)}")


if __name__ == "__main__":
    main()
