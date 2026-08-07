"""
GUI script to visualize segments over complete signals.

This version follows the current codebase pattern:
1) Build DatasetConfig from JSON file.
2) Resolve segment file from Segmenter config (or CLI override).
3) Build SegmentManager and iterate its dataframe rows.
"""

from datasets_util.waveform_files import read_sigmf_file
from segments.segments_core import SegmentManager, Segmenter
from datasets_util.naming_conventions import DatasetConfig
import argparse
import os
import sys
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import Button, TextBox
from signal_processing.ppg import QUALITY_PIPELINES


PIPELINE_COLORS = [
    "#D81B60",
    "#FFC107",  # roe
    "#1E88E5",  #
    "#004D40",  #
]

PIPELINE_MARKERS = [
    "o",  # circle
    "s",  # square
    "^",  # triangle
    "D",  # diamond
    "v",  # inverted triangle
]

PPG_COLOR = "#232222"


ALL_PIPELINES = list(QUALITY_PIPELINES.keys())

# Ensure project root is in path for imports when running this script directly.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def scale_signal(signal: np.ndarray, target_min: float = 0.1, target_max: float = 0.9) -> np.ndarray:
    """Scale signal to a specified range."""
    min_val = np.min(signal)
    max_val = np.max(signal)
    if max_val - min_val == 0:
        return np.full_like(signal, target_min)  # Avoid division by zero
    scaled_signal = (signal - min_val) / (max_val - min_val)
    scaled_signal = scaled_signal * (target_max - target_min) + target_min
    return scaled_signal


class SegmentVisualizer:
    """GUI-based segment visualizer with navigation controls."""

    def __init__(
        self,
        dataset_config: DatasetConfig,
        segment_manager: SegmentManager,
        file_id: Optional[str] = None,
        modality: Optional[str] = None,
        start_index: int = 0,
    ) -> None:
        self.dataset_config = dataset_config
        self.waveform_ids = list(ALL_PIPELINES)

        segments_df = segment_manager.get_segments_dataframe().copy()
        required_cols = {
            "segment_id",
            "file_id",
            "modality",
            "start_sample",
            "duration",
        }
        missing = sorted(required_cols - set(segments_df.columns))
        if missing:
            raise ValueError(
                f"Missing required columns in segments dataframe: {missing}"
            )

        if file_id:
            segments_df = segments_df[segments_df["file_id"] == file_id]
        if modality:
            segments_df = segments_df[segments_df["modality"] == modality]

        if segments_df.empty:
            raise ValueError("No segments available after applying filters.")

        segments_df = segments_df.sort_values(
            by=["file_id", "modality", "start_sample", "segment_id"]
        ).reset_index(drop=True)

        self.segments_df = segments_df
        self.current_segment_idx = max(
            0, min(start_index, len(self.segments_df) - 1))

        # Cache complete waveforms to avoid re-reading on each navigation.
        self._signal_cache: dict[tuple[str, str, str],
                                 tuple[np.ndarray, float]] = {}

        self.fig = None
        self.ax = None
        self.info_text = None
        self.textbox = None
        self._create_figure()

    def _resolve_fs(self, row: pd.Series, metadata: dict) -> float:
        fs = metadata.get("global", {}).get("core:sample_rate")
        if fs is not None:
            return float(fs)

        if row["modality"] == "ppg":
            return float(self.dataset_config.get_ppg_fs())
        if row["modality"] == "ecg":
            return float(self.dataset_config.get_ecg_fs())

        raise ValueError(
            f"Could not infer sampling frequency for modality={row['modality']}"
        )

    def _get_complete_signal(self, row: pd.Series, waveform_id: str) -> tuple[np.ndarray, float]:
        cache_key = (str(row["file_id"]), str(row["modality"]), waveform_id)
        if cache_key in self._signal_cache:
            return self._signal_cache[cache_key]

        file_id = str(row["file_id"])
        path = self.dataset_config.get_gen_complete_path(file_id, waveform_id)
        signal, metadata = read_sigmf_file(path)
        fs = self._resolve_fs(row, metadata)

        self._signal_cache[cache_key] = (signal, fs)
        return signal, fs

    def _create_figure(self) -> None:
        self.fig, self.ax = plt.subplots(figsize=(14, 6))
        plt.subplots_adjust(bottom=0.25)

        ax_prev = plt.axes((0.20, 0.08, 0.08, 0.05))
        ax_next = plt.axes((0.30, 0.08, 0.08, 0.05))
        ax_first = plt.axes((0.08, 0.08, 0.08, 0.05))
        ax_last = plt.axes((0.42, 0.08, 0.08, 0.05))

        self.btn_first = Button(ax_first, "First", hovercolor="0.975")
        self.btn_prev = Button(ax_prev, "Previous", hovercolor="0.975")
        self.btn_next = Button(ax_next, "Next", hovercolor="0.975")
        self.btn_last = Button(ax_last, "Last", hovercolor="0.975")

        self.btn_first.on_clicked(self._on_first)
        self.btn_prev.on_clicked(self._on_prev)
        self.btn_next.on_clicked(self._on_next)
        self.btn_last.on_clicked(self._on_last)

        ax_textbox = plt.axes((0.58, 0.08, 0.20, 0.05))
        self.textbox = TextBox(
            ax_textbox,
            "Segment #:",
            initial=str(self.current_segment_idx),
            color="lightblue",
            hovercolor="0.975",
        )
        self.textbox.on_submit(self._on_textbox_submit)

        self.info_text = plt.text(
            0.68,
            0.15,
            "",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            transform=self.fig.transFigure,
        )

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._update_plot()
        plt.show()

    def _update_plot(self) -> None:
        assert self.ax is not None
        assert self.info_text is not None
        self.ax.clear()

        row = self.segments_df.iloc[self.current_segment_idx]
        file_id = str(row["file_id"])
        mod = str(row["modality"])

        start_sample = int(row["start_sample"])
        duration = int(row["duration"])
        end_sample_current = start_sample + duration

        loaded_waveforms: list[tuple[str, np.ndarray, float]] = []
        for waveform_id in self.waveform_ids:
            try:
                signal, fs = self._get_complete_signal(row, waveform_id)
                loaded_waveforms.append((waveform_id, signal, fs))
            except FileNotFoundError:
                continue

        if len(loaded_waveforms) == 0:
            raise FileNotFoundError(
                f"None of the configured waveforms were found for file_id={row['file_id']}. "
                f"Tried: {self.waveform_ids}"
            )

        base_fs = loaded_waveforms[0][2]

        # Also overlap the original raw PPG waveform.
        ppg_path = self.dataset_config.get_raw_complete_path(row["file_id"])
        ppg_signal, metadata_info = read_sigmf_file(ppg_path)
        ppg_fs = float(metadata_info["global"]["core:sample_rate"])
        # Normalize for better visualization together with SQI
        ppg_signal = scale_signal(ppg_signal)

        colors = plt.cm.get_cmap("tab10", max(3, len(loaded_waveforms)))
        plotted_ids = []
        for idx, (waveform_id, signal, fs) in enumerate(loaded_waveforms):
            time_complete = np.arange(len(signal)) / fs

            self.ax.plot(
                time_complete,
                scale_signal(signal),
                color=PIPELINE_COLORS[idx % len(PIPELINE_COLORS)],
                linewidth=1.8,
                alpha=0.85,
                marker=PIPELINE_MARKERS[idx % len(PIPELINE_MARKERS)],
                markevery=250,
                markersize=3,
                label=waveform_id,
                zorder=2,
            )
            plotted_ids.append(waveform_id)

        if len(plotted_ids) == 0:
            raise ValueError(
                f"Segment {row['segment_id']} is empty for all loaded waveforms."
            )
        # add the PPG waveform to the plot
        time_ppg = np.arange(len(ppg_signal)) / ppg_fs
        self.ax.plot(
            time_ppg,
            ppg_signal,
            color=PPG_COLOR,
            linewidth=1.5,
            alpha=0.95,
            label="Raw PPG",
            zorder=5,
        )
        # Mark all segments in the current file/modality over the complete waveform timeline.
        file_segments = self.segments_df[
            (self.segments_df["file_id"] == row["file_id"])
            & (self.segments_df["modality"] == row["modality"])
        ]
        for _, seg_row in file_segments.iterrows():
            other_start = int(seg_row["start_sample"]) / base_fs
            other_end = (int(seg_row["start_sample"]) +
                         int(seg_row["duration"])) / base_fs
            is_current = seg_row["segment_id"] == row["segment_id"]

            if is_current:
                self.ax.axvspan(
                    other_start,
                    other_end,
                    color="tab:blue",
                    alpha=0.10,
                    zorder=4,
                )
            else:
                self.ax.axvspan(
                    other_start,
                    other_end,
                    color="tab:gray",
                    alpha=0.06,
                    zorder=0,
                )

        self.ax.axvline(
            x=start_sample / base_fs,
            color="tab:blue",
            linestyle="--",
            alpha=0.6,
            linewidth=1,
        )
        self.ax.axvline(
            x=end_sample_current / base_fs,
            color="tab:blue",
            linestyle="--",
            alpha=0.6,
            linewidth=1,
        )

        q = row.get("quality_indicator", np.nan)

        self.ax.set_xlabel("Time (seconds)")
        self.ax.set_ylabel("Amplitude")
        self.ax.set_title(
            f"file_id={file_id} | modality={mod} | segment={row['segment_id']}"
        )
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc="upper right", fontsize=10)

        q_text = f"{float(q):.4f}" if pd.notna(q) else "n/a"
        info = (
            f"Segment {self.current_segment_idx + 1} of {len(self.segments_df)}\n"
            f"file_id: {file_id}\n"
            f"modality: {mod}\n"
            f"start_sample: {start_sample}\n"
            f"duration: {duration} samples ({duration / base_fs:.2f} s)\n"
            f"quality_indicator: {q_text}\n"
            f"waveforms plotted: {', '.join(plotted_ids)}\n"
            f"segments in file/modality: {len(file_segments)}\n"
            f"\nShortcuts:\n"
            f"Left/Right: Previous/Next\n"
            f"Home/End: First/Last\n"
            f"Textbox: Jump to index"
        )
        self.info_text.set_text(info)
        plt.draw()

    def _on_prev(self, _event) -> None:
        assert self.textbox is not None
        if self.current_segment_idx > 0:
            self.current_segment_idx -= 1
            self.textbox.set_val(str(self.current_segment_idx))
            self._update_plot()

    def _on_next(self, _event) -> None:
        assert self.textbox is not None
        if self.current_segment_idx < len(self.segments_df) - 1:
            self.current_segment_idx += 1
            self.textbox.set_val(str(self.current_segment_idx))
            self._update_plot()

    def _on_first(self, _event) -> None:
        assert self.textbox is not None
        self.current_segment_idx = 0
        self.textbox.set_val(str(self.current_segment_idx))
        self._update_plot()

    def _on_last(self, _event) -> None:
        assert self.textbox is not None
        self.current_segment_idx = len(self.segments_df) - 1
        self.textbox.set_val(str(self.current_segment_idx))
        self._update_plot()

    def _on_textbox_submit(self, text: str) -> None:
        assert self.textbox is not None
        try:
            segment_num = int(text.strip())
            if segment_num < 0 or segment_num >= len(self.segments_df):
                print(
                    f"Invalid segment index: {segment_num}. "
                    f"Valid range: 0-{len(self.segments_df) - 1}"
                )
                self.textbox.set_val(str(self.current_segment_idx))
                return

            self.current_segment_idx = segment_num
            self._update_plot()
        except ValueError:
            print(f"Invalid input: '{text}'. Please enter an integer.")
            self.textbox.set_val(str(self.current_segment_idx))

    def _on_key(self, event) -> None:
        if event.key == "left":
            self._on_prev(None)
        elif event.key == "right":
            self._on_next(None)
        elif event.key == "home":
            self._on_first(None)
        elif event.key == "end":
            self._on_last(None)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize segments on complete signals with interactive GUI"
    )
    parser.add_argument(
        "json_file",
        help="Path to dataset config JSON.",
    )
    parser.add_argument(
        "--file-id",
        default=None,
        help="Optional filter to only show one file_id.",
    )
    parser.add_argument(
        "--modality",
        choices=["ppg", "ecg", "bioimp"],
        default=None,
        help="Optional filter to only show one modality.",
    )
    parser.add_argument(
        "--segmenter-file",
        default=None,
        help=(
            "Optional segmenter JSON file path override. "
            "By default, uses SEGMENTER_FILE from dataset config."
        ),
    )
    parser.add_argument(
        "--segments-file",
        default=None,
        help=(
            "Optional segments CSV file name/path override. "
            "By default, uses segmenter output_file_name."
        ),
    )
    parser.add_argument(
        "--waveform-id",
        default=None,
        help=(
            "Deprecated. Kept for backward compatibility; ignored because "
            "all IDs in ALL_PIPELINES are plotted."
        ),
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Initial segment index to display.",
    )

    args = parser.parse_args()

    dataset_config = DatasetConfig(args.json_file)

    segmenter_file = args.segmenter_file
    if segmenter_file is None:
        segmenter_file = dataset_config.get_segmenter_file_name()

    segmenter = Segmenter(segmenter_file, args.json_file)

    segments_file = args.segments_file
    if segments_file is None:
        segments_file = dataset_config.get_segments_file_name()

    print(f"Dataset config : {args.json_file}")
    print(f"Segmenter file : {segmenter_file}")
    print(f"Segments file  : {segments_file}")
    print(f"Waveform IDs   : {ALL_PIPELINES}")

    segment_manager = SegmentManager(dataset_config, segments_file)

    SegmentVisualizer(
        dataset_config=dataset_config,
        segment_manager=segment_manager,
        file_id=args.file_id,
        modality=args.modality,
        start_index=args.start_index,
    )

    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
