"""
Compute SQI statistics/histograms over all segments for all SQI pipelines.

The script:
1) loads DatasetConfig from JSON,
2) resolves segments via Segmenter/SegmentManager,
3) iterates all segment rows for a chosen modality,
4) aggregates SQI samples inside segments for each waveform in ALL_PIPELINES,
5) writes per-pipeline statistics CSV and one histogram PDF per pipeline.
"""

from datasets_util.waveform_files import read_sigmf_file
from segments.segments_core import SegmentManager, Segmenter
from datasets_util.naming_conventions import DatasetConfig
import argparse
from pathlib import Path
import os
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure project root is in path for imports when running this script directly.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _resolve_default_pipeline_for_modality(dataset_config: DatasetConfig, modality: str) -> str:
    if modality == "ppg":
        return str(dataset_config.get_value("PPG_SQI_OUTPUT_WAVEFORM", "sqi_sumall"))
    if modality == "ecg":
        return str(dataset_config.get_value("ECG_SQI_OUTPUT_WAVEFORM", "sqi_sumall"))
    if modality.startswith("bioimp"):
        return str(dataset_config.get_value("BIOIMP_SQI_OUTPUT_WAVEFORM", "sqi_sumall"))
    raise ValueError(f"Unsupported modality: {modality}")


def _resolve_pipelines(dataset_config: DatasetConfig, modality: str, pipelines_arg: str | None) -> List[str]:
    """Resolve which SQI waveform folders to evaluate.

    Defaults to the single pipeline configured in the dataset JSON for the selected modality.
    """
    if pipelines_arg is None or pipelines_arg.strip() == "":
        return [_resolve_default_pipeline_for_modality(dataset_config, modality)]

    normalized = pipelines_arg.strip().lower()
    if normalized == "configured":
        return [_resolve_default_pipeline_for_modality(dataset_config, modality)]

    if normalized == "all":
        if modality == "ppg":
            from signal_processing.ppg import QUALITY_PIPELINES as PPG_QUALITY_PIPELINES
            return list(PPG_QUALITY_PIPELINES.keys())
        if modality == "ecg":
            from signal_processing.ecg import QUALITY_PIPELINES as ECG_QUALITY_PIPELINES
            return list(ECG_QUALITY_PIPELINES.keys())
        if modality.startswith("bioimp"):
            from signal_processing.bioimpedance import QUALITY_PIPELINES as BIOIMP_QUALITY_PIPELINES
            return list(BIOIMP_QUALITY_PIPELINES.keys())

    pipelines = [p.strip() for p in pipelines_arg.split(",") if p.strip()]
    if not pipelines:
        raise ValueError(
            "No valid pipelines were provided. Use --pipelines configured, --pipelines all, or a comma-separated list."
        )
    return pipelines


def _build_segment_manager(
        dataset_config: DatasetConfig,
        segmenter_file: str | None,
        segments_file: str | None,
) -> SegmentManager:
    if segmenter_file is None:
        segmenter_file = dataset_config.get_segmenter_file_name()

    # extract file, without folders or extension:
    file_name = Path(dataset_config.config_path).stem

    # segmenter = Segmenter(segmenter_file, file_name)
    # if segments_file is None:
    #    segments_file = segmenter.get_segments_file_name()
    segments_file = dataset_config.get_segments_file_name()

    return SegmentManager(dataset_config, segments_file)


def _compute_stats(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {
            "n_samples": 0,
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "min": np.nan,
            "max": np.nan,
            "p01": np.nan,
            "p05": np.nan,
            "p10": np.nan,
            "p25": np.nan,
            "p75": np.nan,
            "p90": np.nan,
            "p95": np.nan,
            "p99": np.nan,
            "frac_ge_0_5": np.nan,
            "frac_ge_0_8": np.nan,
        }

    return {
        "n_samples": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p01": float(np.percentile(values, 1)),
        "p05": float(np.percentile(values, 5)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "frac_ge_0_5": float(np.mean(values >= 0.5)),
        "frac_ge_0_8": float(np.mean(values >= 0.8)),
    }


def _gather_values_by_pipeline(
        dataset_config: DatasetConfig,
        segments_df: pd.DataFrame,
        modality: str,
    pipelines: List[str],
) -> tuple[Dict[str, np.ndarray], Dict[str, int], int]:
    required_cols = {"file_id", "modality", "start_sample", "duration"}
    missing = sorted(required_cols - set(segments_df.columns))
    if missing:
        raise ValueError(f"Missing required segment columns: {missing}")

    segments_df = segments_df[segments_df["modality"] == modality].copy()
    if segments_df.empty:
        raise ValueError(f"No segments found for modality='{modality}'.")

    signal_cache: Dict[tuple[str, str], np.ndarray] = {}
    values_by_pipeline: Dict[str, List[np.ndarray]] = {
        pipeline: [] for pipeline in pipelines
    }
    missing_files_per_pipeline: Dict[str, int] = {
        pipeline: 0 for pipeline in pipelines
    }

    for _, row in segments_df.iterrows():
        file_id = str(row["file_id"])
        start_sample = int(row["start_sample"])
        duration = int(row["duration"])
        if duration <= 0:
            continue

        for pipeline in pipelines:
            cache_key = (file_id, pipeline)

            if cache_key not in signal_cache:
                waveform_path = dataset_config.get_gen_complete_path(
                    file_id, pipeline)
                try:
                    if modality == "ppg" or modality == "ecg":
                        signal, _ = read_sigmf_file(waveform_path)
                    else:  # bioimpedance
                        signal = pd.read_csv(waveform_path)["quality"].values
                    signal_cache[cache_key] = np.asarray(signal, dtype=float)
                except FileNotFoundError:
                    missing_files_per_pipeline[pipeline] += 1
                    continue

            signal = signal_cache[cache_key]
            if start_sample < 0 or start_sample >= len(signal):
                continue

            end_sample = min(start_sample + duration, len(signal))
            segment_values = signal[start_sample:end_sample]
            if segment_values.size == 0:
                continue

            values_by_pipeline[pipeline].append(segment_values)

    concatenated = {
        pipeline: np.concatenate(chunks) if len(
            chunks) > 0 else np.array([], dtype=float)
        for pipeline, chunks in values_by_pipeline.items()
    }

    return concatenated, missing_files_per_pipeline, len(segments_df)


def _save_histogram(
        values: np.ndarray,
        pipeline: str,
        bins: int,
        out_file: Path,
    should_show: bool = False,
) -> None:
    plt.figure(figsize=(10, 6))
    if values.size > 0:
        plt.hist(values, bins=bins, alpha=0.8, color="tab:blue")
    else:
        plt.hist([], bins=bins, alpha=0.8, color="tab:blue")
    plt.xlabel("SQI value")
    plt.ylabel("Frequency")
    plt.title(f"SQI histogram: {pipeline}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=150)
    if should_show:
        plt.show()
    plt.close()


def _save_cumulative_pdf(
        values: np.ndarray,
        pipeline: str,
        bins: int,
        out_file: Path,
        should_show: bool = False,
) -> None:
    plt.figure(figsize=(10, 6))
    if values.size > 0:
        hist, bin_edges = np.histogram(values, bins=bins, density=True)
        bin_widths = np.diff(bin_edges)
        cdf = np.cumsum(hist * bin_widths)
        cdf = np.clip(cdf, 0.0, 1.0)
        plt.plot(bin_edges[1:], cdf, color="tab:green", linewidth=2.0)
    else:
        plt.plot([], [], color="tab:green", linewidth=2.0)

    plt.xlabel("SQI value")
    plt.ylabel("Cumulative probability")
    plt.title(f"SQI cumulative PDF (CDF): {pipeline}")
    plt.ylim(0.0, 1.02)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=150)
    if should_show:
        plt.show()
    plt.close()


def _save_superimposed_histogram(
        values_by_pipeline: Dict[str, np.ndarray],
    pipelines: List[str],
        bins: int,
        out_file: Path,
        should_show: bool = False,
) -> None:
    non_empty = [v for v in values_by_pipeline.values() if v.size > 0]
    if len(non_empty) == 0:
        return

    global_min = min(float(np.min(v)) for v in non_empty)
    global_max = max(float(np.max(v)) for v in non_empty)
    if np.isclose(global_min, global_max):
        global_max = global_min + 1e-6

    bin_edges = np.linspace(global_min, global_max, bins + 1)
    bin_edges_list = bin_edges.tolist()

    plt.figure(figsize=(12, 7))
    colors = plt.cm.get_cmap("tab10", max(3, len(pipelines)))

    for idx, pipeline in enumerate(pipelines):
        values = values_by_pipeline[pipeline]
        if values.size == 0:
            continue

        plt.hist(
            values,
            bins=bin_edges_list,
            histtype="step",
            linewidth=1.8,
            alpha=0.95,
            color=colors(idx),
            label=pipeline,
        )

    plt.xlabel("SQI value")
    plt.ylabel("Frequency")
    plt.title("SQI histograms (all pipelines, shared x-axis)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=180)
    if should_show:
        plt.show()
    plt.close()


def _save_superimposed_cumulative_pdf(
        values_by_pipeline: Dict[str, np.ndarray],
    pipelines: List[str],
        bins: int,
        out_file: Path,
        should_show: bool = False,
) -> None:
    non_empty = [v for v in values_by_pipeline.values() if v.size > 0]
    if len(non_empty) == 0:
        return

    marker_styles = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "*"]

    global_min = min(float(np.min(v)) for v in non_empty)
    global_max = max(float(np.max(v)) for v in non_empty)
    if np.isclose(global_min, global_max):
        global_max = global_min + 1e-6

    bin_edges = np.linspace(global_min, global_max, bins + 1)
    bin_edges_list = bin_edges.tolist()

    plt.figure(figsize=(12, 7))
    colors = plt.cm.get_cmap("tab10", max(3, len(pipelines)))

    for idx, pipeline in enumerate(pipelines):
        values = values_by_pipeline[pipeline]
        if values.size == 0:
            continue

        hist, _ = np.histogram(values, bins=bin_edges_list, density=True)
        bin_widths = np.diff(bin_edges)
        cdf = np.cumsum(hist * bin_widths)
        cdf = np.clip(cdf, 0.0, 1.0)
        marker_count = min(5, cdf.size)
        markevery = None
        if marker_count > 1:
            markevery = np.unique(
                np.linspace(0, cdf.size - 1, marker_count, dtype=int)
            ).tolist()

        plt.plot(
            bin_edges[1:],
            cdf,
            linewidth=2.0,
            alpha=0.95,
            color=colors(idx),
            marker=marker_styles[idx % len(marker_styles)],
            markersize=5,
            markevery=markevery,
            label=pipeline,
        )

    plt.xlabel("SQI value")
    plt.ylabel("Cumulative probability")
    plt.title("SQI cumulative PDF (CDF) (all pipelines, shared x-axis)")
    plt.ylim(0.0, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=180)
    if should_show:
        plt.show()
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute SQI histograms/statistics over all segments for all pipelines."
    )
    # make it mandatory to provide the dataset config JSON file path
    parser.add_argument(
        "json_file",
        help="Path to dataset config JSON.",
    )
    parser.add_argument(
        "--segmenter-file",
        default=None,
        help="Optional segmenter JSON path override.",
    )
    parser.add_argument(
        "--segments-file",
        default=None,
        help="Optional segments CSV name/path override.",
    )
    parser.add_argument(
        "--modality",
        default="ppg",
        choices=["ppg", "ecg", "bioimp"],
        help="Modality used to filter segment rows.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=120,
        help="Number of bins for each histogram.",
    )
    parser.add_argument(
        "--output-prefix",
        default="sqi_statistics",
        help="Prefix used for output file names.",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Show histograms interactively in addition to saving PDFs.",
    )
    parser.add_argument(
        "--pipelines",
        default="configured",
        help=(
            "Pipelines to analyze: 'configured' (default), 'all', "
            "or comma-separated names like 'sqi_sumall,sqi_neurokit'."
        ),
    )
    args = parser.parse_args()

    dataset_config = DatasetConfig(args.json_file)
    pipelines = _resolve_pipelines(
        dataset_config, args.modality, args.pipelines)
    segment_manager = _build_segment_manager(
        dataset_config=dataset_config,
        segmenter_file=args.segmenter_file,
        segments_file=args.segments_file,
    )
    segments_df = segment_manager.get_segments_dataframe()

    values_by_pipeline, missing_files_per_pipeline, n_segments = _gather_values_by_pipeline(
        dataset_config=dataset_config,
        segments_df=segments_df,
        modality=args.modality,
        pipelines=pipelines,
    )

    output_dir = Path(
        dataset_config.get_dataset_machine_learning_path()) / "sqi_statistics"
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_rows = []
    for pipeline in pipelines:
        values = values_by_pipeline[pipeline]
        stats = _compute_stats(values)
        stats_rows.append(
            {
                "pipeline": pipeline,
                "modality": args.modality,
                "n_segments_considered": n_segments,
                "missing_waveform_files": missing_files_per_pipeline[pipeline],
                **stats,
            }
        )

        hist_file = output_dir / f"{args.output_prefix}_{pipeline}_hist.pdf"
        _save_histogram(
            values=values,
            pipeline=pipeline,
            bins=args.bins,
            out_file=hist_file,
            should_show=args.show_plots,
        )

        cdf_file = output_dir / f"{args.output_prefix}_{pipeline}_cum_pdf.pdf"
        _save_cumulative_pdf(
            values=values,
            pipeline=pipeline,
            bins=args.bins,
            out_file=cdf_file,
            should_show=args.show_plots,
        )

    superimposed_file = output_dir / \
        f"{args.output_prefix}_all_pipelines_superimposed_hist.pdf"
    _save_superimposed_histogram(
        values_by_pipeline=values_by_pipeline,
        pipelines=pipelines,
        bins=args.bins,
        out_file=superimposed_file,
        should_show=args.show_plots,
    )

    superimposed_cdf_file = output_dir / \
        f"{args.output_prefix}_all_pipelines_superimposed_cum_pdf.pdf"
    _save_superimposed_cumulative_pdf(
        values_by_pipeline=values_by_pipeline,
        pipelines=pipelines,
        bins=args.bins,
        out_file=superimposed_cdf_file,
        should_show=args.show_plots,
    )

    stats_df = pd.DataFrame(stats_rows)
    stats_file = output_dir / f"{args.output_prefix}_summary.csv"
    stats_df.to_csv(stats_file, index=False)

    print("========================================")
    print("SQI statistics completed")
    print("========================================")
    print(f"Config file         : {args.json_file}")
    print(f"Modality            : {args.modality}")
    print(f"Pipelines           : {pipelines}")
    print(f"Segments considered : {n_segments}")
    print(f"Output folder       : {output_dir}")
    print(f"Summary CSV         : {stats_file}")
    print(f"Superimposed hist   : {superimposed_file}")
    print(f"Superimposed cumPDF : {superimposed_cdf_file}")


if __name__ == "__main__":
    main()
