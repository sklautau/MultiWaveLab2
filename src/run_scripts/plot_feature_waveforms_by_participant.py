"""
Plot grouped feature waveforms over segment order for each participant.

The script expects a features CSV where metadata columns are present, typically
on the right side, and feature columns are numeric predictors.

Nested-loop behavior:
- Outer loop: iterate over participant_id values.
- Inner loop: iterate over groups of N feature columns and plot them together.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


DEFAULT_METADATA_COLUMNS = {
    "segment_id",
    "modality",
    "start_sample",
    "duration",
    "quality_indicator",
    "relative_path",
    "participant_id",
    "datetime",
    "GLC",
    "session_id",
}


def _segment_sort_key(segment_id: str) -> tuple[str, int]:
    """
    Sort segment IDs using their numeric suffix when available.
    Examples: seg_id2 < seg_id10.
    """
    text = str(segment_id)
    digits = ""
    for ch in reversed(text):
        if ch.isdigit():
            digits = ch + digits
        else:
            break
    if digits:
        return text[: len(text) - len(digits)], int(digits)
    return text, -1


def _resolve_feature_columns(df: pd.DataFrame, metadata_columns: set[str]) -> List[str]:
    feature_candidates = [
        c for c in df.columns
        if c not in metadata_columns and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not feature_candidates:
        raise ValueError(
            "No numeric feature columns were found after excluding metadata columns."
        )
    return feature_candidates


def _group_features_by_identifier(feature_columns: List[str]) -> Dict[str, List[str]]:
    """
    Group features by the second underscore-separated field.

    Examples:
    - ppg_u5_m3_w10 -> group 'u5'
    - ppg_i6_IMF_1 -> group 'i6'
    """
    groups: Dict[str, List[str]] = {}
    for feature_name in feature_columns:
        parts = str(feature_name).split("_")
        if len(parts) >= 2 and parts[1]:
            group_id = parts[1]
        else:
            group_id = "ungrouped"
        groups.setdefault(group_id, []).append(feature_name)
    return groups


def _plot_feature_group_for_participant(
    participant_df: pd.DataFrame,
    participant_id: str,
    group_id: str,
    feature_group: List[str],
    output_file: Path,
    show_plot: bool,
    standardized: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))

    x = participant_df["segment_order"]
    for feature_name in feature_group:
        y = pd.to_numeric(participant_df[feature_name], errors="coerce")
        ax.plot(x, y, marker="o", linewidth=1.2,
                markersize=3, label=feature_name)

    ax.set_title(
        f"Participant {participant_id} - Group {group_id} ({len(feature_group)} features)"
    )
    ax.set_xlabel("Segment order (sorted by segment_id)")
    if standardized:
        ax.set_ylabel("Feature value (z-score)")
    else:
        ax.set_ylabel("Feature value")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8, ncol=1)

    fig.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=200)
    print(
        f"Saved plot: {output_file} (participant {participant_id}, group {group_id})")

    if show_plot:
        plt.show()

    plt.close(fig)


def _standardize_feature_columns_with_reference(
    df_to_transform: pd.DataFrame,
    df_reference: pd.DataFrame,
    feature_columns: List[str],
) -> pd.DataFrame:
    """Fit scaler on reference dataframe, transform target dataframe."""
    out = df_to_transform.copy()
    reference_matrix = df_reference[feature_columns].apply(
        pd.to_numeric, errors="coerce")
    transform_matrix = out[feature_columns].apply(
        pd.to_numeric, errors="coerce")

    scaler = StandardScaler()
    scaler.fit(reference_matrix)
    out[feature_columns] = scaler.transform(transform_matrix)
    return out


def plot_feature_waveforms(
    features_csv: Path,
    output_dir: Path,
    participant_ids: list[str] | None,
    show_plots: bool,
    standardize_features: bool,
) -> None:
    df_all = pd.read_csv(features_csv)

    required_columns = {"participant_id", "segment_id"}
    missing = required_columns - set(df_all.columns)
    if missing:
        raise ValueError(
            f"Input CSV is missing required columns: {sorted(missing)}")

    metadata_columns = {
        c for c in DEFAULT_METADATA_COLUMNS if c in df_all.columns}
    feature_columns = _resolve_feature_columns(df_all, metadata_columns)
    feature_groups = _group_features_by_identifier(feature_columns)

    df = df_all.copy()

    if participant_ids:
        df = df[df["participant_id"].astype(str).isin(participant_ids)].copy()

    if df.empty:
        raise ValueError("No rows available after participant filtering.")

    if standardize_features:
        df = _standardize_feature_columns_with_reference(
            df_to_transform=df,
            df_reference=df_all,
            feature_columns=feature_columns,
        )
        print("Feature scaling: enabled (StandardScaler z-score, fit on all participants)")
    else:
        print("Feature scaling: disabled")

    participants = sorted(
        df["participant_id"].dropna().astype(str).unique().tolist())

    print(f"Input rows: {len(df)}")
    print(f"Participants to plot: {len(participants)}")
    print(f"Numeric feature columns: {len(feature_columns)}")
    print(f"Feature identifier groups: {len(feature_groups)}")

    # Outer loop: participants
    for participant_id in participants:
        participant_df = df[df["participant_id"].astype(
            str) == participant_id].copy()

        # Sort by textual prefix + numeric suffix from segment_id (e.g., seg_id2 < seg_id10).
        segment_text = participant_df["segment_id"].astype(str)
        participant_df["_segment_prefix"] = segment_text.str.replace(
            r"\d+$", "", regex=True
        )
        participant_df["_segment_numeric_suffix"] = pd.to_numeric(
            segment_text.str.extract(r"(\d+)$", expand=False),
            errors="coerce",
        ).fillna(np.inf)

        order = np.lexsort(
            (
                participant_df["segment_id"].astype(str).to_numpy(),
                participant_df["_segment_numeric_suffix"].to_numpy(),
                participant_df["_segment_prefix"].astype(str).to_numpy(),
            )
        )
        participant_df = participant_df.iloc[order].reset_index(drop=True)
        participant_df["segment_order"] = range(len(participant_df))
        participant_df = participant_df.drop(
            columns=["_segment_prefix", "_segment_numeric_suffix"]
        )

        # Inner loop: groups based on identifier (second underscore-separated token)
        for group_id in sorted(feature_groups.keys()):
            feature_group = feature_groups[group_id]
            file_name = f"participant_{participant_id}_group_{group_id}.pdf"
            output_file = output_dir / file_name
            _plot_feature_group_for_participant(
                participant_df=participant_df,
                participant_id=participant_id,
                group_id=group_id,
                feature_group=feature_group,
                output_file=output_file,
                show_plot=show_plots,
                standardized=standardize_features,
            )

    print(f"Plots saved under: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read a features CSV and plot grouped feature waveforms over segment order "
            "for each participant."
        )
    )
    parser.add_argument(
        "features_csv",
        help="Path to features CSV file (e.g., features7_train.csv).",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=None,
        help=(
            "Deprecated. Grouping now uses the feature group identifier "
            "(second underscore-separated token)."
        ),
    )
    parser.add_argument(
        "--participant-id",
        action="append",
        default=None,
        help=(
            "Participant ID to include. Repeat this option to include multiple IDs. "
            "If omitted, all participants are plotted."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output folder for plot images. "
            "Default: <features_csv_parent>/<features_csv_stem>_participant_feature_waveforms"
        ),
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display plots interactively in addition to saving files.",
    )
    parser.add_argument(
        "--standardize-features",
        action="store_true",
        help=(
            "Standardize each numeric feature with StandardScaler (mean=0, std=1) "
            "before plotting."
        ),
    )

    args = parser.parse_args()

    if args.group_size is not None:
        print(
            "Warning: --group-size is deprecated and ignored. Grouping is identifier-based.")

    features_csv = Path(args.features_csv)
    if not features_csv.exists():
        raise FileNotFoundError(f"Features CSV not found: {features_csv}")

    if args.output_dir is None:
        output_dir = features_csv.parent / (
            f"{features_csv.stem}_participant_feature_waveforms"
        )
    else:
        output_dir = Path(args.output_dir)

    plot_feature_waveforms(
        features_csv=features_csv,
        output_dir=output_dir,
        participant_ids=args.participant_id,
        show_plots=args.show_plots,
        standardize_features=args.standardize_features,
    )


if __name__ == "__main__":
    main()
