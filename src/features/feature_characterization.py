"""
Feature characterization focused on per-feature behavior versus glucose.

This script intentionally does NOT perform feature selection.
It reads the same train input used by features/feature_selection.py:
datasetConfig.get_splitted_features_file_name("train").
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
from scipy.stats import kurtosis, pearsonr, skew, spearmanr
from sklearn.feature_selection import mutual_info_regression

from datasets_util.naming_conventions import DatasetConfig


TARGET_COLUMN = "GLC"


def _detect_subject_column(df: pd.DataFrame) -> str:
    if "participant_id" in df.columns:
        return "participant_id"
    if "SUBJECT_ID" in df.columns:
        return "SUBJECT_ID"
    raise ValueError(
        "Could not detect subject column. Expected 'participant_id' or 'SUBJECT_ID'."
    )


def _compute_variance_decomposition(
        features_df: pd.DataFrame,
        feature_column: str,
        id_column: str,
) -> dict:
    df = features_df[[id_column, feature_column]].dropna()
    n = len(df)
    groups = df.groupby(id_column)
    k = len(groups)

    if k < 2 or n <= k:
        return {
            "n_subjects": int(k),
            "n_observations": int(n),
            "between_variance": np.nan,
            "within_variance": np.nan,
            "total_variance": np.nan,
            "ICC": np.nan,
            "within_subject_variance_pct": np.nan,
        }

    overall_mean = df[feature_column].mean()
    counts = groups.size()
    mean_count = counts.mean()
    means = groups[feature_column].mean()

    ss_between = np.sum(counts * (means - overall_mean) ** 2)
    ss_within = groups.apply(
        lambda g: np.sum((g[feature_column] - g[feature_column].mean()) ** 2),
        include_groups=False,
    ).sum()

    df_between = k - 1
    df_within = n - k
    if df_between <= 0 or df_within <= 0:
        return {
            "n_subjects": int(k),
            "n_observations": int(n),
            "between_variance": np.nan,
            "within_variance": np.nan,
            "total_variance": np.nan,
            "ICC": np.nan,
            "within_subject_variance_pct": np.nan,
        }

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within

    between_variance = max((ms_between - ms_within) / mean_count, 0.0)
    within_variance = ms_within
    total_variance = between_variance + within_variance

    if total_variance <= 0:
        icc = np.nan
        within_subject_variance_pct = np.nan
    else:
        icc = between_variance / total_variance
        within_subject_variance_pct = 100.0 * within_variance / total_variance

    return {
        "n_subjects": int(k),
        "n_observations": int(n),
        "between_variance": float(between_variance),
        "within_variance": float(within_variance),
        "total_variance": float(total_variance),
        "ICC": float(icc),
        "within_subject_variance_pct": float(within_subject_variance_pct),
    }


def _centered_pearson(
        df: pd.DataFrame,
        feature_col: str,
        target_col: str,
        subject_col: str,
) -> tuple[float, float]:
    tmp = df[[subject_col, feature_col, target_col]].dropna().copy()
    if tmp.empty:
        return np.nan, np.nan

    tmp["x_centered"] = tmp[feature_col] - \
        tmp.groupby(subject_col)[feature_col].transform("mean")
    tmp["y_centered"] = tmp[target_col] - \
        tmp.groupby(subject_col)[target_col].transform("mean")

    # Need non-degenerate vectors after centering.
    if np.isclose(tmp["x_centered"].std(ddof=0), 0.0) or np.isclose(tmp["y_centered"].std(ddof=0), 0.0):
        return np.nan, np.nan

    r, p = pearsonr(tmp["x_centered"], tmp["y_centered"])
    return float(r), float(p)


def _rm_corr(
        df: pd.DataFrame,
        feature_col: str,
        target_col: str,
        subject_col: str,
) -> tuple[float, float]:
    tmp = df[[subject_col, feature_col, target_col]].dropna().copy()
    if tmp[subject_col].nunique() < 3:
        return np.nan, np.nan

    try:
        out = pg.rm_corr(data=tmp, x=feature_col,
                         y=target_col, subject=subject_col)
        if out.empty:
            return np.nan, np.nan
        return float(out.loc[0, "r"]), float(out.loc[0, "pval"])
    except Exception:
        return np.nan, np.nan


def _safe_mutual_info(feature: pd.Series, target: pd.Series, random_state: int) -> float:
    tmp = pd.concat([feature, target], axis=1).dropna()
    if len(tmp) < 5:
        return np.nan

    x = tmp.iloc[:, 0].to_numpy(dtype=float).reshape(-1, 1)
    y = tmp.iloc[:, 1].to_numpy(dtype=float)

    if np.isclose(np.std(x), 0.0):
        return 0.0

    mi = mutual_info_regression(x, y, random_state=random_state)
    return float(mi[0])


def _auto_interpretation(row: pd.Series) -> str:
    missing_pct = row["missing_pct"]
    n_unique = row["n_unique"]
    icc = row["icc"]
    within_pct = row["within_subject_variance_pct"]
    rm_r = abs(row["rm_corr_r"]) if pd.notna(row["rm_corr_r"]) else 0.0
    centered_r = abs(row["centered_pearson_r"]) if pd.notna(
        row["centered_pearson_r"]) else 0.0
    pearson_r = abs(row["pearson_r"]) if pd.notna(row["pearson_r"]) else 0.0

    if missing_pct >= 40.0:
        return "High missingness"
    if n_unique <= 3:
        return "Low-resolution / near-constant feature"
    if pd.notna(icc) and icc >= 0.75 and rm_r < 0.2 and centered_r < 0.2:
        return "Mostly subject-specific"
    if rm_r >= 0.3 and pd.notna(within_pct) and within_pct >= 50.0:
        return "Dynamic biomarker"
    if rm_r >= 0.3 and pd.notna(icc) and icc >= 0.5:
        return "Dynamic with strong subject baseline"
    if pearson_r >= 0.3 and centered_r < 0.15 and pd.notna(icc) and icc >= 0.6:
        return "Mostly between-subject association"
    return "Mixed behavior"


def _add_composite_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    mi = out["mutual_information_glc"].copy()
    mi_min = mi.min(skipna=True)
    mi_max = mi.max(skipna=True)
    if pd.isna(mi_min) or pd.isna(mi_max) or np.isclose(mi_max - mi_min, 0.0):
        out["mi_norm"] = 0.0
    else:
        out["mi_norm"] = (mi - mi_min) / (mi_max - mi_min)

    for col in ["pearson_r", "spearman_rho", "rm_corr_r", "centered_pearson_r"]:
        out[f"abs_{col}"] = out[col].abs()

    out["composite_score"] = (
        0.30 * out["abs_rm_corr_r"].fillna(0.0)
        + 0.25 * out["abs_centered_pearson_r"].fillna(0.0)
        + 0.20 * out["abs_pearson_r"].fillna(0.0)
        + 0.15 * out["abs_spearman_rho"].fillna(0.0)
        + 0.10 * out["mi_norm"].fillna(0.0)
    )
    return out


def _save_scatter(
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        output_file: Path,
        title: str,
        xlabel: str,
        ylabel: str,
        annotate_top_n: int = 12,
) -> None:
    plot_df = df[["feature", x_col, y_col, "composite_score"]].dropna().copy()
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8.8, 6.5))
    scatter = ax.scatter(
        plot_df[x_col],
        plot_df[y_col],
        c=plot_df["composite_score"],
        cmap="viridis",
        alpha=0.85,
        s=48,
        edgecolors="white",
        linewidths=0.4,
    )
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Composite score")

    top_df = plot_df.sort_values(
        "composite_score", ascending=False).head(annotate_top_n)
    for _, row in top_df.iterrows():
        ax.annotate(
            row["feature"],
            (row[x_col], row[y_col]),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=8,
            alpha=0.85,
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300)
    plt.close(fig)


def _save_metrics_heatmap(df: pd.DataFrame, output_file: Path) -> None:
    heatmap_cols = [
        "pearson_r",
        "spearman_rho",
        "mutual_information_glc",
        "icc",
        "within_subject_variance_pct",
        "rm_corr_r",
        "centered_pearson_r",
        "mean",
        "std",
        "cv",
        "skewness",
        "kurtosis",
        "missing_pct",
        "n_unique",
        "composite_score",
    ]
    plot_df = df[["feature"] + heatmap_cols].set_index("feature")

    # Robust z-score scaling per metric for visualization comparability.
    scaled = plot_df.copy()
    for col in scaled.columns:
        series = scaled[col]
        std = series.std(skipna=True)
        if pd.isna(std) or np.isclose(std, 0.0):
            scaled[col] = 0.0
        else:
            scaled[col] = (series - series.mean(skipna=True)) / std

    # Keep figure readable with many features.
    max_rows = min(len(scaled), 200)
    scaled = scaled.sort_values(
        "composite_score", ascending=False).head(max_rows)

    fig_height = max(7.0, 0.18 * len(scaled) + 2.5)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    sns.heatmap(
        scaled,
        cmap="coolwarm",
        center=0,
        linewidths=0.2,
        linecolor="white",
        cbar_kws={"label": "z-score"},
        ax=ax,
    )
    ax.set_title("Feature characterization heatmap (top by composite score)")
    ax.set_xlabel("Metrics")
    ax.set_ylabel("Features")
    fig.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300)
    plt.close(fig)


def characterize_features(dataset_config: DatasetConfig, random_state: int = 42) -> pd.DataFrame:
    statistics_output_folder = Path(
        dataset_config.get_statistics_output_folder())
    statistics_output_folder.mkdir(parents=True, exist_ok=True)

    metadata_columns = dataset_config.get_chosen_metadata_columns()
    if TARGET_COLUMN in metadata_columns:
        metadata_columns.remove(TARGET_COLUMN)

    input_file_name = dataset_config.get_splitted_features_file_name("train")
    print("Opening", input_file_name)
    df = pd.read_csv(input_file_name)

    if "participant_id" not in df.columns and "SUBJECT_ID" not in df.columns:
        df = df.rename(columns={"ID": "SUBJECT_ID"})

    subject_col = _detect_subject_column(df)

    feature_cols = [
        c for c in df.columns
        if c not in (metadata_columns + [TARGET_COLUMN])
    ]

    print(f"Detected {len(feature_cols)} feature columns.")
    print(f"Subject column used: {subject_col}")

    rows = []
    for feature in feature_cols:
        raw_series = df[feature]
        numeric_series = pd.to_numeric(raw_series, errors="coerce")

        feature_target = pd.concat(
            [numeric_series, df[TARGET_COLUMN]], axis=1).dropna()

        if len(feature_target) >= 3:
            try:
                pearson_r, pearson_p = pearsonr(
                    feature_target.iloc[:, 0], feature_target.iloc[:, 1])
            except Exception:
                pearson_r, pearson_p = np.nan, np.nan

            try:
                spearman_rho, spearman_p = spearmanr(
                    feature_target.iloc[:, 0], feature_target.iloc[:, 1])
                spearman_rho = float(spearman_rho)
                spearman_p = float(spearman_p)
            except Exception:
                spearman_rho, spearman_p = np.nan, np.nan
        else:
            pearson_r, pearson_p = np.nan, np.nan
            spearman_rho, spearman_p = np.nan, np.nan

        mi = _safe_mutual_info(
            numeric_series, df[TARGET_COLUMN], random_state=random_state)
        rm_r, rm_p = _rm_corr(df.assign(_feat=numeric_series),
                              "_feat", TARGET_COLUMN, subject_col)
        centered_r, centered_p = _centered_pearson(
            df.assign(_feat=numeric_series), "_feat", TARGET_COLUMN, subject_col
        )

        variance_info = _compute_variance_decomposition(
            df.assign(_feat=numeric_series),
            feature_column="_feat",
            id_column=subject_col,
        )

        missing_pct = float(numeric_series.isna().mean() * 100.0)
        n_unique = int(raw_series.nunique(dropna=True))
        mean_val = float(numeric_series.mean(skipna=True)
                         ) if numeric_series.notna().any() else np.nan
        std_val = float(numeric_series.std(skipna=True, ddof=1)
                        ) if numeric_series.notna().sum() > 1 else np.nan
        if pd.notna(mean_val) and not np.isclose(mean_val, 0.0) and pd.notna(std_val):
            cv_val = float(std_val / abs(mean_val))
        else:
            cv_val = np.nan

        skew_val = float(skew(numeric_series.dropna(), bias=False)
                         ) if numeric_series.notna().sum() > 2 else np.nan
        kurt_val = float(kurtosis(numeric_series.dropna(
        ), fisher=True, bias=False)) if numeric_series.notna().sum() > 3 else np.nan

        rows.append(
            {
                "feature": feature,
                "pearson_r": float(pearson_r) if pd.notna(pearson_r) else np.nan,
                "pearson_pvalue": float(pearson_p) if pd.notna(pearson_p) else np.nan,
                "spearman_rho": float(spearman_rho) if pd.notna(spearman_rho) else np.nan,
                "spearman_pvalue": float(spearman_p) if pd.notna(spearman_p) else np.nan,
                "mutual_information_glc": mi,
                "icc": variance_info["ICC"],
                "within_subject_variance_pct": variance_info["within_subject_variance_pct"],
                "rm_corr_r": rm_r,
                "rm_corr_pvalue": rm_p,
                "centered_pearson_r": centered_r,
                "centered_pearson_pvalue": centered_p,
                "mean": mean_val,
                "std": std_val,
                "cv": cv_val,
                "skewness": skew_val,
                "kurtosis": kurt_val,
                "missing_pct": missing_pct,
                "n_unique": n_unique,
                "n_subjects": variance_info["n_subjects"],
                "n_observations": variance_info["n_observations"],
                "between_variance": variance_info["between_variance"],
                "within_variance": variance_info["within_variance"],
                "total_variance": variance_info["total_variance"],
            }
        )

    metrics_df = pd.DataFrame(rows)
    metrics_df = _add_composite_score(metrics_df)
    metrics_df["interpretation"] = metrics_df.apply(
        _auto_interpretation, axis=1)
    metrics_df = metrics_df.sort_values(
        "composite_score", ascending=False).reset_index(drop=True)

    csv_file = statistics_output_folder / "feature_characterization_table.csv"
    # xlsx_file = statistics_output_folder / "feature_characterization_table.xlsx"
    metrics_df.to_csv(csv_file, index=False)
    # metrics_df.to_excel(xlsx_file, index=False)

    _save_scatter(
        metrics_df,
        x_col="mutual_information_glc",
        y_col="icc",
        output_file=statistics_output_folder / "feature_characterization_mi_vs_icc.png",
        title="Mutual Information vs ICC",
        xlabel="Mutual information with GLC",
        ylabel="ICC",
    )
    _save_scatter(
        metrics_df,
        x_col="rm_corr_r",
        y_col="icc",
        output_file=statistics_output_folder /
        "feature_characterization_rmcorr_vs_icc.png",
        title="Repeated-Measures Correlation vs ICC",
        xlabel="Repeated-measures correlation (r)",
        ylabel="ICC",
    )
    _save_scatter(
        metrics_df,
        x_col="pearson_r",
        y_col="rm_corr_r",
        output_file=statistics_output_folder /
        "feature_characterization_pearson_vs_rmcorr.png",
        title="Pearson vs Repeated-Measures Correlation",
        xlabel="Pearson correlation with GLC",
        ylabel="Repeated-measures correlation (r)",
    )
    _save_metrics_heatmap(
        metrics_df,
        output_file=statistics_output_folder /
        "feature_characterization_metrics_heatmap.png",
    )

    print("========================================")
    print("Feature characterization completed")
    print("========================================")
    print(f"Input file           : {input_file_name}")
    print(f"Output folder        : {statistics_output_folder}")
    print(f"Table CSV            : {csv_file}")
    # print(f"Table Excel          : {xlsx_file}")
    print(
        "Figures              : "
        f"{statistics_output_folder / 'feature_characterization_mi_vs_icc.png'}, "
        f"{statistics_output_folder / 'feature_characterization_rmcorr_vs_icc.png'}, "
        f"{statistics_output_folder / 'feature_characterization_pearson_vs_rmcorr.png'}, "
        f"{statistics_output_folder / 'feature_characterization_metrics_heatmap.png'}"
    )

    return metrics_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Characterize all features against glucose without feature selection."
    )
    parser.add_argument(
        "json_file",
        metavar="JSON_FILE",
        help="Path to the input JSON config file.",
    )
    args = parser.parse_args()

    sns.set_theme(style="whitegrid", context="paper")
    dataset_config = DatasetConfig(args.json_file)

    input_file_name = dataset_config.get_splitted_features_file_name("train")
    print("Opening", input_file_name)
    df = pd.read_csv(input_file_name)

    calculate_num_unique_values(df)
    exit(-1)

    characterize_features(dataset_config)


def calculate_num_unique_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the number of unique values for each feature in the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing features.

    Returns:
    pd.DataFrame: A DataFrame with two columns: 'feature' and 'n_unique',
                  where 'feature' is the feature name and 'n_unique' is the
                  number of unique values for that feature.
    """
    unique_counts = df.nunique(dropna=True)
    result_df = pd.DataFrame({
        'feature': unique_counts.index,
        'n_unique': unique_counts.values,
        'n_bits': np.log2(unique_counts.values)
    })

    file_name = 'feature_unique_counts.csv'
    result_df.to_csv(file_name, index=False)
    print("Saved unique counts to ", file_name)

    return result_df


if __name__ == "__main__":
    # main()
    # create a toy Dataframe that allows to test method _compute_variance_decomposition()
    df = pd.DataFrame({
        'participant_id': [1, 1, 1, 1, 2, 2, 2, 2],
        'x1': [1, 1, 1, 1, 500, 500, 500, 500],
        'x2': [500, 1, 1, 500, 1, 500, 500, 1]
    })
    # example with ICC = 1.0 (all values for each subject are the same)
    result_df = _compute_variance_decomposition(
        df, 'x1', 'participant_id')
    print(result_df)

    # example with ICC = 0.0 (all values for each subject are different)
    result_df = _compute_variance_decomposition(
        df, 'x2', 'participant_id')
    print(result_df)
