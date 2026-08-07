'''
Methods to output Latex
'''
import re
from pathlib import Path
import tempfile
import os

import pandas as pd

from pathlib import Path
import os
import pandas as pd
import numbers

from pandas.api.types import is_numeric_dtype


def get_integer_valued_columns(df: pd.DataFrame) -> list[str]:
    """
    Return columns whose non-missing values are all integers,
    even if the dtype is float.
    """
    integer_columns = []

    for col in df.columns:
        s = df[col].dropna()

        if not is_numeric_dtype(s):
            continue

        if (s % 1 == 0).all():
            integer_columns.append(col)

    return integer_columns


def format_number(v):
    if abs(v) < 0.01 and v != 0:
        return f"{v:.2e}"   # e.g. 3.45e-04
    else:
        return f"{v:.2f}"   # e.g. 12.35


def to_latex_table(
    df: pd.DataFrame,
    input_file_name: str,
    caption: str = "",
) -> str:
    """
    Converts a CSV file into a generic LaTeX table.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to convert to LaTeX.
    caption : str, optional
        Caption of the LaTeX table.
    """

    ncols = len(df.columns)

    integer_columns = get_integer_valued_columns(df)

    lines = []

    lines.append(r"\begin{table}[htb]")
    lines.append(r"\centering")

    if caption:
        lines.append(
            rf"\caption{{{_latex_escape(caption).rstrip('.')}.}}"
        )

    lines.append(
        rf"\label{{tab:{Path(input_file_name).stem}}}"
    )

    # Left-align every column. Change to "c" * ncols if preferred.
    lines.append(r"\begin{tabular}{" + "l" * ncols + "}")
    lines.append(r"\toprule")

    # Header
    header = [
        _latex_escape(str(col))
        for col in df.columns
    ]
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")

    # Data
    for _, row in df.iterrows():
        values = []
        # check if the column is integer-valued and format accordingly
        for i, v in enumerate(row):
            col_name = df.columns[i]
            if pd.isna(v):
                values.append("")
            elif col_name in integer_columns:
                values.append(str(int(v)))
            elif isinstance(v, numbers.Number):
                values.append(format_number(v))
            else:
                values.append(_latex_escape(str(v)))
        lines.append(" & ".join(values) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    latex = "\n".join(lines)

    # Save alongside the CSV
    tex_file_name = os.path.splitext(input_file_name)[0] + ".tex"
    print(f"Saving LaTeX table to: {tex_file_name}")

    with open(tex_file_name, "w", encoding="utf-8") as f:
        f.write(latex)

    return latex


def _latex_escape(text: str) -> str:
    """Escape LaTeX special characters."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _pretty_metric_name(metric: str) -> str:

    # Remove train_/test_ prefixes
    metric = re.sub(r"^(train|test)_", "", metric)

    mapping = {
        "rmse": "RMSE",
        "mae": "MAE",
        "mse": "MSE",
        "mape": "MAPE",
        "smape": "SMAPE",
        "r2": r"$R^2$",
    }

    base = re.sub(r"_(mean|std)$", "", metric)
    pretty = mapping.get(base, _latex_escape(base.upper()))

    if metric.endswith("_std"):
        return f"SD({pretty})"

    return pretty


def _base_metric(metric: str) -> str:
    """rmse_mean -> rmse"""
    return re.sub(r"_(mean|std)$", "", metric)


def _merge_columns(df, metrics, best_rows=None):
    merged = pd.DataFrame()
    merged["Model"] = df["model"]

    for metric in metrics:

        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"

        values = []

        for i in range(len(df)):
            s = (
                f"{df.loc[i, mean_col]:.2f}"
                r" $\pm$ "
                f"{df.loc[i, std_col]:.2f}"
            )

            if best_rows is not None and i == best_rows[metric]:
                s = rf"\textbf{{{s}}}"

            values.append(s)

        merged[_pretty_metric_name(metric)] = values

    return merged


def old_merge_columns(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """
    Creates one column per metric with values such as:
        72.08 ± 13.20
    """

    merged = pd.DataFrame()

    if "model" in df.columns:
        merged["Model"] = df["model"]

    for metric in metrics:

        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"

        if mean_col not in df.columns:
            raise ValueError(f"Missing column '{mean_col}'")

        if std_col not in df.columns:
            raise ValueError(f"Missing column '{std_col}'")

        merged[_pretty_metric_name(mean_col)] = (
            df[mean_col].map(lambda x: f"{x:.2f}")
            + r" $\pm$ "
            + df[std_col].map(lambda x: f"{x:.2f}")
        )

    return merged


def to_latex_CV_results_table(
    input_file_name: str,
    train_columns: list[str],
    test_columns: list[str],
    table_title: str,
) -> str:
    """
    Creates a LaTeX table with grouped Train/Test columns.
    """

    df = pd.read_csv(input_file_name)

    # find best results
    best_train = {}
    for metric in train_columns:
        mean_col = f"{metric}_mean"

        if metric.endswith("r2"):
            best_train[metric] = df[mean_col].idxmax()
        else:
            best_train[metric] = df[mean_col].idxmin()

    best_test = {}
    for metric in test_columns:
        mean_col = f"{metric}_mean"

        if metric.endswith("r2"):
            best_test[metric] = df[mean_col].idxmax()
        else:
            best_test[metric] = df[mean_col].idxmin()

    train_df = _merge_columns(df, train_columns, best_train)
    test_df = _merge_columns(df, test_columns, best_test)

    train_headers = list(train_df.columns[1:])
    test_headers = list(test_df.columns[1:])

    n_train = len(train_headers)
    n_test = len(test_headers)

    lines = []

    lines.append(r"\begin{table}[htb]")
    lines.append(r"\centering")
    lines.append(
        rf"\caption{{{_latex_escape(table_title).rstrip('.')}.}}"
    )
    # no need to escape _ or other chars in labels
    lines.append(
        rf"\label{{tab:{Path(input_file_name).stem}}}"
    )

    # One left-aligned column for model, centered columns for metrics
    lines.append(
        r"\begin{tabular}{l" + "c" * (n_train + n_test) + "}"
    )

    lines.append(r"\toprule")

    # First header row
    lines.append(
        rf"& \multicolumn{{{n_train}}}{{c}}{{Train}}"
        rf" & \multicolumn{{{n_test}}}{{c}}{{Validation}} \\"
    )

    # cmidrules
    lines.append(
        rf"\cmidrule(lr){{2-{1+n_train}}}"
        rf"\cmidrule(lr){{{2+n_train}-{1+n_train+n_test}}}"
    )

    # Second header row
    header = ["Model"] + train_headers + test_headers
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")

    # Data rows
    for i in range(len(df)):
        row = [_latex_escape(str(train_df.iloc[i, 0]))]

        row.extend(str(train_df.iloc[i, j])
                   for j in range(1, len(train_df.columns)))
        row.extend(str(test_df.iloc[i, j])
                   for j in range(1, len(test_df.columns)))

        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    # save to a .tex file, with the same name as the CSV file but with .tex extension
    tex_file_name = os.path.splitext(input_file_name)[0] + ".tex"
    print(f"Saving LaTeX table to: {tex_file_name}")
    with open(tex_file_name, "w") as f:
        f.write("\n".join(lines))

    return "\n".join(lines)


if __name__ == "__main__":
    # Example usage

    # ----------------------------------------------------------------------
    # Create an example DataFrame
    # ----------------------------------------------------------------------

    df = pd.DataFrame({
        "model": ["PLS", "ElasticNet", "RandomForest", "SVR"],

        # Test metrics
        "rmse_mean": [72.08, 69.42, 65.11, 63.84],
        "rmse_std":  [13.20, 11.80, 10.45,  9.32],

        "mae_mean":  [56.15, 54.80, 50.93, 49.27],
        "mae_std":   [10.53,  9.91,  8.70,  8.15],

        "r2_mean":   [0.08, 0.16, 0.31, 0.38],
        "r2_std":    [0.21, 0.18, 0.14, 0.11],

        # Train metrics
        "train_rmse_mean": [70.50, 65.31, 41.22, 30.94],
        "train_rmse_std":  [3.53, 3.01, 2.15, 1.44],

        "train_mae_mean":  [53.66, 50.12, 29.11, 20.84],
        "train_mae_std":   [3.21, 2.85, 1.82, 1.20],

        "train_r2_mean":   [0.17, 0.29, 0.86, 0.94],
        "train_r2_std":    [0.04, 0.03, 0.02, 0.01],
    })

    # ----------------------------------------------------------------------
    # Save to a temporary CSV
    # ----------------------------------------------------------------------
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".csv",
        delete=False,
        newline=""
    ) as f:

        csv_name = f.name
        df.to_csv(csv_name, index=False)

    # ----------------------------------------------------------------------
    # Generate the LaTeX table
    # ----------------------------------------------------------------------

    latex = to_latex_CV_results_table(
        input_file_name=csv_name,
        train_columns=["train_rmse", "train_mae", "train_r2"],
        test_columns=["rmse", "mae", "r2"],
        table_title="Comparison of regression models (exp1_this)"
    )

    print(latex)

    # ----------------------------------------------------------------------
    # Remove temporary file
    # ----------------------------------------------------------------------

    os.remove(csv_name)
