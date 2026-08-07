'''
IEB2 dataset corresponds to a small experiment measuring glucose (GLC) levels
with 19 participants. Each participant has two sessions: S1 and S2 (120 minutes
later). This script evaluates personalized regression for GLC prediction with a
rigorous LOSO protocol:

1) Residual personalization:
   - trains a global model on all other participants
   - compute a subject-specific scalar correction from S1 residual mean
   - apply correction to S2 predictions

2) Gaussian Process personalization (Bayesian update):
   - trains a global GP prior on all other participants
   - when S1 arrives, update posterior by conditioning on S1
   - predict S2 with personalized posterior
'''

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from datasets_util.naming_conventions import LOGIDENTIFIER, DatasetConfig

RANDOM_STATE = 42  # seed for random number generators for reproducibility
S1_LABEL = "S1"  # label identifying session 1
S2_LABEL = "S2"  # label identifying session 2


@dataclass
class GPBundle:
    imputer: SimpleImputer
    scaler: StandardScaler
    gp: GaussianProcessRegressor
    x_train_scaled: np.ndarray
    y_train: np.ndarray


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    r2_value = float("nan") if y_true_arr.size < 2 else float(
        r2_score(y_true_arr, y_pred_arr))
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))),
        "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "r2": r2_value,
        "bias": float(np.mean(y_pred - y_true)),
    }


def _read_if_exists(path: str) -> pd.DataFrame | None:
    if os.path.exists(path) and os.path.getsize(path) > 0:
        print(f"Reading: {path}")
        return pd.read_csv(path)
    return None


def load_dataset(dataset_config: DatasetConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a train pool (train+validation) and a locked test set."""
    train_csv = dataset_config.get_selected_features_file_name(
        train_or_test="train")
    test_csv = dataset_config.get_selected_features_file_name(
        train_or_test="test")
    validation_csv = dataset_config.get_selected_features_file_name(
        train_or_test="validation")

    train_frames = []
    for path in [train_csv, validation_csv]:
        df_part = _read_if_exists(path)
        if df_part is not None:
            train_frames.append(df_part)

    test_df = _read_if_exists(test_csv)

    if not train_frames:
        raise FileNotFoundError(
            "No training input CSV found. Expected train and/or validation split CSV files."
        )
    if test_df is None:
        raise FileNotFoundError(
            "No test CSV found. Expected a test split CSV file."
        )

    train_pool = pd.concat(train_frames, ignore_index=True)
    print(LOGIDENTIFIER +
          "Using concatenated train/validation split as the training pool and a separate locked test split.")
    required_cols = ["participant_id", "session_id", "GLC"]
    missing = [col for col in required_cols if col not in train_pool.columns]
    if missing:
        raise ValueError(
            f"Input data must contain {required_cols}. Missing columns: {missing}"
        )
    missing = [col for col in required_cols if col not in test_df.columns]
    if missing:
        raise ValueError(
            f"Test data must contain {required_cols}. Missing columns: {missing}"
        )

    train_pool = train_pool.copy()
    test_df = test_df.copy()
    for frame_name, frame in [("train_pool", train_pool), ("test_df", test_df)]:
        frame["session_id"] = frame["session_id"].astype(str).str.strip()
        frame = frame[frame["session_id"].isin([S1_LABEL, S2_LABEL])].copy()
        frame["GLC"] = pd.to_numeric(frame["GLC"], errors="coerce")
        frame.dropna(subset=["participant_id",
                     "session_id", "GLC"], inplace=True)
        frame.reset_index(drop=True, inplace=True)
        if frame_name == "train_pool":
            train_pool = frame
        else:
            test_df = frame

    print(LOGIDENTIFIER + f"Loaded train pool for LOSO: {train_pool.shape}")
    print(LOGIDENTIFIER +
          f"Train pool participants: {train_pool['participant_id'].nunique()}")
    print(LOGIDENTIFIER +
          f"Train pool rows by session: {train_pool['session_id'].value_counts().to_dict()}")
    print(LOGIDENTIFIER + f"Loaded locked test split: {test_df.shape}")
    print(LOGIDENTIFIER +
          f"Locked test participants: {test_df['participant_id'].nunique()}")
    print(LOGIDENTIFIER +
          f"Locked test rows by session: {test_df['session_id'].value_counts().to_dict()}")

    return train_pool.reset_index(drop=True), test_df.reset_index(drop=True)


def get_feature_columns(dataset_config: DatasetConfig, df: pd.DataFrame) -> list[str]:
    '''Discard metadata columns and return only numeric feature columns for model training.'''
    metadata_columns = set(dataset_config.get_chosen_metadata_columns())
    metadata_columns.update(dataset_config.get_segments_columns())
    metadata_columns.update(
        dataset_config.get_modality_expanded_segments_columns())
    metadata_columns.update(
        ["participant_id", "session_id", "datetime", "GLC"])

    feature_cols: list[str] = []
    for col in df.columns:
        if col in metadata_columns:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)

    if not feature_cols:
        raise ValueError(
            "No numeric feature columns were found after removing metadata columns.")

    print(LOGIDENTIFIER +
          f"Number of feature columns used: {len(feature_cols)}")
    return feature_cols


def fit_global_residual_model(train_df: pd.DataFrame, feature_cols: list[str]) -> Pipeline:
    '''Fit a global Ridge regression model with preprocessing and return the pipeline.'''
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("regressor", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
    ])

    model.fit(train_df[feature_cols], train_df["GLC"])
    return model


def tune_global_residual_model(train_df: pd.DataFrame, feature_cols: list[str]) -> Pipeline:
    '''Select Ridge alpha using grouped inner cross-validation on the training pool.'''
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("regressor", Ridge(random_state=RANDOM_STATE)),
    ])

    groups = train_df["participant_id"].astype(str)
    unique_groups = groups.nunique()
    n_splits = max(2, min(5, unique_groups))
    inner_cv = GroupKFold(n_splits=n_splits)

    param_grid = {
        "regressor__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
    }

    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=inner_cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        refit=True,
    )
    search.fit(train_df[feature_cols], train_df["GLC"], groups=groups)
    print(LOGIDENTIFIER +
          f"Best Ridge alpha: {search.best_params_['regressor__alpha']}")
    return search.best_estimator_


def fit_dummy_regressor(train_df: pd.DataFrame, feature_cols: list[str]) -> DummyRegressor:
    """Fit a constant dummy baseline on the provided training data."""
    model = DummyRegressor(strategy="mean")
    model.fit(train_df[feature_cols], train_df["GLC"])
    return model


def fit_global_gp_model(train_df: pd.DataFrame, feature_cols: list[str]) -> GPBundle:
    '''Fit a global Gaussian Process model with preprocessing and return a GPBundle.'''
    x_train = train_df[feature_cols]
    y_train = train_df["GLC"].to_numpy(dtype=float)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    x_train_imputed = imputer.fit_transform(x_train)
    x_train_scaled = scaler.fit_transform(x_train_imputed)

    kernel = (
        C(1.0, (1e-3, 1e3))
        * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
        + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-6, 1e1))
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=2,
        random_state=RANDOM_STATE,
    )
    gp.fit(x_train_scaled, y_train)

    return GPBundle(
        imputer=imputer,
        scaler=scaler,
        gp=gp,
        x_train_scaled=x_train_scaled,
        y_train=y_train,
    )


def tune_global_gp_model(train_df: pd.DataFrame, feature_cols: list[str]) -> GPBundle:
    '''Select a GP kernel family and alpha using grouped inner cross-validation.'''
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("gp", GaussianProcessRegressor(
            normalize_y=True,
            random_state=RANDOM_STATE,
        )),
    ])

    kernel_candidates = [
        C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-3,
                                                                         1e3)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-6, 1e1)),
        C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3),
                                     nu=1.5) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-6, 1e1)),
        C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3),
                                     nu=2.5) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-6, 1e1)),
    ]

    groups = train_df["participant_id"].astype(str)
    unique_groups = groups.nunique()
    n_splits = max(2, min(5, unique_groups))
    inner_cv = GroupKFold(n_splits=n_splits)

    param_grid = {
        "gp__kernel": kernel_candidates,
        "gp__alpha": [1e-6, 1e-4, 1e-2],
        "gp__n_restarts_optimizer": [0, 1],
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=inner_cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        refit=True,
    )
    search.fit(train_df[feature_cols], train_df["GLC"], groups=groups)
    best_pipeline = search.best_estimator_

    print(LOGIDENTIFIER + f"Best GP params: {search.best_params_}")

    imputer = best_pipeline.named_steps["imputer"]
    scaler = best_pipeline.named_steps["scaler"]
    gp = best_pipeline.named_steps["gp"]

    x_train = train_df[feature_cols]
    y_train = train_df["GLC"].to_numpy(dtype=float)
    x_train_scaled = scaler.transform(imputer.transform(x_train))

    return GPBundle(
        imputer=imputer,
        scaler=scaler,
        gp=gp,
        x_train_scaled=x_train_scaled,
        y_train=y_train,
    )


def gp_predict(bundle: GPBundle, x_df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    '''Predict using a Gaussian Process model with preprocessing.'''
    x = x_df[feature_cols].to_numpy(dtype=float)
    x_scaled = bundle.scaler.transform(bundle.imputer.transform(x))
    return bundle.gp.predict(x_scaled)


def gp_personalized_predict_s2(
    bundle: GPBundle,
    subject_s1: pd.DataFrame,
    subject_s2: pd.DataFrame,
    feature_cols: list[str],
) -> np.ndarray:
    '''Perform personalized prediction for S2 using a global GP model and S1 data.'''
    x_s1 = subject_s1[feature_cols]
    y_s1 = subject_s1["GLC"].to_numpy(dtype=float)
    x_s2 = subject_s2[feature_cols]

    x_s1_scaled = bundle.scaler.transform(bundle.imputer.transform(x_s1))
    x_s2_scaled = bundle.scaler.transform(bundle.imputer.transform(x_s2))

    x_post = np.vstack([bundle.x_train_scaled, x_s1_scaled])
    y_post = np.concatenate([bundle.y_train, y_s1])

    # Bayesian update with fixed kernel hyperparameters from global prior.
    posterior_gp = GaussianProcessRegressor(
        kernel=bundle.gp.kernel_,
        alpha=1e-6,
        normalize_y=True,
        optimizer=None,
        random_state=RANDOM_STATE,
    )
    posterior_gp.fit(x_post, y_post)
    return posterior_gp.predict(x_s2_scaled)


def evaluate_locked_test_set(
    dataset_config: DatasetConfig,
    train_pool: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate the same methods on the locked test split without using it for training."""
    if test_df.empty:
        print(LOGIDENTIFIER +
              "Locked test split is empty; skipping final test-set evaluation.")
        return pd.DataFrame(), pd.DataFrame()

    train_pool = train_pool.copy()
    test_df = test_df.copy()

    residual_model = tune_global_residual_model(train_pool, feature_cols)
    gp_bundle = tune_global_gp_model(train_pool, feature_cols)
    baseline1_model = fit_dummy_regressor(train_pool, feature_cols)

    per_subject_rows: list[dict[str, Any]] = []
    participants = sorted(
        test_df["participant_id"].astype(str).unique().tolist())

    for participant in participants:
        subject_df = test_df[test_df["participant_id"].astype(
            str) == participant].copy()
        subject_s1 = subject_df[subject_df["session_id"] == S1_LABEL].copy()
        subject_s2 = subject_df[subject_df["session_id"] == S2_LABEL].copy()

        if subject_s1.empty or subject_s2.empty:
            print(
                LOGIDENTIFIER
                + f"Skipping locked-test participant {participant}: missing S1 or S2 data "
                + f"(S1 rows={len(subject_s1)}, S2 rows={len(subject_s2)})."
            )
            continue

        y_s1 = subject_s1["GLC"].to_numpy(dtype=float)
        y_s2 = subject_s2["GLC"].to_numpy(dtype=float)

        yhat_s1_global = residual_model.predict(subject_s1[feature_cols])
        yhat_s2_global = residual_model.predict(subject_s2[feature_cols])
        scalar_correction = float(np.mean(y_s1 - yhat_s1_global))
        residual_metrics = regression_metrics(
            y_s2, yhat_s2_global + scalar_correction)
        per_subject_rows.append({
            "split": "locked_test",
            "participant_id": participant,
            "method": "residual_personalization",
            "n_s1": int(len(subject_s1)),
            "n_s2": int(len(subject_s2)),
            "scalar_correction": scalar_correction,
            **residual_metrics,
        })

        baseline1_metrics = regression_metrics(
            y_s2,
            baseline1_model.predict(subject_s2[feature_cols]),
        )
        per_subject_rows.append({
            "split": "locked_test",
            "participant_id": participant,
            "method": "baseline_dummy_no_subject_data",
            "n_s1": int(len(subject_s1)),
            "n_s2": int(len(subject_s2)),
            "scalar_correction": np.nan,
            **baseline1_metrics,
        })

        baseline2_train_df = pd.concat(
            [train_pool, subject_s1], ignore_index=True)
        baseline2_model = fit_dummy_regressor(baseline2_train_df, feature_cols)
        baseline2_metrics = regression_metrics(
            y_s2,
            baseline2_model.predict(subject_s2[feature_cols]),
        )
        per_subject_rows.append({
            "split": "locked_test",
            "participant_id": participant,
            "method": "baseline_dummy_with_s1",
            "n_s1": int(len(subject_s1)),
            "n_s2": int(len(subject_s2)),
            "scalar_correction": np.nan,
            **baseline2_metrics,
        })

        baseline3_metrics = regression_metrics(
            y_s2,
            repeat_subject_s1_as_prediction(subject_s1, len(subject_s2)),
        )
        per_subject_rows.append({
            "split": "locked_test",
            "participant_id": participant,
            "method": "baseline_repeat_s1_glc",
            "n_s1": int(len(subject_s1)),
            "n_s2": int(len(subject_s2)),
            "scalar_correction": np.nan,
            **baseline3_metrics,
        })

        gp_metrics = regression_metrics(
            y_s2,
            gp_personalized_predict_s2(
                gp_bundle, subject_s1, subject_s2, feature_cols),
        )
        per_subject_rows.append({
            "split": "locked_test",
            "participant_id": participant,
            "method": "gp_personalization",
            "n_s1": int(len(subject_s1)),
            "n_s2": int(len(subject_s2)),
            "scalar_correction": np.nan,
            **gp_metrics,
        })

    if not per_subject_rows:
        return pd.DataFrame(), pd.DataFrame()

    per_subject_df = pd.DataFrame(per_subject_rows)
    summary_df = (
        per_subject_df.groupby("method", as_index=False)
        .agg(
            participants=("participant_id", "nunique"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            bias_mean=("bias", "mean"),
            bias_std=("bias", "std"),
        )
        .sort_values(by="rmse_mean", ascending=True)
        .reset_index(drop=True)
    )

    print("\nLocked test-set summary:")
    print(summary_df.to_string(index=False))

    return per_subject_df, summary_df


def repeat_subject_s1_as_prediction(subject_s1: pd.DataFrame, n_predictions: int) -> np.ndarray:
    """Repeat S1 GLC values cyclically to match the size of the S2 set."""
    if subject_s1.empty:
        raise ValueError(
            "subject_s1 must not be empty when using the S1-repeat baseline")

    s1_values = subject_s1["GLC"].to_numpy(dtype=float)
    if n_predictions <= 0:
        return np.array([], dtype=float)

    return np.resize(s1_values, n_predictions)


def loso_cross_validation_for_ieb2(
    dataset_config: DatasetConfig,
    train_pool: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''Perform Leave-One-Subject-Out (LOSO) cross-validation for IEB2 dataset.
    Returns per-subject metrics and a summary of all methods.'''
    if train_pool is None:
        train_pool, _ = load_dataset(dataset_config)

    feature_cols = get_feature_columns(dataset_config, train_pool)

    participants = sorted(
        train_pool["participant_id"].astype(str).unique().tolist())
    per_subject_rows: list[dict[str, Any]] = []
    skipped: list[str] = []

    for participant in participants:
        participant_mask = train_pool["participant_id"].astype(
            str) == participant
        subject_df = train_pool[participant_mask].copy()
        train_df = train_pool[~participant_mask].copy()

        subject_s1 = subject_df[subject_df["session_id"] == S1_LABEL].copy()
        subject_s2 = subject_df[subject_df["session_id"] == S2_LABEL].copy()

        if subject_s1.empty or subject_s2.empty:
            print(
                LOGIDENTIFIER
                + f"Skipping participant {participant}: missing S1 or S2 data "
                + f"(S1 rows={len(subject_s1)}, S2 rows={len(subject_s2)})."
            )
            skipped.append(participant)
            continue

        print(LOGIDENTIFIER + f"LOSO participant: {participant}")

        # Method 1: residual personalization (scalar correction).
        residual_model = tune_global_residual_model(train_df, feature_cols)
        y_s1 = subject_s1["GLC"].to_numpy(dtype=float)
        y_s2 = subject_s2["GLC"].to_numpy(dtype=float)

        yhat_s1_global = residual_model.predict(subject_s1[feature_cols])
        yhat_s2_global = residual_model.predict(subject_s2[feature_cols])
        scalar_correction = float(np.mean(y_s1 - yhat_s1_global))
        yhat_s2_residual_personalized = yhat_s2_global + scalar_correction

        residual_metrics = regression_metrics(
            y_s2, yhat_s2_residual_personalized)
        per_subject_rows.append({
            "participant_id": participant,
            "s2-s1": (y_s2[0]-y_s1[0]),
            "method": "residual_personalization",
            "n_s1": int(len(subject_s1)),
            "n_s2": int(len(subject_s2)),
            "scalar_correction": scalar_correction,
            **residual_metrics,
        })

        # Baseline 1: dummy regressor trained without any data from the held-out subject.
        baseline1_model = fit_dummy_regressor(train_df, feature_cols)
        yhat_s2_baseline1 = baseline1_model.predict(subject_s2[feature_cols])
        baseline1_metrics = regression_metrics(y_s2, yhat_s2_baseline1)
        per_subject_rows.append({
            "participant_id": participant,
            "method": "baseline_dummy_no_subject_data",
            "n_s1": int(len(subject_s1)),
            "n_s2": int(len(subject_s2)),
            "scalar_correction": np.nan,
            **baseline1_metrics,
        })

        # Baseline 2: dummy regressor that also includes S1 from the target subject.
        baseline2_train_df = pd.concat(
            [train_df, subject_s1], ignore_index=True)
        baseline2_model = fit_dummy_regressor(baseline2_train_df, feature_cols)
        yhat_s2_baseline2 = baseline2_model.predict(subject_s2[feature_cols])
        baseline2_metrics = regression_metrics(y_s2, yhat_s2_baseline2)
        per_subject_rows.append({
            "participant_id": participant,
            "method": "baseline_dummy_with_s1",
            "n_s1": int(len(subject_s1)),
            "n_s2": int(len(subject_s2)),
            "scalar_correction": np.nan,
            **baseline2_metrics,
        })

        # Baseline 3: directly repeat the S1 measured GLC for the S2 prediction.
        yhat_s2_baseline3 = repeat_subject_s1_as_prediction(
            subject_s1, len(subject_s2))
        baseline3_metrics = regression_metrics(y_s2, yhat_s2_baseline3)
        per_subject_rows.append({
            "participant_id": participant,
            "method": "baseline_repeat_s1_glc",
            "n_s1": int(len(subject_s1)),
            "n_s2": int(len(subject_s2)),
            "scalar_correction": np.nan,
            **baseline3_metrics,
        })

        # Method 2: GP personalization via posterior update with S1.
        gp_bundle = tune_global_gp_model(train_df, feature_cols)
        yhat_s2_gp_personalized = gp_personalized_predict_s2(
            gp_bundle,
            subject_s1,
            subject_s2,
            feature_cols,
        )

        gp_metrics = regression_metrics(y_s2, yhat_s2_gp_personalized)
        per_subject_rows.append({
            "participant_id": participant,
            "method": "gp_personalization",
            "n_s1": int(len(subject_s1)),
            "n_s2": int(len(subject_s2)),
            "scalar_correction": np.nan,
            **gp_metrics,
        })

    if not per_subject_rows:
        raise RuntimeError(
            "LOSO did not run for any participant. Check session_id labels and input data columns."
        )

    per_subject_df = pd.DataFrame(per_subject_rows)
    summary_df = (
        per_subject_df.groupby("method", as_index=False)
        .agg(
            participants=("participant_id", "nunique"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            bias_mean=("bias", "mean"),
            bias_std=("bias", "std"),
        )
        .sort_values(by="rmse_mean", ascending=True)
        .reset_index(drop=True)
    )

    if skipped:
        print(LOGIDENTIFIER +
              f"Skipped participants ({len(skipped)}): {skipped}")

    print("\nPer-method LOSO summary:")
    print(summary_df.to_string(index=False))

    return per_subject_df, summary_df


def save_results(
    dataset_config: DatasetConfig,
    per_subject_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    test_per_subject_df: pd.DataFrame | None = None,
    test_summary_df: pd.DataFrame | None = None,
) -> None:
    output_dir = dataset_config.machine_learning_path
    os.makedirs(output_dir, exist_ok=True)

    per_subject_path = os.path.join(
        output_dir,
        "loso_ieb2_personalization_subject_metrics.csv",
    )
    summary_path = os.path.join(
        output_dir,
        "loso_ieb2_personalization_summary.csv",
    )

    per_subject_df.to_csv(per_subject_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(LOGIDENTIFIER + f"Saved per-subject metrics to: {per_subject_path}")
    print(LOGIDENTIFIER + f"Saved summary metrics to: {summary_path}")

    if test_per_subject_df is not None and test_summary_df is not None:
        test_per_subject_path = os.path.join(
            output_dir,
            "locked_test_ieb2_personalization_subject_metrics.csv",
        )
        test_summary_path = os.path.join(
            output_dir,
            "locked_test_ieb2_personalization_summary.csv",
        )
        test_per_subject_df.to_csv(test_per_subject_path, index=False)
        test_summary_df.to_csv(test_summary_path, index=False)
        print(LOGIDENTIFIER +
              f"Saved locked-test per-subject metrics to: {test_per_subject_path}")
        print(LOGIDENTIFIER +
              f"Saved locked-test summary metrics to: {test_summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LOSO personalization for IEB2 (residual and GP Bayesian update)."
    )
    parser.add_argument(
        "json_file",
        metavar="JSON_FILE",
        help="Path to the input JSON configuration file.",
    )

    args = parser.parse_args()
    dataset_config = DatasetConfig(args.json_file)

    train_pool, test_df = load_dataset(dataset_config)

    # concatenate train_pool and test_df:
    all_data = pd.concat([train_pool, test_df], ignore_index=True)

    print("Using the LOSO routine on all data...")
    per_subject, summary = loso_cross_validation_for_ieb2(
        dataset_config, train_pool=all_data)

    print("Using the LOSO routine on the train/validation pool only...")
    print("Final untouched test-set evaluation using the locked split...")
    test_per_subject, test_summary = evaluate_locked_test_set(
        dataset_config,
        train_pool,
        test_df,
        get_feature_columns(dataset_config, train_pool),
    )

    save_results(dataset_config, per_subject, summary,
                 test_per_subject, test_summary)
