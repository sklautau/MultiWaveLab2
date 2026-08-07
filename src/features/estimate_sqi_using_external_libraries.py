"""SQI estimators with a legacy-compatible wrapper API.

This module used to compare several external SQI implementations. The current
codebase only maintains the NeuroKit2- and custom-based estimators used by the
active PPG processing pipeline, so this wrapper now exposes those methods using
the same result shape consumed by SQI comparison scripts.

I would choose these five:

templatematch
Excellent beat morphology metric.
Sensitive to motion artifacts and distorted pulses.
ho2025
Completely different principle.
Measures consistency between two independent beat detectors.
Good for identifying erroneous peak detections.
entropy
Captures randomness and irregularity.
Detects noisy segments that may still have plausible morphology.
perfusion
Measures physiological signal strength (AC/DC ratio).
Useful when contact pressure or peripheral perfusion changes.
relative_power
Frequency-domain measure.
Detects broadband noise and loss of cardiac spectral content.
"""

import logging
from signal_processing.ppg_quality import estimate_sqi_custom_version
from signal_processing.ppg_quality import _sqi_neurokit_tm
from datasets_util.util_visualize_plots import plot_signal_with_sqi
from datasets_util.util_visualize_plots import plot_rmse_matrix
from datasets_util.util_various_methods import pairwise_mse
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np

NEUROKIT_METHOD_QUALITIES = [
    "templatematch",
    "ho2025",
    "entropy",
    # "perfusion",
    # "relative_power"
]

# Ensure project root is available when this file is run directly.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _format_result(name: str, sqi_for_each_signal_sample: np.ndarray) -> dict[str, Any]:
    return {
        "name": name,
        "sqi_per_pulse": np.array([]),
        "sqi_for_each_signal_sample": sqi_for_each_signal_sample,
    }


def _safe_run_estimator(
    ppg: np.ndarray,
    fs: int,
    name: str,
    estimator: Callable[[np.ndarray, int], np.ndarray],
) -> dict[str, Any] | None:
    """
    Execute an SQI estimator safely.

    Performs:
      - exception handling
      - shape validation
      - finite-value validation
      - range clipping
      - result formatting
    """
    try:
        sqi = np.asarray(estimator(ppg, fs), dtype=float)
    except Exception:
        logger.exception("Estimator '%s' failed.", name)
        return None

    if sqi.ndim != 1:
        logger.warning(
            "%s returned an array with shape %s; expected a 1-D vector.",
            name,
            sqi.shape,
        )
        return None

    if len(sqi) != len(ppg):
        logger.warning(
            "%s returned %d samples; expected %d.",
            name,
            len(sqi),
            len(ppg),
        )
        return None

    if not np.all(np.isfinite(sqi)):
        logger.warning(
            "%s produced NaN or Inf values.",
            name,
        )
        return None

    outside = (sqi < 0.0) | (sqi > 1.0)
    if np.any(outside):
        logger.warning(
            "%s produced %d values outside [0, 1]. "
            "Values will be clipped.",
            name,
            np.count_nonzero(outside),
        )

    sqi = np.clip(sqi, 0.0, 1.0)

    return _format_result(name, sqi)


logger = logging.getLogger(__name__)


def estimate_sqi_neurokit(
    ppg: np.ndarray,
    fs: int,
    method_quality: str = "templatematch",
) -> dict[str, Any] | None:
    """
    Estimate a sample-wise SQI using NeuroKit.

    For NeuroKit methods that require both a filtered and a raw signal
    (currently 'perfusion' and 'relative_power'), this wrapper passes the
    available signal as both inputs. This is a pragmatic workaround when
    the original raw signal is unavailable.
    """
    ppg = np.asarray(ppg, dtype=float)

    ppg_raw = None
    if method_quality in {"perfusion", "relative_power"}:
        ppg_raw = ppg

    def estimator(signal: np.ndarray, sampling_rate: int) -> np.ndarray:
        result = _sqi_neurokit_tm(
            signal,
            sampling_rate,
            method_quality=method_quality,
            ppg_raw=ppg_raw
        )

        try:
            sqi = result["sqi_for_each_signal_sample"]
        except KeyError:
            raise KeyError(
                f"NeuroKit result does not contain "
                f"'sqi_for_each_signal_sample'. "
                f"Returned keys: {list(result.keys())}"
            )

        return np.asarray(sqi, dtype=float)

    return _safe_run_estimator(
        ppg=ppg,
        fs=fs,
        name=f"NeuroKit2 - {method_quality}",
        estimator=estimator,
    )


def old_estimate_sqi_neurokit(ppg: np.ndarray, fs: int, method_quality: str = "templatematch") -> dict[str, Any] | None:
    ppg_raw_for_method = None
    if method_quality in {"perfusion", "relative_power"}:
        # "Trick" required by NeuroKit: explicitly provide the same signal as raw input.
        # NeuroKit requires `ppg_raw` for these SQIs.
        # When only a filtered signal is available, we intentionally pass the
        # filtered signal as both `ppg` and `ppg_raw` so the estimator can run.
        # This changes the semantics of the SQI but avoids an exception.
        ppg_raw_for_method = np.asarray(ppg, dtype=float)

    return _safe_run_estimator(
        np.asarray(ppg, dtype=float),
        fs,
        "NeuroKit2 - " + method_quality,
        lambda x, rate: np.asarray(
            _sqi_neurokit_tm(
                x,
                rate,
                method_quality=method_quality,
                ppg_raw=ppg_raw_for_method,
            )[
                "sqi_for_each_signal_sample"
            ],
            dtype=float,
        ),
    )


def estimate_sqi_custom(ppg: np.ndarray, fs: int) -> dict[str, Any] | None:
    return _safe_run_estimator(
        np.asarray(ppg, dtype=float),
        fs,
        "Custom (spectral+peaks)",
        lambda x, rate: np.asarray(
            estimate_sqi_custom_version(x, rate), dtype=float),
    )


def run_all_sqi_estimators(ppg: np.ndarray, fs: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    report: list[tuple[str, bool]] = []

    # NeuroKit2 variants controlled by method_quality
    for method_quality in NEUROKIT_METHOD_QUALITIES:
        result = estimate_sqi_neurokit(ppg, fs, method_quality=method_quality)
        if result is not None:
            results.append(result)
            report.append((method_quality, True))
        else:
            report.append((method_quality, False))

    # Additional non-NeuroKit estimator(s)
    custom_result = estimate_sqi_custom(ppg, fs)
    if custom_result is not None:
        results.append(custom_result)
        report.append(("custom_spectral_peaks", True))
    else:
        report.append(("custom_spectral_peaks", False))

    print("SQI estimator report:")
    for method_quality, ok in report:
        status = "OK" if ok else "FAILED"
        print(f"  - {method_quality}: {status}")

    return results


if __name__ == "__main__":
    fs = 60
    t = np.linspace(0, 30, 30 * fs)
    ppg = 0.5 + 0.3 * np.sin(2 * np.pi * 1.2 * t)
    ppg += 0.05 * np.random.randn(len(t))

    results = run_all_sqi_estimators(ppg, fs)
    if len(results) >= 2:
        plot_signal_with_sqi(ppg, results, fs, note="Demo PPG signal")
        mse_results = pairwise_mse(
            results, key_name="sqi_for_each_signal_sample")
        print("MSE between estimators:", mse_results)
        plot_rmse_matrix(mse_results)
