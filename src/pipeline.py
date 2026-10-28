from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

READINGS_CANDIDATES = ("vm_cpu_readings.csv.gz", "vm_cpu_readings.csv")
METADATA_CANDIDATES = ("vm_metadata.csv.gz", "vm_metadata.csv")
TARGETS = ["active_vm_count", "aggregate_cpu_utilization"]


def load_readings(data_dir: str | Path) -> pd.DataFrame:
    base = Path(data_dir)
    path = next((base / name for name in READINGS_CANDIDATES if (base / name).exists()), None)
    if path is None:
        raise FileNotFoundError(
            f"Missing vm_cpu_readings.csv[.gz] in {data_dir}. Download/extract the data-v1 release."
        )
    df = pd.read_csv(path, compression="infer")
    required = {"timestamp", "vm_id", "avg_cpu"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Readings file is missing required columns: {sorted(missing)}")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["avg_cpu"] = pd.to_numeric(df["avg_cpu"], errors="coerce")
    df = df.dropna(subset=["timestamp", "vm_id"])
    # A valid CPU reading means avg_cpu is numeric and finite; out-of-range values
    # are treated as invalid rather than silently clipped.
    valid_cpu = np.isfinite(df["avg_cpu"]) & df["avg_cpu"].between(0, 100)
    df["valid_cpu"] = valid_cpu
    return df


def build_targets(readings: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Aggregate VM readings into required timestamp-level targets.

    Duplicate (timestamp, vm_id) records are diagnosed and collapsed to one VM
    observation per timestamp using the mean of its valid avg_cpu values.
    """
    df = readings.copy()
    duplicate_pairs = int(df.duplicated(["timestamp", "vm_id"], keep=False).sum())
    valid = df.loc[df["valid_cpu"]].copy()
    # One row per VM per timestamp, satisfying the active-VM counting contract.
    per_vm = (
        valid.groupby(["timestamp", "vm_id"], as_index=False)
        .agg(avg_cpu=("avg_cpu", "mean"))
    )
    targets = (
        per_vm.groupby("timestamp")
        .agg(
            active_vm_count=("vm_id", "nunique"),
            aggregate_cpu_utilization=("avg_cpu", "mean"),
        )
        .sort_index()
        .reset_index()
    )
    ts = targets["timestamp"].to_numpy(dtype=float)
    deltas = np.diff(ts)
    report = {
        "input_rows": int(len(df)),
        "valid_cpu_rows": int(len(valid)),
        "invalid_cpu_rows": int((~df["valid_cpu"]).sum()),
        "duplicate_timestamp_vm_rows": duplicate_pairs,
        "unique_vms_with_valid_reading": int(per_vm["vm_id"].nunique()),
        "timestamp_count": int(targets["timestamp"].nunique()),
        "timestamp_min": float(targets["timestamp"].min()),
        "timestamp_max": float(targets["timestamp"].max()),
        "expected_cadence_seconds": 300,
        "non_300_second_gaps": int(np.sum(~np.isclose(deltas, 300))),
        "missing_grid_timestamps_between_min_max": (
            int(round((ts[-1] - ts[0]) / 300)) + 1 - len(ts) if len(ts) > 1 else 0
        ),
    }
    return targets, report


def forecast_persistence(history: np.ndarray, horizon: int) -> np.ndarray:
    return np.repeat(float(history[-1]), horizon)


def forecast_moving_average(history: np.ndarray, horizon: int, window: int = 6) -> np.ndarray:
    return np.repeat(float(np.mean(history[-min(window, len(history)):])), horizon)


def forecast_linear_trend(history: np.ndarray, horizon: int, window: int = 24) -> np.ndarray:
    y = np.asarray(history[-min(window, len(history)):], dtype=float)
    if len(y) < 2:
        return forecast_persistence(y, horizon)
    x = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(x, y, deg=1)
    future_x = np.arange(len(y), len(y) + horizon, dtype=float)
    return intercept + slope * future_x


FORECASTERS = {
    "persistence": forecast_persistence,
    "moving_average_6": forecast_moving_average,
    "linear_trend_24": forecast_linear_trend,
}


def rolling_origin_validation(
    series: np.ndarray,
    horizon: int = 24,
    min_train: int = 62,
    stride: int = 12,
) -> pd.DataFrame:
    """Validate only on chronological windows inside the first 110 observations."""
    y = np.asarray(series, dtype=float)
    # End of validation windows cannot exceed the public training window.
    public_train_end = min(110, len(y))
    rows = []
    origins = list(range(min_train, public_train_end - horizon + 1, stride))
    # Ensure a useful latest origin is included without exceeding public window.
    latest = public_train_end - horizon
    if latest >= min_train and latest not in origins:
        origins.append(latest)
    for origin in sorted(set(origins)):
        actual = y[origin:origin + horizon]
        history = y[:origin]
        for name, fn in FORECASTERS.items():
            pred = np.asarray(fn(history, horizon), dtype=float)
            err = pred - actual
            rows.append({
                "model": name,
                "origin_index": origin,
                "horizon": horizon,
                "mae": float(np.mean(np.abs(err))),
                "rmse": float(np.sqrt(np.mean(err ** 2))),
            })
    return pd.DataFrame(rows, columns=["model", "origin_index", "horizon", "mae", "rmse"])


def select_and_forecast(series: np.ndarray, horizon: int = 24) -> tuple[str, np.ndarray, pd.DataFrame]:
    """Select by rolling-origin validation MAE, then refit forecast on all allowed history."""
    series = np.asarray(series, dtype=float)
    # The caller passes only the allowed training history. For the official dataset
    # this is exactly the first 110 points; shorter smoke datasets use their full
    # pre-holdout history. Cap at 110 to preserve the official split if extra rows
    # are accidentally passed by a caller.
    public_train = series[:110]
    if len(public_train) < 2:
        raise ValueError("At least two training observations are required.")
    # Constant targets (as in this curated complete-trace population) should use
    # the exact persistence forecast instead of numerically equivalent regressions.
    if np.allclose(public_train, public_train[0], rtol=0.0, atol=1e-12):
        validation = rolling_origin_validation(public_train, horizon=horizon)
        return "persistence", forecast_persistence(public_train, horizon), validation
    validation = rolling_origin_validation(public_train, horizon=horizon)
    if validation.empty:
        return "persistence", forecast_persistence(public_train, horizon), validation
    scores = validation.groupby("model", as_index=False).agg(
        mean_mae=("mae", "mean"), mean_rmse=("rmse", "mean")
    ).sort_values(["mean_mae", "mean_rmse", "model"])
    selected = str(scores.iloc[0]["model"]) if len(scores) else "persistence"
    forecast = FORECASTERS[selected](public_train, horizon)
    return selected, np.asarray(forecast, dtype=float), validation
