import numpy as np
import pandas as pd

from src.pipeline import build_targets, forecast_persistence, forecast_moving_average, select_and_forecast


def test_targets_count_vm_once_per_timestamp():
    df = pd.DataFrame({
        "timestamp": [0, 0, 0, 300],
        "vm_id": ["a", "a", "b", "a"],
        "avg_cpu": [20.0, 40.0, 60.0, 50.0],
        "valid_cpu": [True, True, True, True],
    })
    targets, _ = build_targets(df)
    assert targets.loc[targets.timestamp == 0, "active_vm_count"].iloc[0] == 2
    assert targets.loc[targets.timestamp == 0, "aggregate_cpu_utilization"].iloc[0] == 45.0


def test_baselines():
    hist = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    assert np.allclose(forecast_persistence(hist, 3), [6, 6, 6])
    assert np.allclose(forecast_moving_average(hist, 2, window=3), [5, 5])


def test_forecast_is_24_finite_values():
    y = np.linspace(10, 20, 110)
    name, forecast, validation = select_and_forecast(y, horizon=24)
    assert len(forecast) == 24
    assert np.isfinite(forecast).all()
    assert name in {"persistence", "moving_average_6", "linear_trend_24"}
    assert len(validation) > 0


def test_constant_target_uses_exact_persistence():
    y = np.full(110, 10000.0)
    name, forecast, _ = select_and_forecast(y, horizon=24)
    assert name == "persistence"
    assert np.all(forecast == 10000.0)


def test_model_selection_does_not_use_values_after_index_109():
    train = np.linspace(5.0, 9.0, 110)
    series_a = np.concatenate([train, np.zeros(24)])
    series_b = np.concatenate([train, np.full(24, 100.0)])
    name_a, pred_a, _ = select_and_forecast(series_a, horizon=24)
    name_b, pred_b, _ = select_and_forecast(series_b, horizon=24)
    assert name_a == name_b
    assert np.allclose(pred_a, pred_b)


def test_cli_supports_small_smoke_dataset(tmp_path):
    import subprocess
    import sys

    rows = []
    for t in range(30):
        for vm, cpu in (("vm-a", 10 + (t % 3)), ("vm-b", 20 + (t % 2))):
            rows.append({"timestamp": t * 300, "vm_id": vm, "avg_cpu": cpu})
    pd.DataFrame(rows).to_csv(tmp_path / "vm_cpu_readings.csv", index=False)
    output = tmp_path / "predictions.csv"
    reports = tmp_path / "reports"
    subprocess.run(
        [sys.executable, "-m", "src.predict", "--data", str(tmp_path),
         "--output", str(output), "--reports", str(reports)],
        check=True,
    )
    result = pd.read_csv(output)
    assert list(result.columns) == ["timestamp", "active_vm_count", "aggregate_cpu_utilization"]
    assert len(result) == 24
    assert result["timestamp"].is_monotonic_increasing
    assert result["timestamp"].is_unique
    assert np.array_equal(result["timestamp"].to_numpy(), np.arange(6 * 300, 30 * 300, 300))
    values = result.to_numpy(dtype=float)
    assert np.isfinite(values).all()
    assert (result["active_vm_count"] >= 0).all()
    assert result["aggregate_cpu_utilization"].between(0, 100).all()
