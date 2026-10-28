from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from src.pipeline import build_targets, load_readings, select_and_forecast, TARGETS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="challenge_data")
    parser.add_argument("--output", default="predictions.csv")
    parser.add_argument("--reports", default="reports")
    args = parser.parse_args()

    readings = load_readings(args.data)
    targets, quality = build_targets(readings)
    # Keep the challenge's official 110/24 split when all 134 timestamps are present.
    # For a reviewer smoke-test dataset, generalize to all timestamps before its final
    # 24 rows rather than hard-coding the production dataset length.
    targets = targets.sort_values("timestamp").reset_index(drop=True)
    if len(targets) < 26:
        raise ValueError(
            f"Need at least 26 aggregate timestamps (2 training + 24 holdout); found {len(targets)}."
        )
    train_end = min(110, len(targets) - 24)
    train = targets.iloc[:train_end].copy()
    holdout_timestamps = targets.iloc[-24:]["timestamp"].to_numpy()

    output = pd.DataFrame({"timestamp": holdout_timestamps})
    validation_records = []
    selected_models = {}
    for target in TARGETS:
        selected, pred, validation = select_and_forecast(train[target].to_numpy(), horizon=24)
        selected_models[target] = selected
        output[target] = pred
        validation["target"] = target
        validation_records.append(validation)

    # Enforce submission contract.
    output = output[["timestamp", "active_vm_count", "aggregate_cpu_utilization"]]
    output["active_vm_count"] = output["active_vm_count"].clip(lower=0).round().astype(int)
    output["aggregate_cpu_utilization"] = output["aggregate_cpu_utilization"].clip(0, 100)
    if len(output) != 24 or output["timestamp"].duplicated().any():
        raise ValueError("Output must contain exactly 24 unique holdout timestamps.")
    if not np.isfinite(output[["active_vm_count", "aggregate_cpu_utilization"]].to_numpy()).all():
        raise ValueError("Predictions contain non-finite values.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    report_dir = Path(args.reports)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "data_quality.json").write_text(json.dumps(quality, indent=2))
    (report_dir / "selected_models.json").write_text(json.dumps(selected_models, indent=2))
    if validation_records:
        validation_all = pd.concat(validation_records, ignore_index=True)
        if validation_all.empty:
            validation_all = pd.DataFrame(columns=["model", "origin_index", "horizon", "mae", "rmse", "target"])
            validation_all.to_csv(report_dir / "rolling_validation.csv", index=False)
            pd.DataFrame(columns=["target", "model", "mean_window_mae", "mean_window_rmse",
                                  "persistence_mean_window_mae", "relative_mae_improvement_vs_persistence",
                                  "normalized_mae_by_training_range", "validation_origins"]).to_csv(
                                      report_dir / "validation_summary.csv", index=False)
        else:
            validation_all.to_csv(report_dir / "rolling_validation.csv", index=False)
        summaries = []
        for target in TARGETS:
            part = validation_all.loc[validation_all["target"] == target]
            base = part.loc[part["model"] == "persistence"]
            base_mae = float(base["mae"].mean()) if len(base) else float("nan")
            train_values = train[target].to_numpy(dtype=float)
            scale = float(np.max(train_values) - np.min(train_values))
            if scale <= 1e-12:
                scale = max(abs(float(np.mean(train_values))), 1.0)
            for model_name, group in part.groupby("model"):
                mae = float(group["mae"].mean())
                summaries.append({
                    "target": target, "model": model_name,
                    "mean_window_mae": mae,
                    "mean_window_rmse": float(group["rmse"].mean()),
                    "persistence_mean_window_mae": base_mae,
                    "relative_mae_improvement_vs_persistence": (1.0 - mae / base_mae) if base_mae > 1e-12 else (0.0 if mae <= 1e-12 else float("nan")),
                    "normalized_mae_by_training_range": mae / scale,
                    "validation_origins": int(group["origin_index"].nunique()),
                })
        if not validation_all.empty:
            pd.DataFrame(summaries).to_csv(report_dir / "validation_summary.csv", index=False)

    print(f"Wrote {len(output)} rows to {output_path}")
    print("Selected models:", selected_models)
    print("Data quality:", json.dumps(quality, indent=2))


if __name__ == "__main__":
    main()
