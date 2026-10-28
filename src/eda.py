from __future__ import annotations

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from src.pipeline import load_readings, build_targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="challenge_data")
    parser.add_argument("--output", default="reports/eda")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    readings = load_readings(args.data)
    targets, quality = build_targets(readings)
    targets.to_csv(out / "aggregate_targets.csv", index=False)

    valid = readings.loc[readings["valid_cpu"]]
    cpu_summary = valid["avg_cpu"].describe(percentiles=[.05, .25, .5, .75, .95, .99])
    cpu_summary.to_csv(out / "cpu_distribution.csv", header=["avg_cpu"])

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(targets["timestamp"], targets["active_vm_count"])
    ax.axvline(targets["timestamp"].iloc[109], linestyle="--", label="110/24 split")
    ax.set(title="Active VMs by timestamp", xlabel="Source timestamp (seconds)", ylabel="Active VM count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "active_vm_count.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(targets["timestamp"], targets["aggregate_cpu_utilization"])
    ax.axvline(targets["timestamp"].iloc[109], linestyle="--", label="110/24 split")
    ax.set(title="Mean VM CPU utilization", xlabel="Source timestamp (seconds)", ylabel="CPU utilization (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "aggregate_cpu_utilization.png", dpi=160)
    plt.close(fig)

    # Segment summaries use metadata when available; missing optional metadata is reported.
    metadata_candidates = [Path(args.data) / "vm_metadata.csv.gz", Path(args.data) / "vm_metadata.csv"]
    metadata_path = next((p for p in metadata_candidates if p.exists()), None)
    if metadata_path is not None:
        metadata = pd.read_csv(metadata_path, compression="infer")
        # Segment at VM level: each VM contributes one trace-mean value, rather
        # than 134 repeated rows. This makes the median and p95 interpretable as
        # distributions across VMs, matching the report.
        vm_means = valid.groupby("vm_id", as_index=False)["avg_cpu"].mean().rename(
            columns={"avg_cpu": "vm_trace_mean_cpu"}
        )
        metadata_required = {"vm_id", "vm_category", "virtual_core_bucket", "memory_gb_bucket"}
        missing_columns = sorted(metadata_required - set(metadata.columns))
        if missing_columns:
            raise ValueError(f"Metadata file is missing required columns: {missing_columns}")
        if metadata["vm_id"].duplicated().any():
            raise ValueError("Metadata must contain exactly one row per vm_id.")
        joined = vm_means.merge(metadata, on="vm_id", how="outer", indicator=True, validate="one_to_one")
        metadata_quality = {
            "metadata_rows": int(len(metadata)),
            "metadata_unique_vms": int(metadata["vm_id"].nunique()),
            "metadata_missing_cells": int(metadata.isna().sum().sum()),
            "reading_vms_without_metadata": int((joined["_merge"] == "left_only").sum()),
            "metadata_vms_without_readings": int((joined["_merge"] == "right_only").sum()),
            "metadata_columns": list(metadata.columns),
        }
        quality.update(metadata_quality)
        joined = joined.loc[joined["_merge"] == "both"].copy()
        for col in ["vm_category", "virtual_core_bucket", "memory_gb_bucket"]:
            if col in joined.columns:
                summary = joined.groupby(col, dropna=False)["vm_trace_mean_cpu"].agg(
                    vm_count="count", mean="mean", median="median",
                    p95=lambda values: values.quantile(0.95),
                )
                summary.to_csv(out / f"segment_{col}.csv")

    (out / "data_quality.json").write_text(json.dumps(quality, indent=2))
    print(f"EDA outputs written to {out}")
    print(json.dumps(quality, indent=2))


if __name__ == "__main__":
    main()
