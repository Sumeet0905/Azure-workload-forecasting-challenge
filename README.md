<<<<<<< HEAD
# Azure Cloud Workload Forecasting — Atomity Take-Home

A reproducible forecasting and analysis pipeline for the Atomity Cloud Workload Forecasting & Analysis challenge, using the fixed `data-v1` release of Microsoft Azure Public Dataset V2.

## Deliverables

- `predictions.csv` — required 24-row forecast file.
- `reports/final_report.md` and `reports/final_report.pdf` — analysis, validation, error analysis, operational recommendation, and limitations.
- `src/` — data aggregation, leakage-aware validation, model selection, and prediction code.
- `tests/` — regression and contract tests.
- `reports/eda/` — data-quality summaries, segment summaries, and diagnostic figures.

## 1. Obtain the official data

Download `data-v1.tar.gz` from the [official challenge releases](https://github.com/atomityhq/forecasting-challenge/releases) and extract it so the project contains:

```text
challenge_data/
├── vm_cpu_readings.csv.gz
├── vm_metadata.csv.gz
├── DATA_CARD.md
└── SHA256SUMS
```

Verify checksums as described in the challenge README. The raw dataset is intentionally not bundled here and should not be committed to your submission repository.

## 2. Set up (Python 3.11+)

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS/Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Generate forecasts

Run from the project root:

```bash
python -m src.predict --data challenge_data --output predictions.csv
```

The output schema is exactly:

```text
timestamp,active_vm_count,aggregate_cpu_utilization
```

The output contains the final 24 timestamps in chronological order. For the official 134-timestamp release, the first 110 timestamps are used for fitting and rolling-origin model selection; the final 24 target values are excluded from fitting and model selection. Timestamps are preserved as source-relative values rather than being interpreted as wall-clock time.

## 4. Re-run EDA and report

```bash
python -m src.eda --data challenge_data --output reports/eda
python -m src.predict --data challenge_data --output predictions.csv
python scripts/build_report.py
```

## 5. Test

```bash
python -m pytest -q
```

The bundled suite currently passes six tests. Tests cover target aggregation, forecasting baselines, holdout separation, output schema, and a small reviewer smoke dataset.

## Method summary

- Constructs `active_vm_count` as the number of distinct VMs with a valid observation at each timestamp.
- Constructs `aggregate_cpu_utilization` as the arithmetic mean of VM-level `avg_cpu` values at each timestamp.
- Compares persistence, a recent moving average, and linear-trend extrapolation using chronological rolling-origin validation.
- Selects models on the training window only and clips forecasts to valid target domains.
- Includes a post-hoc public-holdout audit in the report for transparent diagnostics; those observed holdout values are not used to select or tune the forecasting model.

## Key findings and limitations

The curated release has 1,340,000 VM-level observations across 10,000 VMs and 134 timestamps. In the supplied trace, every VM appears at every timestamp, so `active_vm_count` is constant and persistence forecasts it exactly. The trace spans only about 11.1 hours, which is too short to learn daily or weekly seasonality reliably. Aggregate mean CPU is not a capacity-weighted demand estimate and hides VM-level peaks; operational use should combine this signal with service-level saturation, capacity-weighted utilization, and a monitored safety margin.

See `reports/final_report.md` for detailed findings, validation results, recommendations, and caveats.
=======
# Azure-workload-forecasting-challenge
>>>>>>> 2e034d875bc35dc725643fbaed79d876f1482b48
