# Cloud Workload Forecasting & Analysis

**Dataset:** Atomity `data-v1`, curated Microsoft Azure Public Dataset V2 subset  
**Forecast cadence:** 5 minutes  
**Training window:** first 110 aggregate timestamps (9 h 10 min)  
**Forecast horizon:** final 24 timestamps (2 hours)

## 1. Executive summary

We built the two required aggregate time series from 1,340,000 VM-level CPU readings covering 10,000 VMs and 134 five-minute timestamps. The source passed the core structural checks: every CPU reading was numeric and within `[0, 100]`, no duplicate `(timestamp, vm_id)` pairs were found, and the aggregate timeline had exactly 300 seconds between consecutive timestamps with no missing grid positions.

The most important finding is that **`active_vm_count` is constant at 10,000 for all 134 timestamps**. This is a consequence of the challenge's curated population: every selected VM has a complete 134-reading trace. The target is therefore perfectly predictable in this dataset, but it does not test real VM churn or autoscaling behavior. Metadata deletion times also mostly lie outside the approximately 11.1-hour observation window, so they should not be used to infer within-window activity when the row-level readings show no churn.

For `aggregate_cpu_utilization`, the target is the mean of all valid VM-level `avg_cpu` readings at a timestamp. Its full-series mean is **9.388%**, standard deviation **0.195 percentage points**, and range **8.984%–9.912%**. A persistence forecast (repeat the latest observed aggregate value) performed best among the three tested candidates under rolling-origin validation. Across three 24-step validation windows, persistence achieved mean window MAE **0.1848 percentage points** and mean window RMSE **0.2174**. The six-reading moving average and 24-reading linear trend were worse on average.

The final submission is a simple, auditable baseline rather than an unnecessarily complex model. The CPU forecast remains a short-horizon estimate of the *mean utilization across VMs*; it is not total CPU demand and cannot directly determine physical capacity without per-VM capacity weighting.

## 2. Data understanding and preprocessing

### Source and schema

The CPU readings file has `timestamp`, `vm_id`, `min_cpu`, `max_cpu`, and `avg_cpu`. The metadata file has one row per VM, with subscription/deployment identifiers, creation/deletion timestamps, VM-level CPU summaries, workload category, virtual-core bucket, and memory bucket. Source timestamps are seconds relative to the trace start, not wall-clock timestamps; we preserve them as supplied and do not invent a timezone or calendar interpretation.

### Quality checks

| Check | Result | Handling / implication |
|---|---:|---|
| CPU reading rows | 1,340,000 | Matches 10,000 × 134 |
| Unique VMs | 10,000 | Expected population |
| Aggregate timestamps | 134 | Required time-series length |
| Timestamp span | 0–39,900 seconds | 133 intervals × 300 seconds |
| Non-300-second gaps | 0 | Cadence is regular |
| Duplicate `(timestamp, vm_id)` rows | 0 | No duplicate collapse needed for the supplied data |
| Invalid or missing `avg_cpu` values | 0 | All readings are finite and in `[0, 100]` |
| Missing five-minute grid points | 0 | No target imputation required |
| Metadata rows / missing values | 10,000 / 0 | One complete metadata row per VM |

For robustness to future inputs, the implementation collapses duplicate VM/timestamp pairs by averaging valid readings before aggregation. This prevents a duplicate record from counting the same VM twice. Invalid CPU readings are excluded from both required targets. If a timestamp had no valid CPU readings, it would not be invented as a zero-demand observation; the pipeline reports timestamp gaps for explicit review.

### Target construction

At each timestamp, `active_vm_count` is the number of distinct VM IDs with a valid CPU observation. `aggregate_cpu_utilization` is the arithmetic mean of the valid per-VM `avg_cpu` values, expressed as a percentage. This is not a sum and does not weight VMs by virtual-core count.

The full target series is constructed before splitting. The first 110 timestamps are the only data passed to validation, model selection, and final forecasting. The last 24 timestamps supply the required output timestamps. Their observed target values are not used for fitting, tuning, or model selection.

## 3. Exploratory workload analysis

### Aggregate CPU behavior

Across the 134 timestamps, mean aggregate CPU utilization is 9.388%, median 9.367%, and standard deviation 0.195 percentage points. The minimum is 8.984% and maximum is 9.912%. In the first 110 observations (the modeling window), the range is 9.112%–9.912%, with mean 9.414%. This is a comparatively narrow aggregate series, even though individual VMs vary much more widely.

The trace contains a visibly elevated interval around source timestamps 15,900–20,400 seconds, where the aggregate series reaches 9.912%. Later, the series declines toward the lower end of its observed range, with a brief increase near 37,500 seconds. That behavior is consistent with local bursts and level changes rather than a clean, persistent linear trend. The short trace does not provide repeated days or weeks, so calendar seasonality is not identifiable and should not be assumed.

### VM-level heterogeneity

The source metadata category counts are 6,180 delay-insensitive VMs, 3,091 interactive VMs, and 729 unknown-category VMs. Using each VM's mean `avg_cpu` across its 134 readings, segment summaries are:

| Segment | VMs | Mean of per-VM trace means (%) | Median of per-VM trace means (%) | 95th percentile of per-VM trace means (%) |
|---|---:|---:|---:|---:|
| Delay-insensitive | 6,180 | 8.236 | 5.616 | 25.747 |
| Interactive | 3,091 | 11.493 | 5.565 | 46.025 |
| Unknown | 729 | 10.233 | 3.894 | 48.393 |

Interactive VMs have a higher mean trace utilization and a wider upper tail in this sample. This suggests that segment-aware monitoring could be useful operationally, but the aggregate series alone cannot show whether a specific segment is responsible for a particular timestamp-level spike.

The virtual-core distribution is skewed toward smaller VMs: 7,534 have 2 cores, 1,908 have 4, 486 have 8, 62 have 24, and 10 have more than 24. Segment differences exist, but some buckets are very small and should not be over-interpreted. In particular, a mean CPU percentage is not equivalent to CPU core consumption; a capacity-weighted target would be a separate, carefully defined extension.

### Lifetimes and churn

The metadata creation timestamp is zero for every selected VM, and deletion timestamps are generally far beyond the 0–39,900-second observation grid. The data card states that the eligible VMs were selected for complete traces across the observation window. Therefore, row-level observations support a constant active population of 10,000; the metadata fields do not demonstrate within-window churn for this curated subset. In a live platform, churn should be measured from timestamp-aligned lifecycle events or actual observation presence, not inferred from lifetime metadata on a mismatched time scale.

## 4. Forecasting approach

We compared three transparent forecasting methods independently for each target:

1. **Persistence:** repeat the most recent observed value for all 24 future steps.
2. **Six-observation moving average:** repeat the mean of the latest six observations (30 minutes).
3. **Linear trend:** fit a straight line to the latest 24 observations and extrapolate 24 steps.

All methods produce a 24-step forecast from a forecast origin. There are no calendar features because source timestamps have no wall-clock semantics and the history is too short to establish recurring daily or weekly patterns. The model is selected independently for each target using mean MAE across chronological validation windows. A constant training target is handled explicitly with exact persistence rather than allowing floating-point regression noise to determine the choice.

## 5. Validation methodology and results

### Time-aware validation

The final 24 observations are reserved for the required holdout. Inside the first 110 observations, we used three rolling-origin 24-step validation windows with origins at indices 62, 74, and 86. At each origin, only observations before that origin are used as history. This simulates repeatedly forecasting the next two hours from the information available at that point, without randomly shuffling time-series rows.

There are only three validation origins because the public training window is short relative to the two-hour forecast horizon. Results should therefore be treated as a useful but noisy estimate, not as a definitive performance guarantee.

### Validation metrics

MAE and RMSE are averaged across the three validation windows. For CPU utilization, errors are measured in **percentage points**. Normalized MAE divides mean window MAE by the range of the first 110 training observations (0.8005 percentage points). Relative improvement is `1 - candidate_MAE / persistence_MAE`; a negative value means the candidate is worse than persistence.

| Target | Model | Mean window MAE | Mean window RMSE | Normalized MAE | Relative MAE improvement vs persistence |
|---|---|---:|---:|---:|---:|
| Active VM count | Persistence | 0.000 | 0.000 | 0.000* | 0.0% |
| Active VM count | Moving average | 0.000 | 0.000 | 0.000* | 0.0% |
| Active VM count | Linear trend | ~0.000 | ~0.000 | ~0.000* | Not meaningful |
| Aggregate CPU (%) | Persistence | 0.1848 | 0.2174 | 0.2308 | 0.0% |
| Aggregate CPU (%) | Moving average | 0.2491 | 0.2790 | 0.3111 | -34.8% |
| Aggregate CPU (%) | Linear trend | 0.4339 | 0.4811 | 0.5420 | -134.8% |

`*` The active-VM target has zero range, so normalized MAE uses its mean as the fallback denominator; the target is constant and its error is exactly zero for persistence and moving average. Relative improvement against a zero-error baseline is not defined for a model with nonzero error.

Persistence is the strongest CPU baseline in this experiment. The moving average smooths the series but lags when the level changes; linear extrapolation overreacts to short-term slopes and compounds error over a 24-step horizon. The more complex trend is not justified by these validation results.

### Peak-period performance and failure analysis

For each validation origin, we defined a high-load point using the 90th percentile of CPU utilization in that origin's available history. This threshold is computed without looking forward from that origin. Only six validation points across the three windows exceeded their respective thresholds, all in the earliest window. On those six points, persistence MAE was approximately **0.073 percentage points**, while linear trend MAE was approximately **0.136 percentage points**. No points exceeded the later origins' thresholds, so this sample does not support a broad claim about peak forecasting performance. The small number of high-load cases is an important limitation.

Likely failure modes are abrupt level changes, short-lived bursts, and nonstationarity. Persistence is competitive when the series remains locally stable, but it cannot anticipate an unseen spike. The mean across 10,000 VMs also suppresses per-VM extremes; aggregate accuracy should not be interpreted as anomaly-detection accuracy or as a guarantee that no individual VM is saturated.

### Frozen holdout audit (post-hoc only)

After generating and saving `predictions.csv` with the model selected exclusively from the first 110 observations, we performed a post-hoc audit against the public final 24 observations. These results were not used to change the model or its parameters.

| Target | Holdout MAE | Holdout RMSE | Interpretation |
|---|---:|---:|---|
| Active VM count | 0.000 | 0.000 | Constant 10,000-VM population |
| Aggregate CPU utilization | 0.1468 percentage points | 0.1848 percentage points | The holdout mean was 9.269%, above the 9.1646% forecast, so persistence under-forecasted on average |

The CPU holdout MAE is lower than the mean rolling-validation MAE, but one holdout is only one two-hour period and should not be treated as proof of generalization. The forecast was a constant 9.1646% for all 24 timestamps, equal to the final training observation.

## 6. Operational recommendation

Use the aggregate CPU forecast as a **short-horizon monitoring signal**, not as the sole autoscaling input. A practical first policy is:

- refresh the forecast every five minutes;
- compare forecast utilization and the recent observed trend with a service-specific warning threshold;
- maintain a small, explicitly monitored guard band calibrated from rolling-origin under-forecast residuals;
- alert or pre-scale only when the forecast plus guard band approaches a threshold and the underlying service has enough spare capacity to respond;
- track actual-versus-forecast residuals and revisit the margin when the error distribution changes.

In this validation sample, only nine of 72 persistence validation points were under-forecasts. The 90th percentile of positive under-forecast residuals was approximately 0.108 percentage points, but it is based on only nine positive cases. Treat this as an initial diagnostic, not a reliable service-level guarantee; production policy should use a larger history and a service-specific cost of under-provisioning.

Over-provisioning wastes capacity and money; under-provisioning risks saturation, latency, and failed workloads. The acceptable trade-off depends on workload criticality and the cost of a missed spike. Importantly, the required aggregate CPU target is an unweighted mean percentage. It cannot directly size the number of CPU cores or hosts. For production capacity planning, add a defensible capacity-weighted utilization target and segment-level forecasts, and combine forecasts with per-service saturation signals and autoscaling constraints.

Because `active_vm_count` is constant in this challenge subset, it should not be used to claim that the model has learned VM churn. A realistic churn experiment needs arrivals/departures that occur within the observed timeline.

## 7. Limitations and future work

1. The trace covers only about 11.1 hours, insufficient for robust daily/weekly seasonality analysis.
2. Three rolling validation origins yield uncertain estimates for a 24-step horizon.
3. Complete-trace curation makes the active-VM count constant and removes the opportunity to evaluate churn forecasting.
4. Aggregate mean CPU hides per-VM peaks and is not capacity-weighted.
5. Segment summaries describe differences between VMs over the trace, not causal explanations for timestamp-level changes.
6. The public holdout audit is post-hoc and not a substitute for an unseen private evaluation set.

Useful next steps are (a) validate on additional chronological windows or a longer trace, (b) forecast segment-level aggregates, (c) estimate intervals from rolling-origin residuals once more validation windows are available, (d) add VM-level high-percentile/saturation targets, and (e) compare the simple baselines with regularized lag-based models only if the larger evaluation design supports reliable model selection.

## Reproducibility

From the repository root, install dependencies and run:

```bash
pip install -r requirements.txt
python -m src.eda --data challenge_data --output reports/eda
python -m src.predict --data challenge_data --output predictions.csv
pytest -q
```

The predictor accepts either the release's `.csv.gz` files or uncompressed `.csv` files. The required output contains exactly 24 chronological timestamps and the three columns specified by `SUBMISSION.md`. Source data is not included in the project archive; obtain it from the official `data-v1` release and do not commit the raw release to Git.
