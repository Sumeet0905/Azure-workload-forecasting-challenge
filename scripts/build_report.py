from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak,
    KeepTogether,
)

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
EDA = REPORTS / "eda"
OUT = REPORTS / "final_report.pdf"


def make_audit_plot() -> Path:
    targets = pd.read_csv(EDA / "aggregate_targets.csv")
    predictions = pd.read_csv(ROOT / "predictions.csv")
    fig, ax = plt.subplots(figsize=(8.1, 3.0))
    ax.plot(targets.timestamp.iloc[:110] / 3600, targets.aggregate_cpu_utilization.iloc[:110],
            label="Observed training history", linewidth=1.5)
    ax.plot(targets.timestamp.iloc[110:] / 3600, targets.aggregate_cpu_utilization.iloc[110:],
            label="Observed holdout (post-hoc audit)", linewidth=1.5)
    ax.plot(predictions.timestamp / 3600, predictions.aggregate_cpu_utilization,
            label="Frozen persistence forecast", linestyle="--", linewidth=1.8)
    ax.axvline(targets.timestamp.iloc[109] / 3600, linestyle=":", linewidth=1.2,
               label="Training / holdout boundary")
    ax.set_xlabel("Source timestamp (hours from trace start)")
    ax.set_ylabel("Mean CPU utilization (%)")
    ax.set_title("Aggregate CPU forecast and post-hoc holdout audit")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7, ncol=2, frameon=False)
    fig.tight_layout()
    path = EDA / "forecast_holdout_audit.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def make_pdf() -> None:
    quality = json.loads((EDA / "data_quality.json").read_text())
    validation = pd.read_csv(REPORTS / "validation_summary.csv")
    targets = pd.read_csv(EDA / "aggregate_targets.csv")
    predictions = pd.read_csv(ROOT / "predictions.csv")
    category = pd.read_csv(EDA / "segment_vm_category.csv")
    audit_plot = make_audit_plot()

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CoverTitle", parent=styles["Title"], fontSize=23,
                              leading=28, alignment=TA_CENTER, spaceAfter=10))
    styles.add(ParagraphStyle(name="CoverSub", parent=styles["Normal"], fontSize=10,
                              leading=15, alignment=TA_CENTER, textColor=colors.HexColor("#4B5563")))
    styles.add(ParagraphStyle(name="Section", parent=styles["Heading1"], fontSize=15,
                              leading=18, spaceBefore=2, spaceAfter=7, textColor=colors.HexColor("#17324D")))
    styles.add(ParagraphStyle(name="Subsection", parent=styles["Heading2"], fontSize=10.5,
                              leading=13, spaceBefore=5, spaceAfter=4, textColor=colors.HexColor("#245A81")))
    styles.add(ParagraphStyle(name="BodySmall", parent=styles["BodyText"], fontSize=8.6,
                              leading=11.4, spaceAfter=5))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=7.4,
                              leading=9.2, spaceAfter=2))

    def P(text, style="BodySmall"):
        return Paragraph(text, styles[style])

    def make_table(rows, widths, font=7.5):
        t = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17324D")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), font),
            ("LEADING", (0, 0), (-1, -1), font + 2),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D5DDE5")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F7FA")]),
        ]))
        return t

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#D5DDE5"))
        canvas.line(18 * mm, 14 * mm, A4[0] - 18 * mm, 14 * mm)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#667085"))
        canvas.drawString(18 * mm, 9 * mm, "Cloud Workload Forecasting & Analysis | Atomity data-v1")
        canvas.drawRightString(A4[0] - 18 * mm, 9 * mm, f"Page {doc.page}")
        canvas.restoreState()

    story = []
    # Page 1 — executive summary and data contract
    story += [Spacer(1, 24 * mm), P("Cloud Workload Forecasting &amp; Analysis", "CoverTitle"),
              P("Reproducible short-horizon forecasts from the curated Azure VM trace", "CoverSub"),
              Spacer(1, 13 * mm), P("Executive summary", "Section"),
              P("This project constructs the two required aggregate targets from 1,340,000 VM-level CPU readings: distinct active VMs and mean VM CPU utilization. It compares persistence, a six-reading moving average, and a 24-reading linear trend using chronological rolling-origin validation. Persistence is selected for both targets; the active-VM target is constant in this curated complete-trace population, while persistence is the strongest CPU baseline among the tested methods."),
              P("Dataset and split", "Subsection"),
              P("The data-v1 subset contains 10,000 VMs with 134 readings per VM at five-minute cadence, covering source timestamps 0–39,900 seconds. The first 110 aggregate timestamps are the modeling window; the final 24 timestamps are the required two-hour holdout. Source timestamps are elapsed seconds, not wall-clock time."),
              make_table([["Quality check", "Observed result"],
                          ["CPU reading rows", f"{quality['input_rows']:,}"],
                          ["Valid avg_cpu rows", f"{quality['valid_cpu_rows']:,}"],
                          ["Unique VMs", f"{quality['unique_vms_with_valid_reading']:,}"],
                          ["Aggregate timestamps", str(quality['timestamp_count'])],
                          ["Duplicate VM/timestamp rows", str(quality['duplicate_timestamp_vm_rows'])],
                          ["Non-300-second gaps", str(quality['non_300_second_gaps'])],
                          ["Missing grid timestamps", str(quality['missing_grid_timestamps_between_min_max'])],
                          ["Metadata missing cells", str(quality.get('metadata_missing_cells', 'not checked'))]],
                         [62 * mm, 90 * mm]),
              Spacer(1, 6 * mm),
              P("Key caveat: the dataset was curated to include only VMs with complete traces. Therefore, active_vm_count is exactly 10,000 at every timestamp. This task does not provide a meaningful test of VM arrivals, deletions, or autoscaling churn.", "BodySmall"),
              PageBreak()]

    # Page 2 — EDA
    story += [P("1. Exploratory analysis", "Section"),
              P("Aggregate CPU utilization averages 9.388% across all timestamps, with a standard deviation of 0.195 percentage points and a range of 8.984%–9.912%. In the first 110 timestamps, the range is 9.112%–9.912%. The short trace shows local level changes, not enough repeated cycles to establish daily or weekly seasonality."),
              Image(str(EDA / "aggregate_cpu_utilization.png"), width=170 * mm, height=62 * mm),
              P("VM-level heterogeneity", "Subsection"),
              P("The metadata population comprises 6,180 delay-insensitive VMs, 3,091 interactive VMs, and 729 unknown-category VMs. Segment summaries below are computed across per-VM trace means, so each VM contributes one value."),
              make_table([["VM category", "VMs", "Mean (%)", "Median (%)", "P95 (%)"]] + [
                  [str(r.vm_category), f"{int(r.vm_count):,}", f"{r['mean']:.3f}", f"{r['median']:.3f}", f"{r.p95:.3f}"]
                  for _, r in category.iterrows()], [47 * mm, 20 * mm, 25 * mm, 25 * mm, 25 * mm]),
              Spacer(1, 4 * mm),
              P("Interactive VMs have a higher mean trace utilization and a wider upper tail in this sample. These are descriptive associations, not causal explanations for timestamp-level changes. The aggregate mean is unweighted by VM core count and suppresses individual VM extremes.", "BodySmall"),
              PageBreak()]

    # Page 3 — modeling
    cpu_rows = validation[validation.target == "aggregate_cpu_utilization"].copy()
    model_order = ["persistence", "moving_average_6", "linear_trend_24"]
    cpu_rows["_order"] = cpu_rows.model.map({m: i for i, m in enumerate(model_order)})
    cpu_rows = cpu_rows.sort_values("_order")
    active_rows = validation[validation.target == "active_vm_count"].set_index("model")
    story += [P("2. Forecasting and validation", "Section"),
              P("Three transparent candidates were evaluated independently for each target: (1) persistence repeats the last value, (2) moving average repeats the mean of the latest six observations (30 minutes), and (3) linear trend fits a line to the latest 24 observations and extrapolates 24 steps. No calendar features were introduced because the source has no wall-clock semantics and the observation window is short."),
              P("Validation protocol", "Subsection"),
              P("The first 110 points were evaluated using three rolling-origin windows with origins at indices 62, 74, and 86. Each validation forecast predicts the next 24 points using only history available before that origin. MAE and RMSE are averaged across windows. CPU errors are percentage points, not relative percent error."),
              make_table([["CPU model", "Mean window MAE", "Mean window RMSE", "Normalized MAE", "Relative MAE vs persistence"]] + [
                  [str(r.model), f"{r.mean_window_mae:.4f}", f"{r.mean_window_rmse:.4f}",
                   f"{r.normalized_mae_by_training_range:.3f}",
                   "baseline" if r.model == "persistence" else f"{100*r.relative_mae_improvement_vs_persistence:.1f}%"]
                  for _, r in cpu_rows.iterrows()], [38 * mm, 29 * mm, 30 * mm, 25 * mm, 39 * mm], font=7.0),
              Spacer(1, 5 * mm),
              P("Persistence achieves CPU MAE 0.1848 percentage points and RMSE 0.2174. The moving average has MAE 0.2491 and the linear trend 0.4339; neither improves on persistence. Smoothing lags level changes, while a fitted short-term slope compounds error over a two-hour horizon."),
              P("For active_vm_count, persistence and moving average both achieve zero validation error because the target is constant. Linear trend has only negligible floating-point residuals; persistence is selected explicitly for a constant series. Relative improvement against a zero-error baseline is not meaningful.", "BodySmall"),
              PageBreak()]

    # Page 4 — errors and holdout audit
    hold_actual = targets.iloc[110:].aggregate_cpu_utilization.to_numpy()
    hold_pred = predictions.aggregate_cpu_utilization.to_numpy()
    hold_mae = float(abs(hold_actual - hold_pred).mean())
    hold_rmse = float(((hold_actual - hold_pred) ** 2).mean() ** 0.5)
    hold_bias = float((hold_pred - hold_actual).mean())
    story += [P("3. Error analysis and frozen holdout audit", "Section"),
              P("Peak-period check", "Subsection"),
              P("At each validation origin, high load was defined as a value above the 90th percentile of that origin's training history. Only six validation points met this criterion, all in the earliest window. Persistence MAE on these six points was 0.073 percentage points, versus 0.136 for linear trend. This is too small a sample to support a broad claim about peak prediction."),
              P("Post-hoc holdout audit", "Subsection"),
              P("After predictions were frozen, we compared them with the public final 24 observations for diagnostic reporting only. These values were not used for model selection or tuning. The holdout CPU MAE is %.4f percentage points and RMSE is %.4f; mean forecast-minus-actual bias is %.4f percentage points. The forecast is constant at %.4f%%, equal to the final training observation." % (hold_mae, hold_rmse, hold_bias, float(predictions.aggregate_cpu_utilization.iloc[0]))),
              Image(str(audit_plot), width=170 * mm, height=63 * mm),
              P("The holdout actual mean is %.3f%%, above the 9.1646%% forecast; the frozen forecast under-forecasts on average in this period. A single two-hour period is not proof of generalization and should not be used to retune the model." % float(hold_actual.mean())),
              PageBreak()]

    # Page 5 — operational recs, limitations, reproducibility
    story += [P("4. Operational recommendation", "Section"),
              P("Treat this aggregate forecast as a short-horizon monitoring signal, not a standalone autoscaling decision. Refresh it every five minutes; compare it with service-specific warning thresholds and recent observed trends; calibrate a guard band from under-forecast residuals; and monitor actual-versus-forecast residuals for drift. The 90th percentile of positive persistence under-forecast residuals in the validation windows is about 0.108 percentage points, but it is based on only nine positive residuals and is not a production guarantee."),
              P("Capacity-planning interpretation", "Subsection"),
              P("The target is the arithmetic mean of VM-level CPU percentages. It is not total CPU demand and cannot directly determine the number of cores or hosts. Production capacity planning should add capacity-weighted utilization, segment-level forecasts, per-service saturation signals, and explicit scaling constraints. Under-provisioning can increase latency and failures; over-provisioning wastes capacity, so margins must reflect workload criticality."),
              P("Limitations", "Subsection"),
              P("• Only about 11.1 hours of data are available; daily/weekly seasonality cannot be learned.\n• Three rolling-origin windows provide a noisy estimate for a 24-step horizon.\n• Complete-trace curation makes active_vm_count constant and prevents a churn forecast evaluation.\n• Aggregate mean CPU hides per-VM peaks and is not capacity-weighted.\n• The public holdout audit is post-hoc and does not replace an unseen private evaluation set."),
              P("Reproduction", "Subsection"),
              P("From the repository root, install requirements, place the official data-v1 files in challenge_data/, and run:<br/><font name='Courier'>python -m src.eda --data challenge_data --output reports/eda</font><br/><font name='Courier'>python -m src.predict --data challenge_data --output predictions.csv</font><br/><font name='Courier'>pytest -q</font><br/><font name='Courier'>python scripts/build_report.py</font>"),
              P("The submission file has exactly 24 chronological rows and the columns timestamp, active_vm_count, aggregate_cpu_utilization. The raw challenge dataset is excluded from the project archive. Six automated tests cover aggregation, baselines, holdout isolation, output contract, and a small reviewer smoke dataset.", "BodySmall")]

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=19 * mm, title="Cloud Workload Forecasting & Analysis",
                            author="Challenge submission")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    make_pdf()
