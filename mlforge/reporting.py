"""Human-friendly reports for people who are new to machine learning."""
from __future__ import annotations

from typing import Iterable


def _percent(value):
    return f"{value * 100:.1f}%" if isinstance(value, (int, float)) else "—"


def _number(value):
    return f"{value:.3f}" if isinstance(value, (int, float)) else "—"


def format_pipeline_report(pipeline) -> str:
    """Return a readable explanation of the completed pipeline run.

    This deliberately avoids exposing raw dictionaries as the primary output.
    The structured ``leaderboard()`` API remains available for applications.
    """
    problem = pipeline.problem_def
    lines = [
        "=" * 72,
        "MLFORGE RESULTS",
        "=" * 72,
        f"Task       : {problem.problem_type.title()}",
        f"Target     : {problem.target}",
    ]

    if pipeline.eda_report_:
        rows, cols = pipeline.eda_report_["shape"]
        lines.append(f"Dataset    : {rows:,} rows, {cols:,} columns")

    if not pipeline.best_:
        lines += ["", "No model has been trained yet. Run pipeline.fit(df) first."]
        return "\n".join(lines)

    best = pipeline.best_
    metrics = best.accuracy_metrics
    # The selector's ``accuracy`` criterion uses weighted F1 for
    # classification, which is safer than raw accuracy when classes are
    # imbalanced.  Make that choice visible instead of showing a misleading
    # accuracy-only ranking.
    primary = "f1_weighted" if problem.problem_type == "classification" else "r2"
    primary_label = "Weighted F1" if problem.problem_type == "classification" else "R² score"
    primary_formatter = _percent if problem.problem_type == "classification" else _number

    selection_metric_description = "weighted F1" if problem.problem_type == "classification" else "R² score"
    lines += [
        "",
        "RECOMMENDED MODEL",
        "-" * 72,
        f"Model      : {best.name}",
        f"{primary_label:<11}: {primary_formatter(metrics.get(primary))}",
    ]

    if problem.problem_type == "classification":
        lines += [
            f"Accuracy   : {_percent(metrics.get('accuracy'))}",
            f"F1 score   : {_percent(metrics.get('f1_weighted'))}",
            f"ROC-AUC    : {_percent(metrics.get('roc_auc'))}",
        ]
    else:
        lines += [
            f"MAE        : {_number(metrics.get('mae'))}",
            f"RMSE       : {_number(metrics.get('rmse'))}",
        ]

    lines += [
        "",
        "WHY THIS MODEL?",
        "-" * 72,
        "It had the highest overall score after balancing prediction quality "
        "with your relevancy preferences.",
    ]
    if problem.problem_type == "classification":
        lines.append(
            "For classification, 'prediction quality' is weighted F1 so that "
            "class imbalance is considered, not just raw accuracy."
        )
    weights = problem.relevancy.normalized_weights()
    for criterion, weight in weights.items():
        if criterion == "accuracy":
            value = pipeline.evaluator.primary_accuracy_score(metrics)
        else:
            value = best.relevancy_scores.get(criterion, 0.0)
        lines.append(f"- {criterion.replace('_', ' ').title():<20} {_percent(value)} (importance {_percent(weight)})")

    lines += [
        "",
        "MODEL COMPARISON",
        "-" * 72,
        f"Rank  Model                 {'Selection quality':<18}Overall   Decision",
        "-" * 72,
    ]
    for rank, row in enumerate(pipeline.leaderboard(), 1):
        if row.get("failure_stage"):
            status = f"failed ({row['failure_stage']})"
        elif row["disqualified"]:
            status = f"not eligible ({row.get('reason') or 'constraint'})"
        else:
            status = "SELECTED" if row["name"] == best.name else "candidate"
        quality = row.get(f"metric.{primary}")
        quality_text = _percent(quality) if problem.problem_type == "classification" else _number(quality)
        score = _number(row.get("final_score"))
        lines.append(f"{rank:<5}{row['name']:<22}{quality_text:<10}{score:<10}{status}")

    lines += [
        "",
        "HOW TO READ THIS",
        "-" * 72,
        f"Selection quality shows {selection_metric_description} on unseen test data.",
        "Overall combines quality with the relevancy weights you configured.",
        "Use the selected model for predictions with pipeline.predict(new_data).",
        "=" * 72,
    ]
    return "\n".join(lines)


def format_automl_report(result) -> str:
    """Return a beginner-friendly report for an :class:`AutoMLResult`.

    Supervised runs reuse the established pipeline report so the existing
    reporting output remains the source of truth.  Clustering has no target
    or predictive accuracy metric, so it receives a short, focused report
    with the selected cluster count and silhouette score when available.
    """
    if result.task in {"classification", "regression"}:
        lines = [result.pipeline.report(), "", "AUTOML NOTES", "-" * 72]
        lines.append(result.explanation)
        if result.warnings:
            lines += ["", "WARNINGS"]
            lines.extend(f"- {warning}" for warning in result.warnings)
        return "\n".join(lines)

    rows = result.leaderboard()
    best = rows[0] if rows else {}
    shape = result.eda.get("shape") if isinstance(result.eda, dict) else None
    lines = [
        "=" * 72,
        "MLFORGE AUTOML RESULTS",
        "=" * 72,
        "Task       : Clustering",
        "Target     : (none — unsupervised)",
    ]
    if shape:
        lines.append(f"Dataset    : {shape[0]:,} rows, {shape[1]:,} columns")
    if best:
        lines += [
            "",
            "RECOMMENDED CLUSTERING",
            "-" * 72,
            f"Method     : {best.get('name', 'kmeans')}",
            f"Clusters   : {best.get('n_clusters', '—')}",
            f"Silhouette : {_number(best.get('silhouette'))}",
        ]
    lines += [
        "",
        "EXPLANATION",
        "-" * 72,
        result.explanation,
    ]
    if result.warnings:
        lines += ["", "WARNINGS"]
        lines.extend(f"- {warning}" for warning in result.warnings)
    lines += ["", "Use result.predict(new_data) to assign new rows to a cluster.", "=" * 72]
    return "\n".join(lines)
