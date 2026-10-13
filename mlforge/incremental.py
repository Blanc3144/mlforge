"""
Step 8: Keep improving as new data arrives.

Two strategies, chosen automatically per model:
  1. True incremental (partial_fit) when the finalized model supports it
     (e.g. SGDClassifier/SGDRegressor) — cheap, no full retrain.
  2. Retrain-on-combined-data otherwise — old + new data are combined
     (with an optional sliding window / max history size) and the whole
     model-selection tournament can optionally be re-run, in case a
     different algorithm has become the more relevant/accurate choice
     as the data distribution shifts.

Every update is versioned so you can compare performance over time and
roll back if a new batch of data made things worse.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd

from .pipeline import MLPipeline


@dataclass
class VersionRecord:
    version: int
    timestamp: str
    n_rows_used: int
    model_name: str
    accuracy_metrics: dict
    note: str = ""


class IncrementalLearner:
    def __init__(self, pipeline: MLPipeline, max_history_rows: Optional[int] = None):
        self.pipeline = pipeline
        self.max_history_rows = max_history_rows
        self.history_df: pd.DataFrame = pipeline.raw_df_.copy() if pipeline.raw_df_ is not None else None
        self.versions: List[VersionRecord] = []
        self._record_version(note="initial fit")

    def _record_version(self, note: str = ""):
        best = self.pipeline.best_
        metrics = best.accuracy_metrics if best else {}
        if best and self.pipeline.X_eval_ is not None:
            predictions = best.model.predict(self.pipeline.X_eval_)
            if self.pipeline.problem_def.problem_type == "classification":
                proba = None
                classes = None
                if hasattr(best.model, "predict_proba"):
                    p = best.model.predict_proba(self.pipeline.X_eval_)
                    if p.shape[1] == 2:
                        proba = p[:, 1]
                    else:
                        proba = p
                    classes = getattr(best.model, "classes_", None)
                metrics = self.pipeline.evaluator.evaluate_classification(
                    self.pipeline.y_eval_, predictions, proba, classes=classes
                )
            else:
                metrics = self.pipeline.evaluator.evaluate_regression(
                    self.pipeline.y_eval_, predictions
                )
            best.accuracy_metrics = metrics
        self.versions.append(VersionRecord(
            version=len(self.versions) + 1,
            timestamp=datetime.now(timezone.utc).isoformat(),
            n_rows_used=len(self.history_df) if self.history_df is not None else 0,
            model_name=best.name if best else "n/a",
            accuracy_metrics=metrics,
            note=note,
        ))

    def update(self, new_df: pd.DataFrame, retrain_full_tournament: bool = False,
               verbose: bool = True) -> "IncrementalLearner":
        """
        Feed in a new batch of labeled data. If the current best model
        supports partial_fit, it's updated cheaply in place. Otherwise
        (or if retrain_full_tournament=True), the model-selection
        tournament is re-run on the combined history, so a better-suited
        algorithm can take over if the data has drifted.
        """
        target = self.pipeline.problem_def.target
        model = self.pipeline.best_.model
        supports_partial = hasattr(model, "partial_fit")

        if self.history_df is None:
            self.history_df = new_df.copy()
        else:
            self.history_df = pd.concat([self.history_df, new_df], ignore_index=True)
            if self.max_history_rows and len(self.history_df) > self.max_history_rows:
                self.history_df = self.history_df.tail(self.max_history_rows).reset_index(drop=True)

        if supports_partial and not retrain_full_tournament:
            X_new = self.pipeline.preprocessor.transform(new_df)
            y_new = new_df[target]
            model.partial_fit(X_new, y_new)
            if verbose:
                print(f"[incremental] partial_fit applied to {self.pipeline.best_.name} "
                      f"with {len(new_df)} new rows")
            self._record_version(note=f"partial_fit +{len(new_df)} rows")
        else:
            if verbose:
                print(f"[incremental] retraining full tournament on "
                      f"{len(self.history_df)} rows (drift-aware re-selection)")
            self.pipeline.fit(self.history_df, verbose=verbose)
            self._record_version(note=f"full retrain on {len(self.history_df)} rows")

        return self

    def version_history(self) -> pd.DataFrame:
        return pd.DataFrame([v.__dict__ for v in self.versions])

    def has_regressed(self, metric: str = None) -> bool:
        """Quick check: did the most recent update make the primary metric worse?"""
        if len(self.versions) < 2:
            return False
        metric = metric or ("f1_weighted" if self.pipeline.problem_def.problem_type == "classification" else "r2")
        prev, curr = self.versions[-2], self.versions[-1]
        if metric not in prev.accuracy_metrics or metric not in curr.accuracy_metrics:
            return False
        # Higher is better for scores such as F1/R2; lower is better for
        # losses such as MAE/RMSE.
        if metric.lower() in {"mae", "rmse", "mse", "log_loss"}:
            return curr.accuracy_metrics[metric] > prev.accuracy_metrics[metric]
        return curr.accuracy_metrics[metric] < prev.accuracy_metrics[metric]
