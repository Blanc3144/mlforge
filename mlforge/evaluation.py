"""
Step 6: Evaluation.

Two distinct things are measured for every trained candidate:
  1. Accuracy-style metrics appropriate to the problem type (and aware
     of class imbalance for classification).
  2. Relevancy sub-scores (interpretability, speed, and any custom
     business scorers), each normalized to 0..1.

This module only *measures*. Combining these into one decision is the
ModelSelector's job (kept separate on purpose, so you can inspect raw
numbers before any weighting is applied).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score,
)

from .preprocessing import Preprocessor
try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:  # older scikit-learn
    def root_mean_squared_error(y_true, y_pred):
        return mean_squared_error(y_true, y_pred) ** 0.5


@dataclass
class CandidateResult:
    name: str
    model: Any
    fit_seconds: float
    accuracy_metrics: Dict[str, float]
    relevancy_scores: Dict[str, float]
    disqualified: bool = False
    disqualified_reason: Optional[str] = None
    error: Optional[str] = None
    validation_metrics: Dict[str, Any] = field(default_factory=dict)
    requires_dense: bool = False
    failure_stage: Optional[str] = None
    warnings: list = field(default_factory=list)
    final_score: Optional[float] = None


def effective_cv_folds(y, requested_cv: int, problem_type: str) -> int:
    """Return the largest usable number of folds for the training labels.

    Classification folds cannot exceed the number of rows in the smallest
    class.  Regression folds are bounded by the number of training rows.  The
    caller still decides what to do when fewer than two folds are possible.
    """
    try:
        requested = int(requested_cv)
    except (TypeError, ValueError):
        return 0
    if requested < 1:
        return 0

    n_rows = len(y)
    if n_rows < 1:
        return 0
    if problem_type == "classification":
        counts = pd.Series(y).value_counts()
        counts = counts[counts > 0]
        if counts.empty:
            return 0
        return min(requested, n_rows, int(counts.min()))
    return min(requested, n_rows)


def _fresh_preprocessor(template: Preprocessor) -> Preprocessor:
    """Create an unfitted preprocessor with the same feature configuration."""
    return Preprocessor(
        numeric_features=(list(template.numeric_features)
                          if template.numeric_features is not None else None),
        categorical_features=(list(template.categorical_features)
                              if template.categorical_features is not None else None),
        scale_numeric=template.scale_numeric,
    )


def _to_dense(X):
    """Convert sparse matrices without changing already-dense inputs."""
    return X.toarray() if hasattr(X, "toarray") else X


class Evaluator:
    def __init__(self, problem_def):
        self.problem_def = problem_def

    def evaluate_classification(self, y_true, y_pred, y_proba=None, classes=None) -> Dict[str, float]:
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        }
        if y_proba is not None:
            try:
                if len(set(y_true)) == 2:
                    metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
                else:
                    # For multiclass problems, predict_proba returns one
                    # column per class.  Passing the estimator's class order
                    # prevents string labels from being paired with the
                    # wrong probability column.
                    metrics["roc_auc"] = roc_auc_score(
                        y_true,
                        y_proba,
                        multi_class="ovr",
                        labels=classes,
                    )
            except ValueError:
                pass
        return metrics

    def evaluate_regression(self, y_true, y_pred) -> Dict[str, float]:
        return {
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": root_mean_squared_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
        }

    def primary_accuracy_score(self, metrics: Dict[str, float]) -> float:
        """Single 0..1 'goodness' number used by the selector's accuracy weight."""
        if self.problem_def.problem_type == "classification":
            return metrics.get("f1_weighted", metrics.get("accuracy", 0.0))
        r2 = metrics.get("r2", 0.0)
        return float(np.clip(r2, 0.0, 1.0))

    def cross_validate(
        self,
        spec,
        train_df,
        target: str,
        preprocessor: Preprocessor,
        requested_cv: int,
        random_state: int = 42,
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """Evaluate one candidate with fold-local preprocessing.

        The returned metrics are intentionally small and explicit.  When CV
        is not possible, the metrics retain the effective fold count and the
        second item is a warning rather than a fabricated score.  Estimator
        errors are allowed to propagate so the pipeline can record them as
        candidate evaluation failures.
        """
        y = train_df[target].reset_index(drop=True)
        raw_features = train_df.drop(columns=[target]).reset_index(drop=True)
        folds = effective_cv_folds(y, requested_cv, self.problem_def.problem_type)
        metrics: Dict[str, Any] = {"cv_folds": folds}
        if folds < 2:
            return metrics, (
                "stability was skipped because fewer than two cross-validation "
                f"folds are possible (effective folds: {folds})."
            )

        if self.problem_def.problem_type == "classification":
            splitter = StratifiedKFold(
                n_splits=folds, shuffle=True, random_state=random_state
            )
            split_iter = splitter.split(raw_features, y)
        else:
            splitter = KFold(n_splits=folds, shuffle=True, random_state=random_state)
            split_iter = splitter.split(raw_features)

        scores = []
        for train_indices, validation_indices in split_iter:
            fold_train = train_df.reset_index(drop=True).iloc[train_indices]
            fold_validation = train_df.reset_index(drop=True).iloc[validation_indices]
            fold_preprocessor = _fresh_preprocessor(preprocessor)
            X_fold_train, y_fold_train = fold_preprocessor.fit_transform(fold_train, target)
            X_fold_validation = fold_preprocessor.transform(
                fold_validation.drop(columns=[target])
            )
            if getattr(spec, "requires_dense", False):
                X_fold_train = _to_dense(X_fold_train)
                X_fold_validation = _to_dense(X_fold_validation)

            model = spec.build_fn()
            model.fit(X_fold_train, y_fold_train)
            predictions = model.predict(X_fold_validation)
            if self.problem_def.problem_type == "classification":
                score = f1_score(
                    fold_validation[target],
                    predictions,
                    average="weighted",
                    zero_division=0,
                )
            else:
                score = r2_score(fold_validation[target], predictions)
            if not np.isfinite(score):
                raise ValueError(
                    "cross-validation produced a non-finite score; "
                    "stability could not be computed"
                )
            scores.append(float(score))

        metrics["cv_mean"] = float(np.mean(scores))
        metrics["cv_std"] = float(np.std(scores))
        return metrics, None

    def relevancy_scores(self, spec, model, X_train, y_train, X_test, y_test,
                          fit_seconds, all_fit_seconds,
                          requires_dense: bool = False) -> Dict[str, float]:
        scores = {
            "interpretability": spec.interpretability,
            "train_speed": self._speed_score(fit_seconds, all_fit_seconds),
            "inference_speed": spec.relative_speed,
        }
        if requires_dense:
            X_train = _to_dense(X_train)
            X_test = _to_dense(X_test)
        for key, fn in self.problem_def.relevancy.custom_scorers.items():
            scores[key] = float(fn(model, X_train, y_train, X_test, y_test, fit_seconds))
        return scores

    @staticmethod
    def _speed_score(fit_seconds: float, all_fit_seconds) -> float:
        if not all_fit_seconds or max(all_fit_seconds) == 0:
            return 1.0
        slowest = max(all_fit_seconds)
        return float(1.0 - (fit_seconds / slowest) * 0.9)  # slowest -> 0.1, instant -> 1.0
