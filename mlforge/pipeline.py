"""
The MLPipeline class orchestrates the full lifecycle end to end:

    problem formulation -> data -> preprocessing -> EDA ->
    train candidates -> evaluate -> select+finalize -> (later) improve

This is the main object most users will interact with directly.
"""
from __future__ import annotations
import time
import warnings as python_warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline as SklearnPipeline

from .problem import ProblemDefinition
from .preprocessing import Preprocessor
from .eda import EDAReport
from .models import ModelZoo
from .evaluation import Evaluator, CandidateResult, effective_cv_folds
from .selector import ModelSelector
from .reporting import format_pipeline_report


class MLPipeline:
    def __init__(self, problem_def: ProblemDefinition, test_size: float = 0.2,
                 random_state: int = 42):
        self.problem_def = problem_def
        self.test_size = test_size
        self.random_state = random_state

        self.preprocessor = Preprocessor()
        self.zoo = ModelZoo(problem_def.problem_type, random_state=random_state)
        self.evaluator = Evaluator(problem_def)
        self.selector = ModelSelector(problem_def)

        self.eda_report_: Optional[dict] = None
        self.candidates_: List[CandidateResult] = []
        self.best_: Optional[CandidateResult] = None
        self.raw_df_: Optional[pd.DataFrame] = None
        self.X_eval_: Optional[object] = None
        self.y_eval_: Optional[pd.Series] = None
        self.train_rows_: Optional[int] = None
        self.test_rows_: Optional[int] = None
        self.train_rows: Optional[int] = None
        self.test_rows: Optional[int] = None
        self.n_train_rows: Optional[int] = None
        self.n_test_rows: Optional[int] = None
        # Explicit aliases make the counts convenient to consume from reports
        # without changing the existing raw-data attributes.
        self.n_train_rows_: Optional[int] = None
        self.n_test_rows_: Optional[int] = None
        self.warnings_: List[str] = []

    # ---- stage: EDA -------------------------------------------------
    def run_eda(self, df: pd.DataFrame, save_plots: bool = False,
                plots_dir: str = "eda_plots") -> dict:
        self.raw_df_ = df
        eda = EDAReport(df, target=self.problem_def.target)
        self.eda_report_ = eda.run()
        if save_plots:
            eda.save_plots(plots_dir)
        return self.eda_report_

    # ---- stage: train + evaluate all candidates ----------------------
    def fit(self, df: pd.DataFrame, model_names: Optional[List[str]] = None,
             verbose: bool = True, tune: bool = False, cv: int = 3,
             tuning_grids: Optional[dict] = None) -> "MLPipeline":
        self.raw_df_ = df
        self.warnings_ = []
        target = self.problem_def.target
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("df must be a non-empty pandas DataFrame")
        if target not in df.columns:
            raise ValueError(f"Target column {target!r} is missing from the dataframe")
        if df[target].isna().any():
            raise ValueError("Target column cannot contain missing values")

        train_df, test_df = train_test_split(
            df, test_size=self.test_size, random_state=self.random_state,
            stratify=df[target] if self.problem_def.problem_type == "classification" else None,
        )
        self.train_rows_ = len(train_df)
        self.test_rows_ = len(test_df)
        self.train_rows = self.train_rows_
        self.test_rows = self.test_rows_
        self.n_train_rows = self.train_rows_
        self.n_test_rows = self.test_rows
        self.n_train_rows_ = self.train_rows_
        self.n_test_rows_ = self.test_rows_

        X_train, y_train = self.preprocessor.fit_transform(train_df, target)
        X_test = self.preprocessor.transform(test_df)
        y_test = test_df[target]
        raw_X_train = train_df.drop(columns=[target])

        specs = self.zoo.all()
        if model_names:
            specs = [s for s in specs if s.name in model_names]

        stability_requested = (
            self.problem_def.relevancy.weights.get("stability", 0) > 0
            and "stability" not in self.problem_def.relevancy.custom_scorers
        )
        effective_stability_cv = None
        if stability_requested:
            effective_stability_cv = effective_cv_folds(
                y_train, cv, self.problem_def.problem_type
            )
            if effective_stability_cv < 2:
                self._record_warning(
                    "stability was skipped because fewer than two cross-validation "
                    f"folds are possible (effective folds: {effective_stability_cv})."
                )

        fit_times: List[float] = []
        results = []
        for spec in specs:
            requires_dense = bool(getattr(spec, "requires_dense", False))
            t0 = time.time()
            fit_seconds = 0.0
            try:
                model = spec.build_fn()
                grid = (tuning_grids or {}).get(spec.name, {})
                if tune and grid:
                    model, _ = self._fit_tuned_candidate(
                        model=model,
                        spec=spec,
                        grid=grid,
                        raw_X_train=raw_X_train,
                        y_train=y_train,
                        requested_cv=cv,
                        requires_dense=requires_dense,
                    )
                else:
                    fit_X_train = self._model_input(X_train, requires_dense)
                    model.fit(fit_X_train, y_train)
                fit_seconds = time.time() - t0
                fit_times.append(fit_seconds)
            except Exception as exc:
                fit_seconds = time.time() - t0
                result = self._failed_candidate(
                    spec,
                    model=None,
                    fit_seconds=fit_seconds,
                    requires_dense=requires_dense,
                    stage="training",
                    exc=exc,
                )
                results.append((spec, result))
                if verbose:
                    print(f"[skipped] {spec.name}: {exc}")
                continue

            prediction_X = self._model_input(X_test, requires_dense)
            try:
                y_pred = model.predict(prediction_X)
                if self.problem_def.problem_type == "classification":
                    y_proba = None
                    classes = None
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(prediction_X)
                        if proba.shape[1] == 2:
                            y_proba = proba[:, 1]
                        else:
                            y_proba = proba
                        classes = getattr(model, "classes_", None)
                    metrics = self.evaluator.evaluate_classification(
                        y_test, y_pred, y_proba, classes=classes
                    )
                else:
                    metrics = self.evaluator.evaluate_regression(y_test, y_pred)
            except Exception as exc:
                result = self._failed_candidate(
                    spec,
                    model=model,
                    fit_seconds=fit_seconds,
                    requires_dense=requires_dense,
                    stage="evaluation",
                    exc=exc,
                )
                results.append((spec, result))
                if verbose:
                    print(f"[failed] {spec.name}: {exc}")
                continue

            result = CandidateResult(
                name=spec.name,
                model=model,
                fit_seconds=fit_seconds,
                accuracy_metrics=metrics,
                relevancy_scores={},
                requires_dense=requires_dense,
            )
            if stability_requested and effective_stability_cv is not None:
                try:
                    validation_metrics, warning = self.evaluator.cross_validate(
                        spec,
                        train_df,
                        target,
                        self.preprocessor,
                        cv,
                        random_state=self.random_state,
                    )
                    result.validation_metrics = validation_metrics
                    if warning:
                        result.warnings.append(warning)
                    if "cv_std" in validation_metrics:
                        result.relevancy_scores["stability"] = float(
                            1.0 - float(np.clip(validation_metrics["cv_std"], 0.0, 1.0))
                        )
                except Exception as exc:
                    result = self._failed_candidate(
                        spec,
                        model=model,
                        fit_seconds=fit_seconds,
                        requires_dense=requires_dense,
                        stage="evaluation",
                        exc=exc,
                        accuracy_metrics=metrics,
                    )
                    results.append((spec, result))
                    if verbose:
                        print(f"[failed] {spec.name}: {exc}")
                    continue

            results.append((spec, result))
            if verbose:
                primary = "f1_weighted" if self.problem_def.problem_type == "classification" else "r2"
                value = metrics.get(primary)
                print(f"  Trained {spec.name:<20} {primary}={value:.3f}  ({fit_seconds:.2f}s)")

        # relevancy scores need the full set of fit times for speed normalization
        for spec, result in results:
            if result.model is None or result.failure_stage is not None:
                continue
            model_X_train = self._model_input(X_train, result.requires_dense)
            model_X_test = self._model_input(X_test, result.requires_dense)
            try:
                result.relevancy_scores.update(self.evaluator.relevancy_scores(
                    spec, result.model, model_X_train, y_train,
                    model_X_test, y_test, result.fit_seconds, fit_times,
                    requires_dense=result.requires_dense,
                ))
                for constraint in self.problem_def.relevancy.hard_constraints:
                    ok = constraint(
                        result.model, model_X_train, y_train, model_X_test, y_test
                    )
                    if not ok:
                        result.disqualified = True
                        result.disqualified_reason = getattr(
                            constraint, "__name__", "hard_constraint_failed"
                        )
                        break
            except Exception as exc:
                result.disqualified = True
                result.disqualified_reason = "evaluation_failed"
                result.failure_stage = "evaluation"
                result.error = f"{type(exc).__name__}: {exc}"
                if verbose:
                    print(f"[failed] {spec.name}: {exc}")

        self.candidates_ = [r for _, r in results]
        self.best_ = self.selector.select(self.candidates_)
        self.X_eval_ = self._model_input(X_test, self.best_.requires_dense)
        self.y_eval_ = y_test
        if verbose:
            score = (
                f"{self.best_.final_score:.3f}"
                if self.best_.final_score is not None
                else "unavailable"
            )
            print(f"  Selected {self.best_.name} (overall score {score})")
        return self

    @staticmethod
    def _model_input(X, requires_dense: bool):
        if requires_dense and hasattr(X, "toarray"):
            return X.toarray()
        return X

    def _record_warning(self, message: str) -> None:
        self.warnings_.append(message)
        python_warnings.warn(message, UserWarning, stacklevel=2)

    @staticmethod
    def _failed_candidate(
        spec,
        model,
        fit_seconds: float,
        requires_dense: bool,
        stage: str,
        exc: Exception,
        accuracy_metrics: Optional[Dict[str, float]] = None,
    ) -> CandidateResult:
        return CandidateResult(
            name=spec.name,
            model=model,
            fit_seconds=fit_seconds,
            accuracy_metrics=accuracy_metrics or {},
            relevancy_scores={},
            disqualified=True,
            disqualified_reason=f"{stage}_failed",
            error=f"{type(exc).__name__}: {exc}",
            requires_dense=requires_dense,
            failure_stage=stage,
        )

    def _fit_tuned_candidate(
        self,
        model,
        spec,
        grid,
        raw_X_train,
        y_train,
        requested_cv: int,
        requires_dense: bool,
    ):
        """Tune an estimator while fitting preprocessing independently per fold."""
        effective_folds = effective_cv_folds(
            y_train, requested_cv, self.problem_def.problem_type
        )
        if effective_folds < 2:
            raise ValueError(
                "Tuning requires at least two usable cross-validation folds "
                f"(effective folds: {effective_folds})"
            )

        transformer = self.preprocessor.build_transformer()
        if requires_dense:
            # ColumnTransformer otherwise may return sparse output for mixed
            # numeric/categorical data.  Make the declared dense requirement
            # part of the search estimator so every fold uses the same input.
            transformer.sparse_threshold = 0
        search = GridSearchCV(
            SklearnPipeline([
                ("preprocessor", transformer),
                ("model", model),
            ]),
            self._qualify_tuning_grid(grid),
            scoring=(
                "f1_weighted"
                if self.problem_def.problem_type == "classification"
                else "r2"
            ),
            cv=effective_folds,
            n_jobs=-1,
            error_score="raise",
        )
        search.fit(raw_X_train, y_train)
        fitted_pipeline = search.best_estimator_
        return fitted_pipeline.named_steps["model"], effective_folds

    @staticmethod
    def _qualify_tuning_grid(grid):
        """Adapt the historical estimator grid API to the wrapper pipeline."""
        if isinstance(grid, dict):
            return {
                key if key.startswith(("model__", "preprocessor__")) else f"model__{key}": value
                for key, value in grid.items()
            }
        if isinstance(grid, list):
            return [MLPipeline._qualify_tuning_grid(item) for item in grid]
        return grid

    def leaderboard(self):
        return self.selector.leaderboard(self.candidates_)

    def predict(self, df: pd.DataFrame):
        if self.best_ is None:
            raise RuntimeError("Call .fit() before .predict().")
        X = self.preprocessor.transform(df)
        return self.best_.model.predict(self._model_input(X, self.best_.requires_dense))

    def summary(self) -> str:
        lines = [self.problem_def.summary()]
        if self.eda_report_:
            lines.append(f"\nRows x Cols: {self.eda_report_['shape']}")
        if self.best_:
            lines.append(f"\nFinalized model: {self.best_.name}")
            lines.append(f"Accuracy metrics: {self.best_.accuracy_metrics}")
            lines.append(f"Relevancy scores: {self.best_.relevancy_scores}")
            weights = self.problem_def.relevancy.normalized_weights()
            contributions = {}
            for criterion, weight in weights.items():
                if criterion == "accuracy":
                    value = self.evaluator.primary_accuracy_score(self.best_.accuracy_metrics)
                else:
                    if criterion not in self.best_.relevancy_scores:
                        contributions[criterion] = "unavailable"
                        continue
                    value = self.best_.relevancy_scores[criterion]
                contributions[criterion] = round(weight * value, 4)
            lines.append(f"Weighted contributions: {contributions}")
            lines.append("Selection reason: highest eligible weighted score across the configured criteria.")
        return "\n".join(lines)

    def report(self) -> str:
        """Return a beginner-friendly report for the latest training run."""
        return format_pipeline_report(self)

    def print_report(self) -> None:
        """Print the beginner-friendly report for the latest training run."""
        print(self.report())
