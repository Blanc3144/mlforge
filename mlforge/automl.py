"""One-call, beginner-friendly AutoML workflows.

``AutoML.run`` is deliberately a thin orchestration layer over the existing
supervised lifecycle.  It adds the pieces that are awkward to express in a
manual first run: target/task inference, useful validation and a small
unsupervised fallback for data without a reliable target column.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
import re
from numbers import Integral, Real
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .eda import EDAReport
from .pipeline import MLPipeline
from .preprocessing import Preprocessor
from .problem import ProblemDefinition, RelevancyCriteria
from .reporting import format_automl_report


_VALID_TASKS = {"auto", "classification", "regression", "clustering"}
_SUPERVISED_TASKS = {"classification", "regression"}
_BUILT_IN_RELEVANCY_KEYS = {
    "accuracy",
    "interpretability",
    "train_speed",
    "inference_speed",
    "stability",
}
@dataclass(frozen=True)
class _TargetGuess:
    column: Optional[str]
    confidence: float
    candidates: Tuple[str, ...] = ()


def _normalise_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def _name_confidence(name: str) -> float:
    normalised = _normalise_column_name(name)
    exact = {
        "target": 0.99,
        "label": 0.97,
        "class": 0.95,
        "outcome": 0.95,
        "response": 0.93,
        "species": 0.95,
        "category": 0.90,
        "y": 0.92,
        "prediction": 0.90,
        "predicted": 0.90,
        "result": 0.88,
    }
    if normalised in exact:
        return exact[normalised]
    if any(
        normalised.startswith(prefix + "_") or normalised.endswith("_" + prefix)
        for prefix in ("target", "label", "outcome", "response", "prediction")
    ):
        return 0.84
    return 0.0


def _low_cardinality_limit(n_rows: int) -> int:
    # A small class count is a useful classification signal, but the limit
    # grows slowly enough that ordinary continuous regression targets are not
    # accidentally treated as labels on realistic dataframes.
    return max(2, min(20, int(max(2, n_rows * 0.05))))


def _infer_target(df: pd.DataFrame) -> _TargetGuess:
    """Return the strongest target-column guess, if one is defensible."""
    n_rows = len(df)
    numeric_columns = sum(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns)
    scored: List[Tuple[float, str]] = []

    for position, column in enumerate(df.columns):
        # ProblemDefinition intentionally uses string target names.  Numeric
        # feature names remain valid for clustering, but cannot be inferred as
        # a supervised target through this public convenience API.
        if not isinstance(column, str) or not column.strip():
            continue
        series = df[column]
        non_missing = series.dropna()
        unique = int(non_missing.nunique())
        if unique <= 1:
            continue

        score = _name_confidence(column)
        # A two-column numeric frame named ``x``/``y`` is often a set of
        # unlabeled coordinates (a particularly common clustering input).
        # Treat a continuous ``y`` there as ambiguous instead of silently
        # removing it from the clustering features.  An explicit target=...
        # still always overrides this heuristic.
        if (
            score >= 0.9
            and _normalise_column_name(column) == "y"
            and len(df.columns) <= 2
            and pd.api.types.is_numeric_dtype(series)
            and unique > _low_cardinality_limit(n_rows)
        ):
            score = 0.0
        if score == 0.0:
            low_cardinality = unique <= _low_cardinality_limit(n_rows)
            if not pd.api.types.is_numeric_dtype(series) and low_cardinality and numeric_columns >= max(1, len(df.columns) - 1):
                # One categorical column alongside numeric measurements is a
                # common shape for a labelled table, but still deserves a
                # warning rather than high confidence.
                score = 0.64
            elif pd.api.types.is_numeric_dtype(series) and low_cardinality:
                # Low-cardinality numeric labels are common too.  Give the
                # final column a small nudge because labelled tables often
                # put the outcome there, but score all such columns so two
                # plausible labels are reported as ambiguous below.
                score = 0.56 if position == len(df.columns) - 1 else 0.52

        if score:
            # A completely missing target is never a useful automatic choice;
            # retain the candidate so the caller can report a precise error.
            if series.isna().any():
                score -= 0.08
            scored.append((score, column))

    if not scored:
        return _TargetGuess(None, 0.0)

    scored.sort(key=lambda item: item[0], reverse=True)
    top_score, top_column = scored[0]
    candidates = tuple(column for _, column in scored)

    # Weak, similarly ranked guesses are more harmful than a useful
    # clustering result.  A strong conventional name wins decisively.
    if top_score < 0.5:
        return _TargetGuess(None, 0.0, candidates)
    if len(scored) > 1 and top_score < 0.85 and top_score - scored[1][0] <= 0.08:
        return _TargetGuess(None, 0.0, candidates)
    return _TargetGuess(top_column, float(np.clip(top_score, 0.0, 1.0)), candidates)


def _infer_task(series: pd.Series) -> Tuple[str, float]:
    if pd.api.types.is_bool_dtype(series) or not pd.api.types.is_numeric_dtype(series):
        return "classification", 0.90

    unique = int(series.nunique())
    if unique <= _low_cardinality_limit(len(series)):
        confidence = 0.90 if unique <= 10 else 0.70
        return "classification", confidence
    return "regression", 0.88


def _validate_dataframe(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if df.empty:
        raise ValueError("df must be a non-empty pandas DataFrame")
    if len(df.columns) == 0:
        raise ValueError("df must contain at least one column")
    if not df.columns.is_unique:
        raise ValueError("df must have unique column names")


def _validate_target(df: pd.DataFrame, target: str) -> None:
    if not isinstance(target, str) or not target.strip():
        raise ValueError("target must be a non-empty column name or None")
    if target not in df.columns:
        raise ValueError(f"Target column {target!r} is missing from the dataframe")
    if df[target].isna().any():
        raise ValueError(f"Target column {target!r} cannot contain missing values")
    if df[target].nunique() < 2:
        raise ValueError(f"Target column {target!r} must contain at least two distinct values")
    if len(df.columns) < 2:
        raise ValueError("A supervised task requires at least one feature column besides the target")
    feature_columns = [column for column in df.columns if column != target]
    if all(df[column].isna().all() for column in feature_columns):
        raise ValueError("At least one feature column must contain a non-missing value")


def _validate_relevancy(criteria: RelevancyCriteria) -> None:
    if not isinstance(criteria, RelevancyCriteria):
        raise TypeError("relevancy must be a RelevancyCriteria or a mapping of settings")
    if not isinstance(criteria.weights, Mapping):
        raise ValueError("Relevancy weights must be supplied as a mapping")
    if not criteria.weights:
        raise ValueError("At least one relevancy weight is required")
    if not isinstance(criteria.custom_scorers, Mapping):
        raise ValueError("custom_scorers must be a mapping")
    if any(not callable(fn) for fn in criteria.custom_scorers.values()):
        raise ValueError("Every custom relevancy scorer must be callable")
    for key, value in criteria.weights.items():
        if not isinstance(key, str) or not key:
            raise ValueError("Relevancy criterion names must be non-empty strings")
        if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(float(value)) or value < 0:
            raise ValueError("Relevancy weights must be finite, non-negative numbers")
        if key not in _BUILT_IN_RELEVANCY_KEYS and key not in criteria.custom_scorers:
            raise ValueError(
                f"Relevancy criterion {key!r} needs a matching custom_scorers entry"
            )
    criteria.normalized_weights()
    if not isinstance(criteria.hard_constraints, Sequence):
        raise ValueError("hard_constraints must be a sequence of callables")
    if any(not callable(fn) for fn in criteria.hard_constraints):
        raise ValueError("Every hard constraint must be callable")


def _make_relevancy(
    relevancy: Optional[Any],
    relevancy_weights: Optional[Mapping[str, float]],
    custom_scorers: Optional[Mapping[str, Callable]],
    hard_constraints: Optional[Sequence[Callable]],
    relevancy_settings: Optional[Mapping[str, Any]],
) -> RelevancyCriteria:
    if relevancy_weights is not None and not isinstance(relevancy_weights, Mapping):
        raise TypeError("relevancy_weights must be a mapping")
    if custom_scorers is not None and not isinstance(custom_scorers, Mapping):
        raise TypeError("custom_scorers must be a mapping")
    if hard_constraints is not None and (
        isinstance(hard_constraints, (str, bytes)) or not isinstance(hard_constraints, Sequence)
    ):
        raise TypeError("hard_constraints must be a sequence of callables")
    if relevancy_settings is not None and not isinstance(relevancy_settings, Mapping):
        raise TypeError("relevancy_settings must be a mapping")

    supplied_direct = any(value is not None for value in (relevancy_weights, custom_scorers, hard_constraints))
    if relevancy_settings is not None:
        if relevancy is not None or supplied_direct:
            raise ValueError("Use either relevancy/relevancy settings or direct relevancy arguments, not both")
        relevancy = relevancy_settings

    if relevancy is None:
        criteria = RelevancyCriteria(
            weights=dict(relevancy_weights) if relevancy_weights is not None else RelevancyCriteria().weights,
            custom_scorers=dict(custom_scorers or {}),
            hard_constraints=list(hard_constraints or []),
        )
    elif isinstance(relevancy, RelevancyCriteria):
        if supplied_direct:
            criteria = RelevancyCriteria(
                weights=dict(relevancy_weights) if relevancy_weights is not None else dict(relevancy.weights),
                custom_scorers=dict(custom_scorers) if custom_scorers is not None else dict(relevancy.custom_scorers),
                hard_constraints=list(hard_constraints) if hard_constraints is not None else list(relevancy.hard_constraints),
            )
        else:
            criteria = relevancy
    elif isinstance(relevancy, Mapping):
        if supplied_direct:
            raise ValueError("Use either relevancy mapping settings or direct relevancy arguments, not both")
        settings = dict(relevancy)
        known_settings = {"weights", "custom_scorers", "hard_constraints"}
        if known_settings.intersection(settings):
            unknown = set(settings) - known_settings
            if unknown:
                raise ValueError(f"Unknown relevancy setting(s): {sorted(unknown)}")
            criteria = RelevancyCriteria(
                weights=dict(settings.get("weights") or RelevancyCriteria().weights),
                custom_scorers=dict(settings.get("custom_scorers") or {}),
                hard_constraints=list(settings.get("hard_constraints") or []),
            )
        else:
            criteria = RelevancyCriteria(weights=settings)
    else:
        raise TypeError("relevancy must be a RelevancyCriteria or a mapping of settings")

    _validate_relevancy(criteria)
    return criteria


class ClusteringPipeline:
    """Small, inspectable unsupervised workflow used by ``AutoML``."""

    def __init__(self, random_state: int = 42, n_clusters: Optional[int] = None, max_clusters: int = 8):
        self.random_state = random_state
        self.n_clusters = n_clusters
        self.max_clusters = max_clusters
        self.preprocessor = Preprocessor()
        self.model = None
        self.model_ = None
        self.raw_df_: Optional[pd.DataFrame] = None
        self.X_: Any = None
        self.labels_: Optional[np.ndarray] = None
        self.candidates_: List[dict] = []

    def fit(self, df: pd.DataFrame) -> "ClusteringPipeline":
        self.raw_df_ = df.copy()
        X = self.preprocessor.fit_transform_features(df)
        if X.shape[1] == 0:
            raise ValueError("Clustering requires at least one usable feature")
        if len(df) < 2:
            raise ValueError("Clustering requires at least two rows")
        if hasattr(X, "getnnz") and hasattr(X, "data"):
            # Keep sparse one-hot data sparse; converting a high-cardinality
            # column to a dense array would make the convenience API needlessly
            # memory-hungry.
            if not np.isfinite(np.asarray(X.data)).all():
                raise ValueError("Clustering features must be finite after preprocessing")
            column_ranges = X.max(axis=0) - X.min(axis=0)
            if hasattr(column_ranges, "toarray"):
                column_ranges = column_ranges.toarray()
            column_ranges = np.asarray(column_ranges).ravel()
            if np.all(column_ranges <= 1e-12):
                raise ValueError("Clustering requires at least one non-constant feature")
        else:
            values = np.asarray(X)
            if not np.isfinite(values).all():
                raise ValueError("Clustering features must be finite after preprocessing")
            if np.all(np.nanstd(values, axis=0) <= 1e-12):
                raise ValueError("Clustering requires at least one non-constant feature")

        if self.n_clusters is not None:
            cluster_counts = [self.n_clusters]
        else:
            upper = min(self.max_clusters, len(df) - 1)
            cluster_counts = list(range(2, upper + 1)) if upper >= 2 else [2]

        candidates: List[dict] = []
        for count in cluster_counts:
            model = KMeans(n_clusters=count, random_state=self.random_state, n_init=10)
            model.fit(X)
            labels = model.labels_
            score: Optional[float] = None
            if len(np.unique(labels)) > 1 and len(np.unique(labels)) < len(df):
                try:
                    score = float(silhouette_score(X, labels))
                except (TypeError, ValueError):
                    score = None
            candidates.append({
                "name": f"kmeans_{count}",
                "n_clusters": count,
                "silhouette": score,
                "inertia": float(model.inertia_),
                "disqualified": False,
                "reason": None,
                "model": model,
            })

        scored = [row for row in candidates if row["silhouette"] is not None]
        best = max(scored, key=lambda row: row["silhouette"]) if scored else candidates[0]
        self.model = best["model"]
        self.model_ = self.model
        self.X_ = X
        self.labels_ = self.model.labels_
        self.candidates_ = candidates
        return self

    def predict(self, df: pd.DataFrame):
        if self.model is None:
            raise RuntimeError("Call .fit() before .predict().")
        X = self.preprocessor.transform_features(df)
        return self.model.predict(X)

    def leaderboard(self) -> List[dict]:
        rows = []
        for candidate in self.candidates_:
            rows.append({key: value for key, value in candidate.items() if key != "model"})
        rows.sort(
            key=lambda row: (
                row["silhouette"] is None,
                -(row["silhouette"] if row["silhouette"] is not None else float("-inf")),
            )
        )
        return rows


@dataclass
class AutoMLResult:
    """The stable, user-facing result of :meth:`AutoML.run`."""

    task: str
    target: Optional[str]
    model: Any
    pipeline: Any
    eda: dict
    warnings: List[str] = field(default_factory=list)
    explanation: str = ""
    target_confidence: Optional[float] = None
    task_confidence: Optional[float] = None

    @property
    def model_(self):
        """Scikit-learn-style alias for callers that prefer fitted attributes."""
        return self.model

    @property
    def best_model(self):
        return self.model

    @property
    def eda_report(self):
        return self.eda

    @property
    def problem_def(self):
        """Expose the supervised problem contract when one exists."""
        return getattr(self.pipeline, "problem_def", None)

    def predict(self, new_data: pd.DataFrame):
        if self.model is None or self.pipeline is None:
            raise RuntimeError("This AutoML result does not contain a fitted model")
        return self.pipeline.predict(new_data)

    def report(self) -> str:
        return format_automl_report(self)

    def print_report(self) -> None:
        """Print the formatted report for interactive use."""
        print(self.report())

    def leaderboard(self) -> List[dict]:
        return self.pipeline.leaderboard()

    def explain(self) -> str:
        """Return the same plain-English explanation shown in ``report``."""
        return self.explanation


class AutoML:
    """Automatic one-call workflow for tabular data."""

    @classmethod
    def run(
        cls,
        df: pd.DataFrame,
        target: Optional[str] = None,
        task: str = "auto",
        relevancy: Optional[Any] = None,
        *,
        relevancy_weights: Optional[Mapping[str, float]] = None,
        custom_scorers: Optional[Mapping[str, Callable]] = None,
        hard_constraints: Optional[Sequence[Callable]] = None,
        relevancy_settings: Optional[Mapping[str, Any]] = None,
        objective: str = "",
        positive_class: Optional[object] = None,
        test_size: float = 0.2,
        random_state: int = 42,
        model_names: Optional[Sequence[str]] = None,
        tune: bool = False,
        cv: int = 3,
        tuning_grids: Optional[dict] = None,
        n_clusters: Optional[int] = None,
        max_clusters: int = 8,
        verbose: bool = False,
    ) -> AutoMLResult:
        """Run a complete supervised or unsupervised workflow.

        With ``task="auto"``, a named or structurally obvious target is
        inferred and its dtype/cardinality determines classification versus
        regression.  When no target is defensible, the method returns a
        KMeans clustering result instead of failing because a label is absent.
        Pass ``target=...`` whenever the inference warning is not appropriate
        for the domain.
        """
        _validate_dataframe(df)
        normalised_task = cls._normalise_task(task)
        if target is not None and (not isinstance(target, str) or not target.strip()):
            raise ValueError("target must be a non-empty column name or None")
        if target is not None and target not in df.columns:
            raise ValueError(f"Target column {target!r} is missing from the dataframe")
        if normalised_task == "clustering" and target is not None:
            raise ValueError("target cannot be supplied when task='clustering'; clustering is unsupervised")

        criteria = _make_relevancy(
            relevancy,
            relevancy_weights,
            custom_scorers,
            hard_constraints,
            relevancy_settings,
        )
        cls._validate_common_options(
            random_state=random_state,
            model_names=model_names,
            tune=tune,
            cv=cv,
            n_clusters=n_clusters,
            max_clusters=max_clusters,
        )
        if not isinstance(verbose, bool):
            raise ValueError("verbose must be a boolean")

        warnings: List[str] = []
        target_confidence: Optional[float] = 1.0 if target is not None else None
        task_confidence: Optional[float] = None

        if normalised_task == "clustering":
            return cls._run_clustering(
                df,
                warnings=["Clustering was requested explicitly; no target column is used."],
                random_state=random_state,
                n_clusters=n_clusters,
                max_clusters=max_clusters,
            )

        if target is None:
            guess = _infer_target(df)
            if guess.column is None:
                if normalised_task == "auto":
                    candidate_note = ""
                    if guess.candidates:
                        candidate_note = f" Possible candidates were: {', '.join(guess.candidates)}."
                    warnings.append(
                        "No reliable target column was detected, so AutoML ran clustering instead."
                        + candidate_note
                        + " Pass target='column_name' to run a supervised task."
                    )
                    return cls._run_clustering(
                        df,
                        warnings=warnings,
                        random_state=random_state,
                        n_clusters=n_clusters,
                        max_clusters=max_clusters,
                    )
                raise ValueError(
                    f"target is required for task={normalised_task!r}; no reliable target column was inferred"
                )
            target = guess.column
            target_confidence = guess.confidence
            confidence_label = cls._confidence_label(guess.confidence)
            warnings.append(
                f"Inferred target {target!r} with {confidence_label} confidence ({guess.confidence:.2f}). "
                "Pass target='column_name' to override this choice."
            )

        # Explicit targets and inferred targets share the same clear checks.
        assert target is not None
        _validate_target(df, target)
        if normalised_task == "auto":
            normalised_task, task_confidence = _infer_task(df[target])
            warnings.append(
                f"Inferred task '{normalised_task}' from target dtype/cardinality "
                f"({cls._confidence_label(task_confidence)} confidence, {task_confidence:.2f})."
            )
        else:
            task_confidence = 1.0

        if normalised_task not in _SUPERVISED_TASKS:
            raise ValueError(f"Unsupported supervised task {normalised_task!r}")

        _validate_split_for_target(df, target, normalised_task, test_size)
        problem = ProblemDefinition(
            problem_type=normalised_task,
            target=target,
            objective=objective,
            positive_class=positive_class,
            relevancy=criteria,
        )
        pipeline = MLPipeline(problem, test_size=float(test_size), random_state=int(random_state))
        cls._validate_model_names(pipeline, model_names)
        pipeline.run_eda(df)
        pipeline.fit(
            df,
            model_names=list(model_names) if model_names is not None else None,
            verbose=verbose,
            tune=tune,
            cv=cv,
            tuning_grids=tuning_grids,
        )
        warnings.extend(pipeline.warnings_)
        model = pipeline.best_.model if pipeline.best_ is not None else None
        explanation = cls._supervised_explanation(normalised_task, target, pipeline)
        return AutoMLResult(
            task=normalised_task,
            target=target,
            model=model,
            pipeline=pipeline,
            eda=pipeline.eda_report_ or {},
            warnings=warnings,
            explanation=explanation,
            target_confidence=target_confidence,
            task_confidence=task_confidence,
        )

    @staticmethod
    def _normalise_task(task: str) -> str:
        if not isinstance(task, str):
            raise ValueError(f"task must be one of {sorted(_VALID_TASKS)}")
        normalised = task.strip().lower()
        if normalised not in _VALID_TASKS:
            raise ValueError(f"task must be one of {sorted(_VALID_TASKS)}, got {task!r}")
        return normalised

    @staticmethod
    def _confidence_label(confidence: float) -> str:
        if confidence >= 0.85:
            return "high"
        if confidence >= 0.65:
            return "medium"
        return "low"

    @staticmethod
    def _validate_common_options(
        *,
        random_state: int,
        model_names: Optional[Sequence[str]],
        tune: bool,
        cv: int,
        n_clusters: Optional[int],
        max_clusters: int,
    ) -> None:
        if not isinstance(random_state, Integral) or isinstance(random_state, bool):
            raise ValueError("random_state must be an integer")
        if model_names is not None:
            if isinstance(model_names, (str, bytes)) or not isinstance(model_names, Sequence):
                raise ValueError("model_names must be a sequence of model names")
            if not model_names or any(not isinstance(name, str) or not name for name in model_names):
                raise ValueError("model_names must contain at least one non-empty string")
        if not isinstance(tune, bool):
            raise ValueError("tune must be a boolean")
        if not isinstance(cv, Integral) or isinstance(cv, bool) or cv < 2:
            raise ValueError("cv must be an integer greater than or equal to 2")
        if n_clusters is not None and (
            not isinstance(n_clusters, Integral) or isinstance(n_clusters, bool) or n_clusters < 2
        ):
            raise ValueError("n_clusters must be an integer greater than or equal to 2")
        if not isinstance(max_clusters, Integral) or isinstance(max_clusters, bool) or max_clusters < 2:
            raise ValueError("max_clusters must be an integer greater than or equal to 2")

    @staticmethod
    def _validate_model_names(pipeline: MLPipeline, model_names: Optional[Sequence[str]]) -> None:
        if model_names is None:
            return
        known = {spec.name for spec in pipeline.zoo.all()}
        unknown = sorted(set(model_names) - known)
        if unknown:
            raise ValueError(f"Unknown model name(s) for {pipeline.problem_def.problem_type}: {unknown}")

    @staticmethod
    def _run_clustering(
        df: pd.DataFrame,
        *,
        warnings: List[str],
        random_state: int,
        n_clusters: Optional[int],
        max_clusters: int,
    ) -> AutoMLResult:
        if n_clusters is not None and n_clusters > len(df):
            raise ValueError("n_clusters cannot be greater than the number of rows")
        if len(df) < 2:
            raise ValueError("Clustering requires at least two rows")
        eda = EDAReport(df, target=None).run()
        pipeline = ClusteringPipeline(
            random_state=int(random_state),
            n_clusters=int(n_clusters) if n_clusters is not None else None,
            max_clusters=int(max_clusters),
        )
        pipeline.fit(df)
        best = pipeline.leaderboard()[0]
        eda["clustering"] = {
            "method": "kmeans",
            "n_clusters": best["n_clusters"],
            "cluster_sizes": {
                int(label): int(count)
                for label, count in zip(*np.unique(pipeline.labels_, return_counts=True))
            },
            "silhouette": best["silhouette"],
        }
        explanation = (
            "No labeled outcome was used. AutoML grouped the rows with KMeans "
            f"into {best['n_clusters']} clusters and selected that size using "
            "the silhouette score when it was available. "
            "Use result.predict(new_data) to assign new rows to a cluster."
        )
        return AutoMLResult(
            task="clustering",
            target=None,
            model=pipeline.model,
            pipeline=pipeline,
            eda=eda,
            warnings=warnings,
            explanation=explanation,
            target_confidence=None,
            task_confidence=1.0,
        )

    @staticmethod
    def _supervised_explanation(task: str, target: str, pipeline: MLPipeline) -> str:
        model_name = pipeline.best_.name if pipeline.best_ is not None else "the selected model"
        action = "classify" if task == "classification" else "predict a numeric value"
        return (
            f"AutoML trained several candidate models to {action} using {target!r}. "
            f"The recommended model is {model_name!r}, selected by balancing holdout "
            "quality with the relevancy settings. Use result.predict(new_data) for new rows."
        )


def _validate_split_for_target(df: pd.DataFrame, target: str, task: str, test_size: float) -> None:
    if not isinstance(test_size, Real) or isinstance(test_size, bool) or not 0 < float(test_size) < 1:
        raise ValueError("test_size must be a number strictly between 0 and 1")
    if len(df) < 4:
        raise ValueError("A supervised task requires at least four rows")
    n_test = int(math.ceil(len(df) * float(test_size)))
    n_train = len(df) - n_test
    if n_test < 2 or n_train < 2:
        raise ValueError("test_size must leave at least two rows in both train and test splits")
    if task != "classification":
        return
    n_classes = int(df[target].nunique())
    if n_test < n_classes or n_train < n_classes:
        raise ValueError(
            "test_size leaves too few rows per split for all target classes; "
            "increase the dataframe size or adjust test_size"
        )
    counts = df[target].value_counts()
    if (counts < 2).any():
        raise ValueError("Each classification target class must contain at least two rows")
