"""
Step 5 (part 1): candidate model registry.

Rather than hardcoding "try these 5 algorithms," ModelZoo exposes a
registry you can add to, and tags every model with metadata used by the
relevancy scoring later (interpretability, relative speed) — this is
what lets the selector reason about relevancy, not just accuracy.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List


@dataclass
class ModelSpec:
    name: str
    build_fn: Callable[[], Any]          # returns a fresh, unfitted estimator
    interpretability: float               # 0..1, higher = more interpretable
    relative_speed: float                 # 0..1, higher = faster to train/infer
    supports_partial_fit: bool = False
    requires_dense: bool = False


class ModelZoo:
    def __init__(self, problem_type: str, random_state: int = 42):
        if problem_type not in ("classification", "regression"):
            raise ValueError("problem_type must be 'classification' or 'regression'")
        self.problem_type = problem_type
        self.random_state = random_state
        self._registry: Dict[str, ModelSpec] = {}
        self._load_defaults()

    def register(self, spec: ModelSpec):
        self._registry[spec.name] = spec

    def get(self, name: str) -> ModelSpec:
        return self._registry[name]

    def all(self) -> List[ModelSpec]:
        return list(self._registry.values())

    def _load_defaults(self):
        if self.problem_type == "classification":
            from sklearn.linear_model import LogisticRegression
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.svm import SVC
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.naive_bayes import GaussianNB
            from sklearn.linear_model import SGDClassifier

            self.register(ModelSpec("logistic_regression",
                          lambda: LogisticRegression(
                              max_iter=1000, random_state=self.random_state
                          ), 0.9, 0.9))
            self.register(ModelSpec("decision_tree",
                          lambda: DecisionTreeClassifier(
                              max_depth=8, random_state=self.random_state
                          ), 0.85, 0.9))
            self.register(ModelSpec("random_forest",
                           lambda: RandomForestClassifier(n_estimators=200, random_state=self.random_state), 0.4, 0.5))
            self.register(ModelSpec("gradient_boosting",
                           lambda: GradientBoostingClassifier(random_state=self.random_state), 0.3, 0.3))
            self.register(ModelSpec("knn",
                          lambda: KNeighborsClassifier(), 0.6, 0.6))
            self.register(ModelSpec("svm",
                           lambda: CalibratedClassifierCV(
                               SVC(random_state=self.random_state),
                               ensemble=False,
                           ), 0.3, 0.3))
            self.register(ModelSpec("naive_bayes",
                          lambda: GaussianNB(), 0.7, 0.95, requires_dense=True))
            self.register(ModelSpec("sgd_linear",
                           lambda: SGDClassifier(loss="log_loss", random_state=self.random_state), 0.85, 0.95,
                          supports_partial_fit=True))
        else:
            from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
            from sklearn.tree import DecisionTreeRegressor
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            from sklearn.neighbors import KNeighborsRegressor
            from sklearn.svm import SVR

            self.register(ModelSpec("linear_regression",
                          lambda: LinearRegression(), 0.95, 0.95))
            self.register(ModelSpec("ridge",
                          lambda: Ridge(), 0.9, 0.95))
            self.register(ModelSpec("decision_tree",
                          lambda: DecisionTreeRegressor(
                              max_depth=8, random_state=self.random_state
                          ), 0.85, 0.9))
            self.register(ModelSpec("random_forest",
                           lambda: RandomForestRegressor(n_estimators=200, random_state=self.random_state), 0.4, 0.5))
            self.register(ModelSpec("gradient_boosting",
                           lambda: GradientBoostingRegressor(random_state=self.random_state), 0.3, 0.3))
            self.register(ModelSpec("knn",
                          lambda: KNeighborsRegressor(), 0.6, 0.6))
            self.register(ModelSpec("svr",
                          lambda: SVR(), 0.3, 0.3))
            self.register(ModelSpec("sgd_linear",
                           lambda: SGDRegressor(random_state=self.random_state), 0.85, 0.95,
                          supports_partial_fit=True))
