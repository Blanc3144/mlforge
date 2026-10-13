"""
Step 1 of the lifecycle: Problem Formulation.

This is the piece most AutoML tools skip. Before touching data, we force
an explicit statement of *what* is being solved and *why a model choice
would be considered right*, not just accurate.
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

VALID_PROBLEM_TYPES = ("classification", "regression")


@dataclass
class RelevancyCriteria:
    """
    Defines what makes a model 'relevant' to the business/problem context,
    independent of raw predictive accuracy.

    weights: relative importance of each criterion, need not sum to 1
             (they're normalized internally). Recognized keys you can use
             out of the box: 'accuracy', 'interpretability', 'train_speed',
             'inference_speed', 'stability'. You can add any custom key as
             long as you also provide a scorer for it in `custom_scorers`.

    custom_scorers: dict mapping a criterion name -> function
             (model, X_train, y_train, X_test, y_test, fit_seconds) -> float in [0, 1]
             Use this to encode real business relevancy, e.g. "must be
             explainable to a regulator", "must run in <50ms", "must not
             use protected attributes", etc.

    hard_constraints: functions that return True/False. Any candidate model
             that fails a hard constraint is disqualified outright, no
             matter how accurate it is. This is the main lever for
             "relevant, not just accurate."
    """
    weights: Dict[str, float] = field(default_factory=lambda: {
        "accuracy": 0.6,
        "interpretability": 0.2,
        "train_speed": 0.1,
        "inference_speed": 0.1,
    })
    custom_scorers: Dict[str, Callable] = field(default_factory=dict)
    hard_constraints: List[Callable] = field(default_factory=list)

    def normalized_weights(self) -> Dict[str, float]:
        if not self.weights:
            raise ValueError("At least one relevancy weight is required")
        if any(not isinstance(v, (int, float)) or v < 0 for v in self.weights.values()):
            raise ValueError("Relevancy weights must be finite, non-negative numbers")
        total = sum(self.weights.values())
        if total <= 0:
            raise ValueError("At least one relevancy weight must be greater than zero")
        return {k: v / total for k, v in self.weights.items()}


@dataclass
class ProblemDefinition:
    """
    The explicit contract for what we're solving.

    problem_type: 'classification' or 'regression'
    target: name of the target column
    objective: a short, plain-English statement of the business objective.
               Not used computationally — it's forced documentation, so
               that six months from now someone (including you) can tell
               whether the model still matches the intent.
    positive_class: for binary classification, which label counts as
               'positive' for precision/recall framing (optional).
    relevancy: a RelevancyCriteria instance. Defaults to a sane generic one.
    """
    problem_type: str
    target: str
    objective: str = ""
    positive_class: Optional[object] = None
    relevancy: RelevancyCriteria = field(default_factory=RelevancyCriteria)

    def __post_init__(self):
        if self.problem_type not in VALID_PROBLEM_TYPES:
            raise ValueError(
                f"problem_type must be one of {VALID_PROBLEM_TYPES}, got {self.problem_type!r}"
            )
        if not self.target or not isinstance(self.target, str):
            raise ValueError("target must be a non-empty column name")
        self.relevancy.normalized_weights()

    def summary(self) -> str:
        lines = [
            f"Problem type : {self.problem_type}",
            f"Target column: {self.target}",
            f"Objective    : {self.objective or '(not stated)'}",
        ]
        w = self.relevancy.normalized_weights()
        lines.append("Relevancy weights:")
        for k, v in w.items():
            lines.append(f"  - {k}: {v:.2f}")
        if self.relevancy.hard_constraints:
            lines.append(f"Hard constraints: {len(self.relevancy.hard_constraints)} defined")
        return "\n".join(lines)
