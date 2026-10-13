"""
Step 7: Model selection — the core "not just accuracy" logic.

Every candidate gets:
  final_score = sum(weight[c] * normalized_score[c] for c in criteria)

with 'accuracy' itself treated as just one more weighted criterion
(default weight 0.6, fully overridable via ProblemDefinition.relevancy).
Candidates that fail a hard constraint are disqualified regardless of
score — this is what prevents "technically most accurate but useless
for the actual problem" from winning.
"""
from __future__ import annotations
from typing import List

from .evaluation import CandidateResult


class ModelSelector:
    def __init__(self, problem_def):
        self.problem_def = problem_def

    def select(self, candidates: List[CandidateResult]) -> CandidateResult:
        weights = self.problem_def.relevancy.normalized_weights()

        eligible = [c for c in candidates if not c.disqualified]
        if not eligible:
            if not candidates:
                raise RuntimeError("No candidate models were evaluated.")

            training_failures = sum(c.failure_stage == "training" for c in candidates)
            evaluation_failures = sum(c.failure_stage == "evaluation" for c in candidates)
            rule_rejections = sum(
                c.disqualified and c.failure_stage is None for c in candidates
            )
            reasons = []
            if training_failures:
                reasons.append(f"{training_failures} failed during training")
            if evaluation_failures:
                reasons.append(f"{evaluation_failures} failed during evaluation")
            if rule_rejections:
                reasons.append(
                    f"{rule_rejections} were disqualified by hard constraints"
                )
            if not reasons:
                reasons.append("all candidates were disqualified")
            raise RuntimeError(
                "No eligible candidate models: " + "; ".join(reasons) + "."
            )

        for c in eligible:
            weighted_score = 0.0
            available_weight = 0.0
            for criterion, weight in weights.items():
                if criterion == "accuracy":
                    from .evaluation import Evaluator
                    ev = Evaluator(self.problem_def)
                    value = ev.primary_accuracy_score(c.accuracy_metrics)
                else:
                    if criterion not in c.relevancy_scores:
                        # A skipped validation metric is intentionally absent;
                        # do not turn missing evidence into a fabricated zero.
                        continue
                    value = c.relevancy_scores[criterion]
                weighted_score += weight * value
                available_weight += weight
            c.final_score = (
                weighted_score / available_weight if available_weight else None
            )

        eligible.sort(
            key=lambda c: (
                c.final_score is None,
                -(c.final_score or 0),
            )
        )
        return eligible[0]

    def leaderboard(self, candidates: List[CandidateResult]) -> List[dict]:
        rows = []
        for c in candidates:
            row = {
                "name": c.name,
                "disqualified": c.disqualified,
                "reason": c.disqualified_reason,
                "error": c.error,
                "failure_stage": c.failure_stage,
                "requires_dense": c.requires_dense,
                "warnings": list(c.warnings),
                "final_score": c.final_score,
                **{f"metric.{k}": v for k, v in c.accuracy_metrics.items()},
                **{f"relevancy.{k}": v for k, v in c.relevancy_scores.items()},
                **{f"validation.{k}": v for k, v in c.validation_metrics.items()},
            }
            rows.append(row)
        rows.sort(key=lambda r: (r["final_score"] is None, -(r["final_score"] or 0)))
        return rows
