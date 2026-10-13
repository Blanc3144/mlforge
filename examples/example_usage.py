"""
End-to-end example: classification problem on a synthetic dataset,
showing every lifecycle stage including a simulated "new data arrives
later" incremental update.
"""
import pandas as pd
from sklearn.datasets import make_classification

from mlforge import (
    ProblemDefinition, RelevancyCriteria, MLPipeline, IncrementalLearner,
)

# ---------------------------------------------------------------------
# 1. Get some data (swap this for DataLoader.from_csv(...) etc.)
# ---------------------------------------------------------------------
X, y = make_classification(n_samples=2000, n_features=10, n_informative=6,
                            weights=[0.8, 0.2], random_state=42)
df = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
df["target"] = y

# ---------------------------------------------------------------------
# 2. Formulate the problem explicitly, including what "relevant" means
#    here. Say this model needs to be explainable (e.g. compliance),
#    so we weight interpretability heavily and add a hard constraint
#    that rejects black-box models entirely.
# ---------------------------------------------------------------------
def must_be_reasonably_fast(model, X_train, y_train, X_test, y_test):
    import time
    t0 = time.time()
    model.predict(X_test[:100])
    return (time.time() - t0) < 1.0  # toy example constraint

relevancy = RelevancyCriteria(
    weights={"accuracy": 0.5, "interpretability": 0.35, "inference_speed": 0.15},
    hard_constraints=[must_be_reasonably_fast],
)

problem = ProblemDefinition(
    problem_type="classification",
    target="target",
    objective="Flag high-risk cases for manual review; reviewers need to "
              "understand *why* a case was flagged, so a fully black-box "
              "model is not acceptable even if marginally more accurate.",
    relevancy=relevancy,
)

# ---------------------------------------------------------------------
# 3-6. EDA, preprocessing, training many candidate algorithms, evaluation
#      — all handled by MLPipeline
# ---------------------------------------------------------------------
pipeline = MLPipeline(problem)
pipeline.run_eda(df)
pipeline.fit(df)

pipeline.print_report()

# ---------------------------------------------------------------------
# 7. Keep improving: simulate a new batch of labeled data coming in
#    later (e.g. next week's reviewed cases) and update the model.
# ---------------------------------------------------------------------
learner = IncrementalLearner(pipeline, max_history_rows=10000)

X_new, y_new = make_classification(n_samples=300, n_features=10, n_informative=6,
                                    weights=[0.8, 0.2], random_state=99)
new_df = pd.DataFrame(X_new, columns=[f"f{i}" for i in range(10)])
new_df["target"] = y_new

learner.update(new_df)  # cheap partial_fit if supported, else smart retrain

print("\nUPDATE CHECK")
print("-" * 72)
print(f"New rows added       : {len(new_df):,}")
print(f"Current model        : {pipeline.best_.name}")
print(f"Performance regressed: {'Yes' if learner.has_regressed() else 'No'}")
