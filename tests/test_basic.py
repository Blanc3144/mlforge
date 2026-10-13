import pandas as pd
from sklearn.datasets import make_classification, make_regression

from mlforge import ProblemDefinition, RelevancyCriteria, MLPipeline, IncrementalLearner
from mlforge.eda import EDAReport


def _classification_df(n=500, seed=0):
    X, y = make_classification(n_samples=n, n_features=6, random_state=seed)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    df["target"] = y
    return df


def _regression_df(n=500, seed=0):
    X, y = make_regression(n_samples=n, n_features=6, noise=5, random_state=seed)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    df["target"] = y
    return df


def test_classification_pipeline_runs():
    df = _classification_df()
    problem = ProblemDefinition(problem_type="classification", target="target")
    pipeline = MLPipeline(problem)
    pipeline.run_eda(df)
    pipeline.fit(df, verbose=False)
    assert pipeline.best_ is not None
    preds = pipeline.predict(df.drop(columns=["target"]))
    assert len(preds) == len(df)


def test_regression_pipeline_runs():
    df = _regression_df()
    problem = ProblemDefinition(problem_type="regression", target="target")
    pipeline = MLPipeline(problem)
    pipeline.fit(df, verbose=False)
    assert pipeline.best_ is not None


def test_hard_constraint_disqualifies():
    df = _classification_df()

    def reject_everything(model, X_train, y_train, X_test, y_test):
        return False

    problem = ProblemDefinition(
        problem_type="classification", target="target",
        relevancy=RelevancyCriteria(hard_constraints=[reject_everything]),
    )
    pipeline = MLPipeline(problem)
    try:
        pipeline.fit(df, verbose=False)
        assert False, "expected RuntimeError when all candidates disqualified"
    except RuntimeError:
        pass


def test_incremental_update():
    df = _classification_df(n=500, seed=1)
    problem = ProblemDefinition(problem_type="classification", target="target")
    pipeline = MLPipeline(problem)
    pipeline.fit(df, verbose=False)

    learner = IncrementalLearner(pipeline)
    new_df = _classification_df(n=100, seed=2)
    learner.update(new_df, verbose=False)
    assert len(learner.versions) == 2
    assert learner.version_history().iloc[-1]["accuracy_metrics"]


def test_eda_supports_all_categorical_data():
    df = pd.DataFrame({"color": ["red", "blue", "red"], "target": ["yes", "no", "yes"]})
    report = EDAReport(df, target="target").run()
    assert report["numeric_summary"] == {}
    assert report["categorical_cardinality"]["color"] == 2


def test_missing_target_is_rejected():
    df = _classification_df().drop(columns=["target"])
    problem = ProblemDefinition(problem_type="classification", target="target")
    try:
        MLPipeline(problem).fit(df, verbose=False)
        assert False, "expected missing target error"
    except ValueError as exc:
        assert "missing" in str(exc).lower()


def test_beginner_report_is_readable_for_classification():
    df = _classification_df(n=200)
    pipeline = MLPipeline(ProblemDefinition(problem_type="classification", target="target"))
    pipeline.run_eda(df)
    pipeline.fit(df, verbose=False)

    report = pipeline.report()
    assert "MLFORGE RESULTS" in report
    assert "RECOMMENDED MODEL" in report
    assert "MODEL COMPARISON" in report
    assert "Selection quality shows" in report
    assert "{" not in report


def test_beginner_report_is_readable_for_regression():
    df = _regression_df(n=200)
    pipeline = MLPipeline(ProblemDefinition(problem_type="regression", target="target"))
    pipeline.run_eda(df)
    pipeline.fit(df, verbose=False)

    report = pipeline.report()
    assert "R² score" in report
    assert "MAE" in report
    assert "RMSE" in report


if __name__ == "__main__":
    test_classification_pipeline_runs()
    test_regression_pipeline_runs()
    test_hard_constraint_disqualifies()
    test_incremental_update()
    print("All tests passed.")
