import pandas as pd
import pytest
from sklearn.datasets import load_iris, make_blobs, make_regression

from mlforge import AutoML


@pytest.fixture
def iris_df():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=[f"feature_{i}" for i in range(4)])
    df["target"] = iris.target
    return df


@pytest.fixture
def regression_df():
    X, y = make_regression(
        n_samples=120,
        n_features=4,
        n_informative=3,
        noise=0.2,
        random_state=7,
    )
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(4)])
    df["target"] = y
    return df


@pytest.fixture
def clustering_df():
    X, _ = make_blobs(
        n_samples=120,
        centers=3,
        n_features=3,
        cluster_std=0.7,
        random_state=11,
    )
    return pd.DataFrame(X, columns=[f"feature_{i}" for i in range(3)])


@pytest.mark.parametrize(
    "target_argument",
    ["target", None],
    ids=["explicit-target", "inferred-target"],
)
def test_automl_detects_iris_classification_and_exposes_results(
    iris_df, target_argument
):
    kwargs = {"task": "auto"}
    if target_argument is not None:
        kwargs["target"] = target_argument

    result = AutoML.run(iris_df, **kwargs)

    assert result.task == "classification"
    assert result.target == "target"
    predictions = result.predict(iris_df.drop(columns=["target"]))
    assert len(predictions) == len(iris_df)
    report = result.report()
    assert isinstance(report, str)
    assert report.strip()
    assert len(result.leaderboard()) > 0
    assert result.leaderboard()[0]["metric.roc_auc"] > 0.5


def test_automl_detects_regression(regression_df):
    result = AutoML.run(regression_df, target="target", task="auto")

    assert result.task == "regression"
    assert result.target == "target"
    predictions = result.predict(regression_df.drop(columns=["target"]))
    assert len(predictions) == len(regression_df)
    assert result.report().strip()
    assert len(result.leaderboard()) > 0


def test_automl_clusters_when_no_target_is_available(clustering_df):
    result = AutoML.run(clustering_df, task="auto")

    assert result.task == "clustering"
    assert result.target is None
    labels = pd.Series(result.predict(clustering_df))
    assert len(labels) == len(clustering_df)
    assert labels.nunique() >= 2
    assert result.report().strip()
    assert len(result.leaderboard()) > 0


@pytest.mark.parametrize(
    "data, kwargs",
    [
        ("not a dataframe", {"task": "auto"}),
        (pd.DataFrame(), {"task": "auto"}),
    ],
    ids=["non-dataframe", "empty-dataframe"],
)
def test_automl_rejects_invalid_data(data, kwargs):
    with pytest.raises((TypeError, ValueError)):
        AutoML.run(data, **kwargs)


def test_automl_rejects_invalid_target_and_task(iris_df):
    with pytest.raises((TypeError, ValueError)):
        AutoML.run(iris_df, target="missing", task="auto")

    with pytest.raises((TypeError, ValueError)):
        AutoML.run(iris_df, target="target", task="not-a-task")
