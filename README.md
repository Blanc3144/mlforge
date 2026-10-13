# mlforge

**mlforge** is a Python library for making tabular machine learning easier to
start, easier to understand, and more practical to evaluate.

It provides two levels of use:

- **Automatic mode:** give it a pandas DataFrame and let it inspect the data,
  infer a likely task, train several models, select a model, and explain the
  result.
- **Controlled mode:** explicitly define the problem and run each lifecycle
  stage yourself.

The central idea is simple:

> A useful model should not be chosen by predictive score alone. It should
> also be relevant to the real problem: interpretable enough, fast enough,
> stable enough, and acceptable under the user's constraints.

---

## Contents

- [What is mlforge?](#what-is-mlforge)
- [Why is it needed?](#why-is-it-needed)
- [Who is it for?](#who-is-it-for)
- [Current capabilities](#current-capabilities)
- [Learning curve](#learning-curve)
- [Installation](#installation)
- [Five-minute quickstart](#five-minute-quickstart)
- [Automatic API](#automatic-api)
- [Classification](#classification)
- [Regression](#regression)
- [Clustering](#clustering)
- [Predictions](#predictions)
- [Reading the report](#reading-the-report)
- [Controlling model selection](#controlling-model-selection)
- [Hyperparameter tuning](#hyperparameter-tuning)
- [Explicit pipeline API](#explicit-pipeline-api)
- [Data loading and EDA](#data-loading-and-eda)
- [Incremental learning](#incremental-learning)
- [Available public components](#available-public-components)
- [How mlforge works internally](#how-mlforge-works-internally)
- [Data requirements and validation](#data-requirements-and-validation)
- [Known limitations](#known-limitations)
- [Future scope](#future-scope)
- [Project structure](#project-structure)
- [Testing](#testing)
- [License and status](#license-and-status)

---

## What is mlforge?

mlforge is an end-to-end **tabular AutoML and model-selection framework**.
It currently works with pandas DataFrames and scikit-learn estimators for:

- Classification: predict a category or class
- Regression: predict a numeric value
- Basic clustering: group unlabeled rows using KMeans

It brings together these stages:

```text
data loading
    -> task and target detection
    -> data inspection
    -> preprocessing
    -> exploratory analysis
    -> multi-model training
    -> evaluation
    -> relevancy-aware selection
    -> readable reporting
    -> prediction
    -> optional incremental updating
```

The library is built for transparent automation. It does not claim to
understand the business meaning of arbitrary columns with certainty. When it
makes an automatic decision, it reports the decision, confidence, and any
warning so that a user can override it.

## Why is it needed?

A beginner often has to learn many separate tools and decisions before
getting a first useful result:

1. Load the data
2. Decide whether the task is classification or regression
3. Identify the target column
4. Handle missing values
5. Encode categories
6. Scale numeric features
7. Choose algorithms
8. Choose evaluation metrics
9. Compare models
10. Explain why one model was selected

Those steps are important, but they are easy to perform incorrectly or in the
wrong order. mlforge provides a safe starting workflow while keeping the
individual components available for advanced users.

### Accuracy is not the whole decision

The most accurate model is not always the most useful model. A model may also
need to be:

- Explainable to users, auditors, or regulators
- Fast enough for the application
- Stable across validation folds
- Acceptable under a domain-specific rule
- Cheap enough to train or operate

mlforge makes those preferences explicit through `RelevancyCriteria` instead
of hiding them in an unexplained model choice.

## Who is it for?

### Beginners

Use the one-call API to learn the complete machine-learning workflow without
having to build every component first.

### Students and educators

Use the report to connect concepts such as preprocessing, holdout testing,
model comparison, and model selection to visible output.

### Python developers

Use the automatic result object as a compact baseline for a data application,
then inspect or replace individual components as the project grows.

### Data scientists building prototypes

Use the model zoo, relevancy weights, custom scorers, constraints, and
optional tuning to create a repeatable first experiment.

### Important boundary

mlforge is currently a development and experimentation library, not a complete
production platform. It does not automatically guarantee fairness, detect all
forms of data leakage, or understand the business context of a dataset.

---

## Current capabilities

mlforge currently provides:

- One-call automatic workflow through `AutoML.run()`
- Automatic best-effort task detection
- Automatic best-effort target detection
- Classification, regression, and basic KMeans clustering
- Numeric missing-value imputation
- Categorical missing-value imputation
- Numeric standard scaling
- Categorical one-hot encoding
- Unknown-category handling during prediction
- Basic EDA reports
- Optional EDA plot export
- Multiple scikit-learn candidate models
- Holdout evaluation
- Multiclass ROC-AUC where probabilities are available
- Relevancy-aware model selection
- Hard constraints and custom relevancy scorers
- Optional stability scoring with fold-local cross-validation
- Optional bounded hyperparameter tuning
- Beginner-friendly text reports
- Machine-readable leaderboards
- Predictions for new rows
- Incremental updates and version history
- Candidate failure isolation and readable warnings

---

## Learning curve

The recommended learning path is intentionally gradual.

### Level 1: Run AutoML

You only need pandas and one call:

```python
from mlforge import AutoML

result = AutoML.run(dataset)
result.print_report()
```

Learn to understand:

- What task was detected
- Which target was selected
- Which model was recommended
- What the reported metrics mean

### Level 2: Predict new data

```python
predictions = result.predict(new_data)
```

Learn that new rows must contain the same feature columns used for training.
Leave out the target column: it is the answer you are asking the model to
predict.

### Level 3: Inspect decisions

```python
result.warnings
result.eda
result.leaderboard()
result.explain()
```

Learn how the automatic workflow made its decisions.

### Level 4: Control the task and relevancy

```python
result = AutoML.run(
    dataset,
    target="species",
    task="classification",
    relevancy_weights={
        "accuracy": 0.70,
        "interpretability": 0.20,
        "inference_speed": 0.10,
    },
)
```

### Level 5: Use the explicit pipeline

Use `ProblemDefinition`, `MLPipeline`, `Evaluator`, `ModelZoo`, and related
components when you need complete control over the workflow.

Most users should be productive after the first two levels. Understanding
the model-selection design and custom scoring requires a basic understanding
of supervised learning, validation metrics, and data leakage.

---

## Installation

### Requirements

- Python 3.9 or newer
- pandas >= 1.5
- NumPy >= 1.23
- scikit-learn >= 1.2

### Install from this project directory

From a terminal opened in the repository:

```bash
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Install mlforge in editable mode:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

Editable mode means that changes to the local source are immediately
available when you import `mlforge`.

### Optional plotting support

To export EDA plots:

```bash
python -m pip install -e ".[plots]"
```

### Verify the installation

```bash
python -c "import mlforge; print(mlforge.__version__)"
```

---

## Five-minute quickstart

Suppose `IRIS.csv` contains these columns:

```text
sepal_length,sepal_width,petal_length,petal_width,species
```

Create `run_iris.py`:

```python
import pandas as pd
from mlforge import AutoML

dataset = pd.read_csv("IRIS.csv")

result = AutoML.run(dataset)
result.print_report()

new_flower = pd.DataFrame([
    {
        "sepal_length": 6.1,
        "sepal_width": 2.8,
        "petal_length": 4.7,
        "petal_width": 1.2,
    }
])

prediction = result.predict(new_flower)
print("Predicted species:", prediction[0])
```

Run it from the directory containing both the script and `IRIS.csv`:

```bash
python run_iris.py
```

The automatic workflow should detect `species` as the target and identify a
classification task. It will train several candidate models, select one, and
print a report before predicting the new flower.

If `IRIS.csv` is in another directory, use its path explicitly:

```python
dataset = pd.read_csv(r"D:\data\IRIS.csv")
```

---

## Automatic API

The main public entry point is:

```python
from mlforge import AutoML

result = AutoML.run(dataset)
```

The default values are equivalent to:

```python
result = AutoML.run(
    dataset,
    target=None,
    task="auto",
    random_state=42,
    test_size=0.2,
)
```

### Automatic target and task detection

When `task="auto"`, mlforge uses best-effort heuristics based on:

- Familiar target names such as `target`, `label`, `species`, `outcome`, or
  `response`
- Column data types
- Target cardinality
- The shape of the table

If a target is detected, its type is used to infer classification or
regression. If no reliable target is detected, automatic mode runs clustering
instead of pretending that an unlabeled dataset is supervised.

The result includes:

```python
result.task             # "classification", "regression", or "clustering"
result.target           # target column name, or None for clustering
result.target_confidence
result.task_confidence
result.warnings
```

Automatic detection is a convenience, not a proof of intent. If the choice
matters, specify the target explicitly:

```python
result = AutoML.run(
    dataset,
    target="species",
    task="auto",
)
```

### Complete `AutoML.run()` options

```python
AutoML.run(
    df,
    target=None,
    task="auto",
    relevancy=None,
    relevancy_weights=None,
    custom_scorers=None,
    hard_constraints=None,
    relevancy_settings=None,
    objective="",
    positive_class=None,
    test_size=0.2,
    random_state=42,
    model_names=None,
    tune=False,
    cv=3,
    tuning_grids=None,
    n_clusters=None,
    max_clusters=8,
    verbose=False,
)
```

The `result` object provides:

```python
result.task
result.target
result.model
result.model_                 # scikit-learn-style alias
result.best_model
result.pipeline
result.eda
result.eda_report             # alias for result.eda
result.warnings
result.explanation
result.report()
result.print_report()
result.explain()
result.predict(new_data)
result.leaderboard()
```

### Option reference

| Option | Purpose |
|---|---|
| `df` | Non-empty pandas DataFrame to analyze and train on. |
| `target` | Outcome column for supervised learning; overrides target inference. |
| `task` | `auto`, `classification`, `regression`, or `clustering`. |
| `relevancy` | A `RelevancyCriteria` object or mapping of weights/settings. |
| `relevancy_weights` | Direct mapping of selection criteria to relative importance. |
| `custom_scorers` | Named functions returning relevancy scores from 0 to 1. |
| `hard_constraints` | Functions that reject a supervised candidate by returning `False`. |
| `relevancy_settings` | Alternative settings mapping; do not combine with other relevancy arguments. |
| `objective` | Human-readable goal, stored as documentation; not interpreted by an AI. |
| `positive_class` | Stored problem metadata; currently does not change metric calculation or thresholds. |
| `test_size` | Fraction reserved for supervised holdout evaluation; default 20%. |
| `random_state` | Seed for splitting and supported estimators. Timing-based rankings can still vary. |
| `model_names` | Optional subset of registered supervised model names. |
| `tune` | Enable grid search for models that have a supplied grid. |
| `cv` | Requested folds for tuning and optional stability estimation; bounded by available rows/classes. |
| `tuning_grids` | Parameter grids keyed by supervised model name. |
| `n_clusters` | Fix the KMeans cluster count instead of searching. |
| `max_clusters` | Maximum searched cluster count, default 8; also bounded by row count. |
| `verbose` | Print supervised training progress; default `False`. |

**Clustering has a separate selection policy.** It uses KMeans and silhouette,
not the supervised relevancy weights, constraints, model list, or tuning grids.
Do not pass `target` with `task="clustering"`.

---

## Classification

Classification predicts categories: species, spam/not-spam, or customer churn.
This example runs without a CSV or network download:

```python
import pandas as pd
from sklearn.datasets import load_iris
from mlforge import AutoML

iris = load_iris()
dataset = pd.DataFrame(
    iris.data,
    columns=["sepal_length", "sepal_width", "petal_length", "petal_width"],
)
dataset["species"] = [iris.target_names[i] for i in iris.target]

result = AutoML.run(dataset, target="species", task="classification")
result.print_report()
```

### Included classification models

| Registry name | Estimator |
|---|---|
| `logistic_regression` | Logistic regression |
| `decision_tree` | Decision tree, default maximum depth 8 |
| `random_forest` | Random forest, default 200 trees |
| `gradient_boosting` | Gradient boosting classifier |
| `knn` | K-nearest neighbors |
| `svm` | SVC wrapped in probability calibration |
| `naive_bayes` | Gaussian naive Bayes |
| `sgd_linear` | SGD classifier with logistic loss; supports partial updates |

To shorten an experiment:

```python
result = AutoML.run(
    dataset,
    target="species",
    model_names=["logistic_regression", "decision_tree", "naive_bayes"],
)
```

## Regression

Regression predicts numbers: prices, durations, or measurements.

```python
import pandas as pd
from sklearn.datasets import make_regression
from mlforge import AutoML

X, y = make_regression(n_samples=200, n_features=3, noise=5, random_state=42)
dataset = pd.DataFrame(X, columns=["feature_a", "feature_b", "feature_c"])
dataset["target"] = y

result = AutoML.run(dataset, target="target", task="regression")
result.print_report()

new_data = pd.DataFrame([
    {"feature_a": 0.2, "feature_b": -0.1, "feature_c": 0.5}
])
print(result.predict(new_data))
```

| Registry name | Estimator |
|---|---|
| `linear_regression` | Ordinary least-squares linear regression |
| `ridge` | Ridge regression |
| `decision_tree` | Decision tree regressor |
| `random_forest` | Random forest regressor |
| `gradient_boosting` | Gradient boosting regressor |
| `knn` | K-nearest neighbors regressor |
| `svr` | Support vector regressor |
| `sgd_linear` | SGD regressor; supports partial updates |

For numeric targets representing category codes, set `task="classification"`
explicitly. A small set of numeric values is not enough to determine intent.

## Clustering

Clustering finds groups without known outcome labels.

```python
import pandas as pd
from sklearn.datasets import make_blobs
from mlforge import AutoML

X, _ = make_blobs(n_samples=150, centers=3, random_state=42)
dataset = pd.DataFrame(X, columns=["measurement_a", "measurement_b"])

result = AutoML.run(dataset, task="clustering", max_clusters=6)
result.print_report()
print(result.leaderboard())

cluster_ids = result.predict(dataset)
grouped_data = dataset.assign(cluster=cluster_ids)
print(grouped_data.head())
```

- Without `n_clusters`, tries KMeans with several counts starting at 2.
- Selects the largest available silhouette score; if none is available,
  retains the first candidate.
- `n_clusters=3` fixes the count rather than searching for it.
- Cluster IDs such as `0`, `1`, and `2` are arbitrary group identifiers, not
  predicted species or ordered quality ratings.
- `result.predict(new_data)` assigns new rows to the nearest fitted center.
- Remove known labels and irrelevant identifiers before explicit clustering:
  otherwise every supplied column is treated as a feature.

KMeans is currently the only clustering algorithm. It can process encoded
categories, but Euclidean distance on one-hot columns may not reflect useful
business similarity. Silhouette is measured on the fitted dataset, not on an
independent test set, and can be expensive on large data.

## Predictions

Always predict through the result or pipeline so preprocessing is reused:

```python
predictions = result.predict(new_data)
```

For the Iris quickstart, prediction columns are exactly:

```text
sepal_length, sepal_width, petal_length, petal_width
```

Keep names, meanings, and units consistent with training. Provide features
only; the current transformer ignores extra columns, including a supplied
target, but including known answers is unnecessary and confusing.

`result.model` is the underlying estimator, not a raw-DataFrame prediction
wrapper. Calling it directly bypasses mlforge's preprocessing.

Every call to `AutoML.run()` trains a new workflow. No server, API key, or
background service is required. There is currently no built-in save/load API;
keep the fitted result in the Python process while using it.

## Reading the report

`result.print_report()` prints text; `result.report()` returns that text as a
string. `result.leaderboard()` returns a list of dictionaries for analysis:

```python
comparison = pd.DataFrame(result.leaderboard())
print(comparison.to_string(index=False))
```

### Metric guide

| Metric | Meaning | Direction |
|---|---|---|
| Accuracy | Fraction of correct classifications | Higher is better |
| Weighted F1 | Class-size-weighted combination of precision and recall | Higher is better |
| Weighted precision | Weighted measure of how often predicted classes are correct | Higher is better |
| Weighted recall | Weighted measure of how many actual class members are recovered | Higher is better |
| ROC-AUC | Probability-ranking quality; multiclass uses one-vs-rest | Higher is better |
| R² | Regression fit compared with predicting a constant mean; can be negative | Higher is better |
| MAE | Average absolute prediction error in target units | Lower is better |
| RMSE | Error in target units, penalizing large errors more strongly | Lower is better |
| Silhouette | Cluster separation versus cohesion, from -1 to 1 | Higher is generally better |

**Weighted F1 does not solve class imbalance or fairness.** A large majority
class can still dominate it. Inspect per-class and per-group errors separately
for important decisions; those audits are not automated here.

The report's **overall score is not accuracy or a probability**. It is a blend
of the configured selection criteria. A model with lower accuracy may win
because it better matches those preferences.

The supervised leaderboard includes model name, eligibility, failure reason,
`final_score`, `metric.*`, `relevancy.*`, and any `validation.*` values. Not every
metric is available for every estimator. Failed models remain visible instead
of silently disappearing.

Detection confidence values are **hand-set heuristic scores**, not calibrated
probabilities. A score of `0.95` does not establish a 95% chance that the chosen
column is the correct business target.

---

## Controlling model selection

The default supervised weights are:

| Criterion key | Weight | What is actually used |
|---|---|---|
| `accuracy` | 0.60 | Weighted F1 for classification; R² clipped to [0, 1] for regression |
| `interpretability` | 0.20 | Predefined estimator metadata, not a computed explanation |
| `train_speed` | 0.10 | Measured fit time relative to the slowest candidate |
| `inference_speed` | 0.10 | Predefined estimator speed rating, not measured prediction latency |

Weights are normalized. For eligible models, selection combines available
criterion scores using their normalized weights. If a criterion is unavailable,
the selector renormalizes over available criteria.

This is **rule-based scoring and reporting**, not LLM reasoning. It does not
use an AI service, infer regulatory requirements from `objective`, prove that
a model is explainable, or measure how much each preprocessing step improved
the score. An interpretability rating is a useful preference, not evidence
about an individual prediction.

### Customize the balance

```python
result = AutoML.run(
    dataset,
    target="species",
    relevancy_weights={
        "accuracy": 0.70,
        "interpretability": 0.20,
        "inference_speed": 0.10,
    },
)
```

Examples in this section assume `dataset` is the Iris DataFrame from the
classification example.

### Include stability

Stability is optional and is not part of the default weights:

```python
result = AutoML.run(
    dataset,
    target="species",
    model_names=["logistic_regression", "decision_tree"],
    cv=3,
    relevancy_weights={
        "accuracy": 0.65,
        "interpretability": 0.20,
        "stability": 0.15,
    },
)
```

When requested, the library fits preprocessing separately within each fold of
the training split and computes the candidate's score variation. The heuristic
is `stability = 1 - clip(cv_std, 0, 1)`.

Inspect `validation.cv_mean`, `validation.cv_std`, and `validation.cv_folds`
in the leaderboard. High stability alone is not enough: a consistently poor
model can also be stable. If there are too few usable folds, stability is
skipped with a warning. Passing `cv=5` alone does not enable stability.

### Custom scorers and hard constraints

Custom scorers contribute a numeric value between 0 and 1. Hard constraints
return `False` to disqualify a model regardless of its score.

```python
from time import perf_counter
from mlforge import AutoML, RelevancyCriteria

def latency_score(model, X_train, y_train, X_test, y_test, fit_seconds):
    start = perf_counter()
    model.predict(X_test)
    elapsed = perf_counter() - start
    return max(0.0, 1.0 - elapsed / 1.0)

def under_one_second(model, X_train, y_train, X_test, y_test):
    start = perf_counter()
    model.predict(X_test)
    return perf_counter() - start < 1.0

criteria = RelevancyCriteria(
    weights={"accuracy": 0.7, "interpretability": 0.2, "latency": 0.1},
    custom_scorers={"latency": latency_score},
    hard_constraints=[under_one_second],
)
result = AutoML.run(dataset, target="species", relevancy=criteria)
```

These are illustrative, single-batch timing checks, not production latency
benchmarks. Callbacks receive **transformed feature matrices**, not the raw
DataFrame. Protected-group data is not passed separately for fairness audits.
Keep callbacks side-effect-free and ensure returned scores are finite and
within [0, 1].

## Hyperparameter tuning

Tuning tries alternative estimator settings. It is opt-in and requires grids:

```python
result = AutoML.run(
    dataset,
    target="species",
    model_names=["logistic_regression", "decision_tree"],
    tune=True,
    cv=3,
    tuning_grids={
        "logistic_regression": {"C": [0.1, 1.0, 10.0]},
        "decision_tree": {"max_depth": [3, 5, 8]},
    },
)
```

- Grid search operates only on the training split.
- Preprocessing is fitted separately inside each search fold.
- Classification tuning uses weighted F1; regression tuning uses R².
- Models with no supplied grid keep their default settings.
- Use estimator parameter names; for the calibrated SVM wrapper, use nested
  names such as `estimator__C`.
- Keep grids to estimator settings. Tuning preprocessing settings is not a
  supported end-to-end workflow.
- Optional stability evaluation currently rebuilds default estimators rather
  than measuring the selected tuned configuration. Do not interpret that CV
  summary as independent validation of tuned parameters.

The holdout is used to compare and choose models, so it is a **model-selection
validation set**, not an untouched final test. For reliable claims about
deployment performance, keep an additional independent test dataset.

## Explicit pipeline API

Use the manual API when you want to control the problem definition or register
new algorithms. This continues the Iris example:

```python
from mlforge import ProblemDefinition, RelevancyCriteria, MLPipeline

problem = ProblemDefinition(
    problem_type="classification",
    target="species",
    objective="Predict iris species from flower measurements.",
    relevancy=RelevancyCriteria(
        weights={"accuracy": 0.7, "interpretability": 0.3}
    ),
)
pipeline = MLPipeline(problem, test_size=0.2, random_state=42)
eda = pipeline.run_eda(dataset)
pipeline.fit(dataset, verbose=False)
pipeline.print_report()

new_data = dataset.drop(columns="species").head(3)
print(pipeline.predict(new_data))  # demonstration, not independent evaluation
```

`MLPipeline` supports supervised classification and regression. Use `AutoML`
for the clustering workflow.

### Register another estimator

```python
from sklearn.linear_model import LogisticRegression
from mlforge.models import ModelSpec

pipeline.zoo.register(ModelSpec(
    name="strongly_regularized_logistic",
    build_fn=lambda: LogisticRegression(C=0.1, max_iter=1000, random_state=42),
    interpretability=0.9,
    relative_speed=0.9,
))
pipeline.fit(dataset, verbose=False)
```

The factory must return a fresh estimator. Set `requires_dense=True` for an
estimator that cannot accept sparse one-hot features. `ModelSpec` is imported
from `mlforge.models`, not from the top-level package.

## Data loading and EDA

### Load tabular data

You may use pandas directly or the convenience wrapper:

```python
from mlforge import DataLoader

dataset = DataLoader.from_csv("IRIS.csv")
print(DataLoader.basic_report(dataset))
```

| Function | Return value |
|---|---|
| `DataLoader.from_csv(path, **kwargs)` | pandas DataFrame |
| `DataLoader.from_dataframe(df)` | Copy of an existing DataFrame |
| `DataLoader.from_dict(data)` | DataFrame built from a dictionary |
| `DataLoader.from_sql(query, connection, **kwargs)` | DataFrame from a caller-provided database connection |
| `DataLoader.basic_report(df)` | Shape, columns, types, missing counts, and duplicate count |

SQL use requires you to supply a working connection and any database-specific
driver. These wrappers load existing data; they do not collect labels or scrape
external sources. Load other formats with pandas and pass the DataFrame in.

### Inspect data and export plots

```python
from mlforge import EDAReport

eda = EDAReport(dataset, target="species")
details = eda.run()
print(details["missing_pct"])
print(details["target_distribution"])

# Requires the optional plots dependency:
# paths = eda.save_plots("eda_plots")
```

EDA provides data types, numeric summaries, categorical cardinality,
missing-value percentages, numeric correlations, and target distribution.
`save_plots()` saves a correlation heatmap and target-distribution chart where
applicable. Review chart usefulness for continuous or high-cardinality targets.

EDA describes the original data. It does not automatically remove duplicates,
repair category spelling, audit fairness, or remove suspicious features.

## Incremental learning

Use a **fitted supervised pipeline** and new rows containing known target labels:

```python
import pandas as pd
from mlforge import AutoML, IncrementalLearner

dataset = pd.read_csv("IRIS.csv")
result = AutoML.run(dataset, target="species")
learner = IncrementalLearner(result.pipeline, max_history_rows=10000)

new_batch = pd.read_csv("new_labeled_iris.csv")
learner.update(new_batch, verbose=False)

print(learner.version_history().to_string(index=False))
print("Latest metric declined:", learner.has_regressed())

# Use the learner's current pipeline after updates:
learner.pipeline.print_report()
```

- Uses `partial_fit` when supported, unless `retrain_full_tournament=True`.
- Otherwise retrains candidates on combined history, optionally limiting
  retained rows with `max_history_rows`.
- `version_history()` returns metadata and metrics, not saved model snapshots.
- `has_regressed(metric=None)` compares adjacent recorded metrics; lower is
  better for recognized loss metrics such as MAE/RMSE.
- A full retrain can change the holdout rows, so metric changes are not a
  controlled before/after comparison. Improvement is not guaranteed.
- There is no automatic drift test, rollback, or scheduling.
- Full retraining does not preserve the original model subset/tuning arguments.
- After retraining, `result.model`, `result.eda`, and its initial explanation may
  refer to the original run. Use `learner.pipeline` for the current state.
- New classes and schema changes require special care; they are not a supported
  seamless partial-update workflow.

---

## Available public components

Unless noted otherwise, import these names from `mlforge`.

| Component | Key methods / attributes | Role |
|---|---|---|
| `AutoML` | `run(df, ...)` | One-call task inference, training, and reporting |
| `AutoMLResult` | `predict`, `report`, `print_report`, `leaderboard`, `explain`, `problem_def` | Fitted workflow, results, and supervised problem metadata |
| `ProblemDefinition` | `summary()`; `problem_type`, `target`, `objective`, `relevancy` | Explicit supervised problem contract |
| `RelevancyCriteria` | `normalized_weights()`; `weights`, `custom_scorers`, `hard_constraints` | Selection preferences and rejection rules |
| `DataLoader` | `from_csv`, `from_dataframe`, `from_dict`, `from_sql`, `basic_report` | Data loading and basic checks |
| `Preprocessor` | `infer_feature_types`, `build_transformer`, `fit_transform`, `transform` | Numeric/categorical feature preparation |
| `Preprocessor` | `fit_transform_features`, `transform_features` | Feature-only preprocessing for unsupervised data |
| `EDAReport` | `run`, `save_plots` | Data summaries and optional PNG plots |
| `ModelZoo` | `all`, `get`, `register` | Supervised estimator registry |
| `Evaluator` | `evaluate_classification`, `evaluate_regression`, `primary_accuracy_score`, `relevancy_scores`, `cross_validate` | Metrics and validation helpers |
| `ModelSelector` | `select`, `leaderboard` | Weighted scoring of eligible candidates |
| `MLPipeline` | `run_eda`, `fit`, `predict`, `report`, `print_report`, `summary`, `leaderboard` | Explicit supervised orchestration |
| `IncrementalLearner` | `update`, `version_history`, `has_regressed` | In-process supervised updates and metric history |
| `mlforge.models.ModelSpec` | Estimator factory, interpretability/speed tags, dense-input flag | Register a new model |
| `mlforge.evaluation.CandidateResult` | Model, metrics, validation, eligibility, failure details | Advanced candidate result container |

### Manual preprocessing contract

`Preprocessor.fit_transform(df, target)` returns **`(X, y)`**, not a processed
DataFrame with target metadata. `transform(new_df)` returns transformed
features. Feature-only fitting returns `X` alone.

Use `numeric_features`, `categorical_features`, and `scale_numeric` to customize
a `Preprocessor`. Transformations may return a NumPy array or sparse matrix.
For supervised evaluation, fit it only on training rows; fitting it on all rows
before splitting would leak information. The built-in pipeline handles this.

The illustrative names `datapreprocessor`, `eda_handeler`, `modelselecter`,
`modelevaluator`, and `trainning_reporter` are **not package exports**. Use the
documented `AutoML` API or the class names above.

## How mlforge works internally

### Supervised path

1. **Validate and infer:** check inputs, identify a likely target or honor the
   override, then infer a task from dtype/cardinality if requested.
2. **Inspect:** compute EDA on the supplied DataFrame for descriptive reporting.
3. **Split:** reserve 20% by default; classification uses stratification.
4. **Fit preprocessing on training rows:** median imputation and scaling for
   numbers; most-frequent imputation and one-hot encoding for categories.
5. **Train:** fit each requested estimator; optionally grid-search supplied
   settings. Candidate-level failures are recorded so others can continue.
6. **Evaluate:** measure each candidate on the holdout. Compute optional
   fold-local stability on training data when weighted.
7. **Apply preferences:** compute relevancy ratings and custom scorers; reject
   candidates that fail hard constraints.
8. **Select:** choose the highest eligible weighted score.
9. **Report and predict:** expose structured results and reuse fitted
   preprocessing for new rows.

The selected estimator remains fitted on the training split. There is no
automatic final refit on all rows. A successful run means a model was selected,
not that it is deployment-ready.

### Unsupervised path

All supplied columns are features. Preprocessing is fitted on the whole
dataset, several KMeans cluster counts are compared where requested, and the
result exposes cluster assignments plus silhouette/inertia information.

## Data requirements and validation

- Use a non-empty pandas DataFrame with unique, preferably string column names.
- Supervised targets must exist, contain no missing values, and have at least
  two distinct values; there must also be a usable feature column.
- Supply enough rows for both train and holdout partitions. Classification
  needs at least two examples per class and enough rows per split for classes;
  individual models or CV may require more.
- Basic clustering needs at least two rows and some non-constant information.
- Ordinary numeric and categorical features are the intended input. Clean
  mixed-type values, infinite values, dates, free text, and complex objects first.
- ID columns, duplicate rows, and target-derived features are not automatically
  removed. Review them before training.
- High-cardinality categories can produce many one-hot columns and large
  memory requirements, especially for dense-only models.
- Use matching feature columns and compatible values for prediction.

### Troubleshooting

| Symptom | What to check |
|---|---|
| `ModuleNotFoundError` | Install with `python -m pip install -e .` using the same interpreter that runs your script. |
| `IRIS.csv` not found | Paths are resolved from the current working directory; use an absolute path or run from the CSV directory. |
| PowerShell activation blocked | Run `.\.venv\Scripts\python.exe run_iris.py` directly; activation is optional. |
| Wrong target or task | Supply `target="..."` and `task="classification"` or `"regression"`. |
| Unexpected clustering | No reliable target was inferred; specify the intended target. |
| Missing feature columns | Match prediction column names to the training feature names. |
| No eligible candidate | Check candidate errors, sample sizes, grids, and hard constraints. |
| High accuracy but poor useful predictions | Audit minority-class errors, leakage, target meaning, and dataset representativeness. |

## Known limitations

| Area | Current boundary |
|---|---|
| Automatic reasoning | Rules and templates, not general AI reasoning or business understanding |
| Detection confidence | Heuristic scores, not statistically calibrated probabilities |
| Reporting | Explains the selection policy; no causal explanation of predictions or measured benefit of every processing step |
| Fairness | No built-in protected-group audit, mitigation, or fairness guarantee |
| Interpretability | Estimator metadata; no SHAP, feature-attribution, or per-row explanation API |
| Validation | Selection uses the holdout; no untouched final-test management or confidence intervals |
| Splitting | Random/stratified splits only; not suitable for time-series or repeated-entity data without additional work |
| Metrics | Fixed primary metrics; no general primary-metric selector or automatic threshold optimization |
| Stability | Optional heuristic from fold variability; current tuned-model caveat is described above |
| Clustering | KMeans only; no supervised relevancy rules or semantic group labels |
| Updating | Metric history, not durable model versioning, fixed-benchmark monitoring, rollback, or drift detection |
| Deployment | No built-in persistence, prediction server, UI, scheduler, or distributed execution |
| Scope | Tabular data only; no dedicated NLP, vision, audio, forecasting, or deep-learning workflow |

**Responsible use:** biased labels or underrepresented groups can produce
biased predictions even with high aggregate scores. Interpretability and
weighted F1 do not establish fairness. Preserve appropriate audit information,
evaluate groups on independent data, protect sensitive data, and involve domain
experts before using models for consequential decisions.

## Future scope

These are **proposed improvements, not implemented APIs or release promises**.

### Reliability and evaluation

- Configurable primary metrics, positive-class metrics, and decision thresholds
- Group-aware and chronological splitting
- Independent final-test evaluation and uncertainty intervals
- Validation of the selected tuned configuration
- Schema validation, leakage warnings, and stronger data-quality checks
- Measured inference latency and clearer selection-score explanations

### Explanations and responsible ML

- Feature importance, coefficients, and decision-tree rules
- Optional local explanations such as SHAP
- Per-class and protected-group metrics, with documented fairness definitions
- Separate protected attributes for auditing rather than blindly using them as features
- Baseline comparisons and controlled measurement of preprocessing improvements

### Usability and lifecycle

- Save/load of preprocessing and model together with environment metadata
- Exportable HTML reports and plots
- A CLI or guided interface for non-programmers
- Stable benchmark datasets for update comparisons, drift checks, and rollback
- Prediction service examples and deployment guidance

### Broader modeling

- Additional clustering methods and distance choices
- Budgeted hyperparameter optimization
- Date/text feature engineering and feature selection
- Dedicated forecasting and other specialized problem workflows

## Project structure

| Path | Purpose |
|---|---|
| `mlforge/automl.py` | Automatic orchestration, detection, result object, clustering |
| `mlforge/problem.py` | Problem contract and relevancy preferences |
| `mlforge/data.py` | Data-source wrappers |
| `mlforge/preprocessing.py` | Fitted feature transformations |
| `mlforge/eda.py` | Descriptive summaries and optional plots |
| `mlforge/models.py` | Supervised model registry |
| `mlforge/evaluation.py` | Candidate metrics and validation |
| `mlforge/selector.py` | Eligibility and weighted model selection |
| `mlforge/pipeline.py` | Explicit supervised workflow |
| `mlforge/reporting.py` | Formatted text reports |
| `mlforge/incremental.py` | Updates and version metadata |
| `examples/example_usage.py` | Worked supervised and update example |
| `tests/` | pytest suite |
| `pyproject.toml` | Package configuration and dependencies |

## Testing

From the project directory with the environment activated:

```bash
python -m pip install pytest
python -m pytest -q
```

Run the included worked example:

```bash
python examples/example_usage.py
```

The tests cover basic supervised workflows, reports, constraints, updates,
automatic task detection, clustering, and some invalid inputs. Passing them
does not prove correctness on every dataset or production readiness.

## License and status

The current package version is **0.1.0**. Install from this source checkout;
this README does not assume a published PyPI package or a public repository URL.

No license is declared in the current package metadata. Confirm licensing with
the project owner before redistribution or reuse requiring explicit permission.

---

**Start simple:** load a DataFrame, call `AutoML.run()`, read the report, verify
the inferred target and task, then predict. Add domain knowledge and independent
evaluation before trusting the result.
