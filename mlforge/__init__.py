"""
mlforge — an opinionated, end-to-end ML lifecycle library.

Formulate the problem -> collect/load data -> preprocess -> EDA ->
train multiple candidate algorithms -> evaluate on accuracy AND
relevancy-to-the-problem -> select + finalize a model -> keep
improving it as new data arrives.
"""

from .problem import ProblemDefinition, RelevancyCriteria
from .data import DataLoader
from .preprocessing import Preprocessor
from .eda import EDAReport
from .models import ModelZoo
from .evaluation import Evaluator
from .selector import ModelSelector
from .pipeline import MLPipeline
from .incremental import IncrementalLearner
from .automl import AutoML, AutoMLResult

__all__ = [
    "ProblemDefinition",
    "RelevancyCriteria",
    "DataLoader",
    "Preprocessor",
    "EDAReport",
    "ModelZoo",
    "Evaluator",
    "ModelSelector",
    "MLPipeline",
    "IncrementalLearner",
    "AutoML",
    "AutoMLResult",
]

__version__ = "0.1.0"
