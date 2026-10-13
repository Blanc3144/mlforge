"""
Step 3: Data preprocessing.

Builds a scikit-learn ColumnTransformer pipeline automatically from the
dataframe's dtypes, with sensible, overridable defaults. Fit once,
reused consistently for training, testing, and every future incremental
update — this is what prevents train/serve skew.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class Preprocessor:
    numeric_features: Optional[List[str]] = None
    categorical_features: Optional[List[str]] = None
    scale_numeric: bool = True
    _fitted_transformer: Optional[ColumnTransformer] = None
    _fitted_columns: Optional[List[str]] = None

    def infer_feature_types(self, df: pd.DataFrame, target: Optional[str] = None):
        """Infer numeric and categorical columns.

        ``target`` is optional so unsupervised workflows can reuse the same
        train/serve-safe preprocessing as supervised pipelines.  The
        original ``fit_transform(df, target)`` API continues to behave the
        same way when a target is supplied.
        """
        features = [c for c in df.columns if target is None or c != target]
        if self.numeric_features is None:
            self.numeric_features = [
                c for c in features if pd.api.types.is_numeric_dtype(df[c])
            ]
        if self.categorical_features is None:
            self.categorical_features = [
                c for c in features if c not in self.numeric_features
            ]
        return self.numeric_features, self.categorical_features

    def build_transformer(self) -> ColumnTransformer:
        numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
        if self.scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))
        numeric_pipeline = Pipeline(numeric_steps)

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])

        transformer = ColumnTransformer([
            ("num", numeric_pipeline, self.numeric_features),
            ("cat", categorical_pipeline, self.categorical_features),
        ])
        return transformer

    def fit_transform(self, df: pd.DataFrame, target: str):
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("df must be a non-empty pandas DataFrame")
        if target not in df.columns:
            raise ValueError(f"Target column {target!r} is missing from the dataframe")
        self.infer_feature_types(df, target)
        X_out = self._fit_transform_features(df)
        y = df[target]
        return X_out, y

    def fit_transform_features(self, df: pd.DataFrame):
        """Fit preprocessing using every column as a feature.

        This is the unsupervised counterpart to :meth:`fit_transform` and is
        intentionally additive: existing callers that provide a target do
        not need to change.
        """
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("df must be a non-empty pandas DataFrame")
        self.infer_feature_types(df, target=None)
        return self._fit_transform_features(df)

    def _fit_transform_features(self, df: pd.DataFrame):
        feature_columns = list(self.numeric_features or []) + list(self.categorical_features or [])
        if not feature_columns:
            raise ValueError("At least one feature column is required")

        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Feature columns are missing from the dataframe: {missing}")

        transformer = self.build_transformer()
        X = df[feature_columns]
        X_out = transformer.fit_transform(X)
        self._fitted_transformer = transformer
        self._fitted_columns = feature_columns
        return X_out

    def transform(self, df: pd.DataFrame):
        return self.transform_features(df)

    def transform_features(self, df: pd.DataFrame):
        """Transform feature-only data with the fitted transformer."""
        if self._fitted_transformer is None:
            raise RuntimeError("Preprocessor must be fit before calling transform().")
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        missing = [c for c in (self._fitted_columns or []) if c not in df.columns]
        if missing:
            raise ValueError(f"Feature columns are missing from the dataframe: {missing}")
        X = df[self._fitted_columns]
        return self._fitted_transformer.transform(X)
