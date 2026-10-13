"""
Step 4: Exploratory Data Analysis.

Produces a structured, programmatic EDA report (dict + optional saved
plots) rather than a notebook dump, so it can be inspected, logged, or
fed into decisions elsewhere in the pipeline (e.g. flagging class
imbalance so the selector picks an appropriate metric).
"""
from __future__ import annotations
import os
from typing import Optional

import pandas as pd


class EDAReport:
    def __init__(self, df: pd.DataFrame, target: Optional[str] = None):
        self.df = df
        self.target = target
        self.report: dict = {}

    def run(self) -> dict:
        df = self.df
        self.report["shape"] = df.shape
        self.report["dtypes"] = df.dtypes.astype(str).to_dict()
        self.report["missing_pct"] = (df.isna().mean() * 100).round(2).to_dict()
        numeric_df = df.select_dtypes(include="number")
        self.report["numeric_summary"] = (
            numeric_df.describe().to_dict() if not numeric_df.empty else {}
        )

        cat_cols = df.select_dtypes(exclude="number").columns.tolist()
        self.report["categorical_cardinality"] = {
            c: int(df[c].nunique()) for c in cat_cols
        }

        if numeric_df.shape[1] > 1:
            self.report["correlation_matrix"] = numeric_df.corr().round(3).to_dict()

        if self.target and self.target in df.columns:
            self.report["target_distribution"] = self._target_distribution()

        return self.report

    def _target_distribution(self) -> dict:
        s = self.df[self.target]
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 20:
            return {
                "kind": "continuous",
                "mean": float(s.mean()),
                "std": float(s.std()),
                "min": float(s.min()),
                "max": float(s.max()),
            }
        counts = s.value_counts(normalize=True).round(3).to_dict()
        imbalance_ratio = None
        if len(counts) >= 2:
            vals = sorted(counts.values())
            imbalance_ratio = round(vals[-1] / vals[0], 2) if vals[0] > 0 else None
        return {
            "kind": "categorical",
            "class_proportions": counts,
            "imbalance_ratio": imbalance_ratio,
        }

    def save_plots(self, output_dir: str = "eda_plots") -> list:
        """Optional: saves a handful of standard plots as PNGs. Requires matplotlib."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(output_dir, exist_ok=True)
        saved = []

        numeric_df = self.df.select_dtypes(include="number")
        if numeric_df.shape[1] > 1:
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(numeric_df.corr(), cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_xticks(range(len(numeric_df.columns)))
            ax.set_yticks(range(len(numeric_df.columns)))
            ax.set_xticklabels(numeric_df.columns, rotation=90)
            ax.set_yticklabels(numeric_df.columns)
            fig.colorbar(im)
            path = os.path.join(output_dir, "correlation_heatmap.png")
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
            saved.append(path)

        if self.target and self.target in self.df.columns:
            fig, ax = plt.subplots()
            self.df[self.target].value_counts().plot(kind="bar", ax=ax)
            ax.set_title(f"Distribution of {self.target}")
            path = os.path.join(output_dir, "target_distribution.png")
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
            saved.append(path)

        return saved
