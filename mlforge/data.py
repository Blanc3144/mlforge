"""
Step 2: Data collection / loading.

Kept intentionally simple and pluggable — the point isn't to reinvent
pandas, it's to give the pipeline one consistent entry point regardless
of where the data physically comes from.
"""
from __future__ import annotations
import pandas as pd


class DataLoader:
    """Thin, uniform wrapper around common data sources."""

    @staticmethod
    def from_csv(path: str, **kwargs) -> pd.DataFrame:
        return pd.read_csv(path, **kwargs)

    @staticmethod
    def from_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()

    @staticmethod
    def from_sql(query: str, connection, **kwargs) -> pd.DataFrame:
        return pd.read_sql(query, connection, **kwargs)

    @staticmethod
    def from_dict(data: dict) -> pd.DataFrame:
        return pd.DataFrame(data)

    @staticmethod
    def basic_report(df: pd.DataFrame) -> dict:
        """Quick sanity report used right after loading."""
        return {
            "n_rows": len(df),
            "n_cols": df.shape[1],
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_per_column": df.isna().sum().to_dict(),
            "duplicate_rows": int(df.duplicated().sum()),
        }
