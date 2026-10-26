"""Reusable, leakage-aware preprocessing helpers."""

from __future__ import annotations

from typing import Literal

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)
from sklearn.model_selection import train_test_split


ScalerName = Literal["standard", "minmax", "robust", "none"]


def add_date_features(data: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
    """Add month and quarter columns while keeping the original date column."""
    if date_column not in data.columns:
        raise ValueError(f"Date column '{date_column}' is not present.")
    result = data.copy()
    dates = pd.to_datetime(result[date_column], errors="coerce")
    result["month"] = dates.dt.month
    result["quarter"] = dates.dt.quarter
    return result


def build_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
    *,
    scaler: ScalerName = "standard",
) -> ColumnTransformer:
    """Build imputing/scaling and one-hot-encoding transforms for model features."""
    scalers = {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "robust": RobustScaler,
        "none": None,
    }
    if scaler not in scalers:
        raise ValueError("scaler must be 'standard', 'minmax', 'robust', or 'none'")

    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    scaler_class = scalers[scaler]
    if scaler_class is not None:
        numeric_steps.append(("scaler", scaler_class()))
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_columns:
        transformers.append(("numeric", Pipeline(numeric_steps), numeric_columns))
    if categorical_columns:
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_columns))
    if not transformers:
        raise ValueError("At least one numeric or categorical feature column is required.")
    return ColumnTransformer(transformers=transformers, remainder="drop")


def split_data(
    data: pd.DataFrame,
    target_column: str,
    *,
    test_size: float = 0.2,
    time_series: bool = False,
    date_column: str = "date",
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split rows before fitting preprocessing; time-series mode preserves chronology."""
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' is not present.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    ordered = data
    if time_series:
        if date_column not in data.columns:
            raise ValueError(f"Date column '{date_column}' is required for a time-series split.")
        dates = pd.to_datetime(data[date_column], errors="coerce")
        if dates.isna().any():
            raise ValueError("Time-series split requires valid dates for every row.")
        ordered = data.assign(_split_date=dates).sort_values("_split_date").drop(columns="_split_date")
        split_at = int(len(ordered) * (1 - test_size))
        if split_at < 1 or split_at >= len(ordered):
            raise ValueError("Not enough rows for the requested time-series split.")
        train, test = ordered.iloc[:split_at], ordered.iloc[split_at:]
    else:
        train, test = train_test_split(
            ordered, test_size=test_size, random_state=random_state
        )
    return (
        train.drop(columns=[target_column]),
        test.drop(columns=[target_column]),
        train[target_column],
        test[target_column],
    )
