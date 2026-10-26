"""Data inspection and cleaning helpers for agribusiness CSV files."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


KNOWN_NUMERIC_COLUMNS = {
    "min_price",
    "minimum_price",
    "max_price",
    "maximum_price",
    "modal_price",
    "arrivals",
    "yield",
    "production",
    "quantity",
}
DATE_COLUMNS = {"date", "arrival_date", "market_date"}


@dataclass
class CleaningConfig:
    """Options controlling category normalization and missing-value handling."""

    crop_mappings: dict[str, str] = field(default_factory=dict)
    market_mappings: dict[str, str] = field(default_factory=dict)
    numeric_imputation: str | None = "median"
    categorical_imputation: str | None = "Unknown"
    numeric_columns: tuple[str, ...] = ()


def standardize_column_name(name: Any) -> str:
    """Convert a column label to lowercase snake_case."""
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name).strip())
    value = re.sub(r"[^0-9a-zA-Z]+", "_", value.lower())
    return value.strip("_")


def standardize_column_names(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with standardized names; reject ambiguous name collisions."""
    result = data.copy()
    names = [standardize_column_name(name) for name in result.columns]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(
            "Column names collide after standardization: " + ", ".join(duplicates)
        )
    result.columns = names
    return result


def inspect_data(data: pd.DataFrame) -> dict[str, Any]:
    """Summarize shape, types, nulls, duplicates, and numeric columns."""
    numeric = data.select_dtypes(include="number")
    raw_summary = numeric.describe().to_dict()
    summary = {
        str(column): {
            str(statistic): None if pd.isna(value) else float(value)
            for statistic, value in statistics.items()
        }
        for column, statistics in raw_summary.items()
    }
    return {
        "shape": {"rows": int(data.shape[0]), "columns": int(data.shape[1])},
        "columns": list(data.columns),
        "data_types": {str(column): str(dtype) for column, dtype in data.dtypes.items()},
        "missing_values": {
            str(column): int(count)
            for column, count in data.isna().sum().items()
        },
        "duplicate_rows": int(data.duplicated().sum()),
        "numerical_summary": summary,
    }


def _normalize_text(data: pd.DataFrame, config: CleaningConfig) -> None:
    for column in data.select_dtypes(include=["object", "string"]).columns:
        values = data[column].astype("string").str.strip()
        values = values.replace("", pd.NA)
        mappings: dict[str, str] = {}
        if column == "crop":
            mappings = config.crop_mappings
        elif column == "market":
            mappings = config.market_mappings
        normalized_mappings = {
            str(key).strip().casefold(): str(value).strip()
            for key, value in mappings.items()
        }
        if normalized_mappings:
            values = values.map(
                lambda value: normalized_mappings.get(value.casefold(), value)
                if pd.notna(value)
                else value
            )
        data[column] = values


def _price_columns(data: pd.DataFrame) -> tuple[str | None, str | None, str | None]:
    minimum = next(
        (column for column in ("min_price", "minimum_price") if column in data),
        None,
    )
    maximum = next(
        (column for column in ("max_price", "maximum_price") if column in data),
        None,
    )
    modal = "modal_price" if "modal_price" in data else None
    return minimum, maximum, modal


def _outlier_records(data: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for column in columns:
        values = pd.to_numeric(data[column], errors="coerce")
        present = values.dropna()
        if present.empty:
            continue
        q1, q3 = present.quantile([0.25, 0.75])
        spread = q3 - q1
        lower, upper = q1 - 1.5 * spread, q3 + 1.5 * spread
        for index, value in values[(values < lower) | (values > upper)].items():
            records.append(
                {"row_index": str(index), "column": column, "value": float(value)}
            )
    return records


def clean_data(
    data: pd.DataFrame, config: CleaningConfig | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean a copy of a dataset and return it with an auditable transformation log.

    Outliers and possible duplicate observations are reported for review, not removed.
    Exact duplicate rows are removed after column and text normalization.
    """
    config = config or CleaningConfig()
    if config.numeric_imputation not in (None, "median"):
        raise ValueError("numeric_imputation must be 'median' or None")
    if config.categorical_imputation not in (None, "Unknown"):
        raise ValueError("categorical_imputation must be 'Unknown' or None")

    cleaned = standardize_column_names(data)
    rows_before = len(cleaned)
    log: list[dict[str, Any]] = [
        {"transformation": "standardize_column_names", "columns": list(cleaned.columns)}
    ]
    _normalize_text(cleaned, config)
    log.append({"transformation": "trim_text_and_apply_category_mappings"})

    invalid_dates: dict[str, int] = {}
    for column in sorted(DATE_COLUMNS.intersection(cleaned.columns)):
        original = cleaned[column]
        parsed = pd.to_datetime(original, errors="coerce", format="mixed")
        invalid = original.notna() & parsed.isna()
        cleaned[f"invalid_{column}"] = invalid.astype(bool)
        invalid_dates[column] = int(invalid.sum())
        cleaned[column] = parsed
    if invalid_dates:
        log.append({"transformation": "parse_dates", "invalid_values": invalid_dates})

    numeric_columns = set(config.numeric_columns) | KNOWN_NUMERIC_COLUMNS
    numeric_columns |= {
        column
        for column in cleaned.columns
        if is_numeric_dtype(cleaned[column])
        and not pd.api.types.is_bool_dtype(cleaned[column])
        and not column.startswith("invalid_")
    }
    numeric_columns = {
        standardize_column_name(column)
        for column in numeric_columns
        if standardize_column_name(column) in cleaned.columns
    }
    invalid_numeric: dict[str, int] = {}
    for column in sorted(numeric_columns):
        original = cleaned[column]
        converted = pd.to_numeric(original, errors="coerce")
        invalid = original.notna() & converted.isna()
        if invalid.any():
            cleaned[f"invalid_{column}"] = invalid.astype(bool)
            invalid_numeric[column] = int(invalid.sum())
        cleaned[column] = converted
    if numeric_columns:
        log.append(
            {
                "transformation": "convert_numeric_fields",
                "columns": sorted(numeric_columns),
                "invalid_values": invalid_numeric,
            }
        )

    minimum, maximum, modal = _price_columns(cleaned)
    price_range_invalid = pd.Series(False, index=cleaned.index)
    if minimum and maximum:
        price_range_invalid = (
            cleaned[minimum].notna()
            & cleaned[maximum].notna()
            & (cleaned[minimum] > cleaned[maximum])
        )
        cleaned["price_range_invalid"] = price_range_invalid.astype(bool)
    if modal and minimum and maximum:
        cleaned["modal_price_suspicious"] = (
            cleaned[modal].notna()
            & cleaned[minimum].notna()
            & cleaned[maximum].notna()
            & ((cleaned[modal] < cleaned[minimum]) | (cleaned[modal] > cleaned[maximum]))
        ).astype(bool)

    exact_duplicates = int(cleaned.duplicated().sum())
    if exact_duplicates:
        cleaned = cleaned.drop_duplicates().copy()
    log.append(
        {
            "transformation": "remove_exact_duplicates",
            "rows_removed": exact_duplicates,
        }
    )

    keys = [column for column in ("date", "crop", "market", "variety") if column in cleaned]
    possible_duplicate_rows = (
        int(cleaned.duplicated(subset=keys, keep=False).sum()) if keys else 0
    )
    if keys:
        log.append(
            {
                "transformation": "review_possible_duplicate_observations",
                "key_columns": keys,
                "rows_flagged": possible_duplicate_rows,
                "rows_removed": 0,
            }
        )

    if config.numeric_imputation == "median":
        imputed: dict[str, int] = {}
        for column in sorted(numeric_columns):
            missing = int(cleaned[column].isna().sum())
            median = cleaned[column].median()
            if missing and pd.notna(median):
                cleaned[column] = cleaned[column].fillna(median)
                imputed[column] = missing
        if imputed:
            log.append({"transformation": "median_imputation", "values_filled": imputed})

    if config.categorical_imputation == "Unknown":
        imputed_categories: dict[str, int] = {}
        for column in cleaned.select_dtypes(include=["object", "string"]).columns:
            missing = int(cleaned[column].isna().sum())
            if missing:
                cleaned[column] = cleaned[column].fillna("Unknown")
                imputed_categories[column] = missing
        if imputed_categories:
            log.append(
                {
                    "transformation": "categorical_imputation",
                    "values_filled": imputed_categories,
                }
            )

    outlier_columns = [
        column
        for column in sorted(numeric_columns)
        if column not in {"date"}
    ]
    outliers = _outlier_records(cleaned, outlier_columns)
    log.append(
        {
            "transformation": "flag_iqr_outliers_for_review",
            "outlier_count": len(outliers),
            "rows_removed": 0,
        }
    )
    report = {
        "rows_before": int(rows_before),
        "rows_after": int(len(cleaned)),
        "exact_duplicates_removed": exact_duplicates,
        "possible_duplicate_rows": possible_duplicate_rows,
        "invalid_dates": invalid_dates,
        "invalid_numeric_values": invalid_numeric,
        "price_range_invalid_rows": int(price_range_invalid.sum()),
        "outlier_count": len(outliers),
        "outliers": outliers,
        "cleaning_log": log,
    }
    return cleaned, report


def read_csv(path: str | Path) -> pd.DataFrame:
    """Read a CSV file with clear errors for unsupported or malformed inputs."""
    input_path = Path(path)
    if input_path.suffix.lower() != ".csv":
        raise ValueError(f"Unsupported input format: {input_path.suffix or '(no extension)'}; use CSV.")
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV was not found: {input_path}")
    try:
        data = pd.read_csv(input_path)
    except (UnicodeDecodeError, pd.errors.ParserError, OSError) as error:
        raise ValueError(f"Could not read CSV file '{input_path}': {error}") from error
    if len(data.columns) == 0:
        raise ValueError(f"Input CSV contains no columns: {input_path}")
    return data
