"""Readable quality checks for cleaned agribusiness datasets."""

from __future__ import annotations

from typing import Any

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from src.data_cleaning import KNOWN_NUMERIC_COLUMNS, _price_columns


DEFAULT_REQUIRED_FIELDS = ("date", "crop", "market")
NON_NEGATIVE_FIELDS = {
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


def _iqr_counts(data: pd.DataFrame) -> dict[str, int]:
    results: dict[str, int] = {}
    columns = set(KNOWN_NUMERIC_COLUMNS) | {
        column
        for column in data.columns
        if is_numeric_dtype(data[column])
        and not is_bool_dtype(data[column])
        and not column.startswith(("invalid_", "price_range_", "modal_price_suspicious"))
    }
    for column in sorted(columns.intersection(data.columns)):
        values = pd.to_numeric(data[column], errors="coerce").dropna()
        if values.empty:
            results[column] = 0
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        spread = q3 - q1
        results[column] = int(((values < q1 - 1.5 * spread) | (values > q3 + 1.5 * spread)).sum())
    return results


def validate_data(
    data: pd.DataFrame,
    *,
    rows_before: int | None = None,
    required_fields: tuple[str, ...] = DEFAULT_REQUIRED_FIELDS,
    source_duplicates_removed: int = 0,
) -> dict[str, Any]:
    """Return validation checks without stopping the pipeline on data-quality issues."""
    required_missing = [field for field in required_fields if field not in data.columns]
    missing_percentages = {
        str(column): round(float(data[column].isna().mean() * 100), 2)
        for column in data.columns
    }
    invalid_dates = {
        column.removeprefix("invalid_"): int(data[column].sum())
        for column in data.columns
        if column.startswith("invalid_") and column.removeprefix("invalid_") in {"date", "arrival_date", "market_date"}
    }
    invalid_numeric = {
        column.removeprefix("invalid_"): int(data[column].sum())
        for column in data.columns
        if column.startswith("invalid_") and column.removeprefix("invalid_") not in {"date", "arrival_date", "market_date"}
    }

    negative_values = {
        column: int((pd.to_numeric(data[column], errors="coerce") < 0).sum())
        for column in sorted(NON_NEGATIVE_FIELDS.intersection(data.columns))
    }
    minimum, maximum, modal = _price_columns(data)
    price_range_count = (
        int(
            (
                pd.to_numeric(data[minimum], errors="coerce")
                > pd.to_numeric(data[maximum], errors="coerce")
            ).sum()
        )
        if minimum and maximum
        else 0
    )
    modal_outside_range = 0
    if modal and minimum and maximum:
        modal_outside_range = int(
            (
                (pd.to_numeric(data[modal], errors="coerce") < pd.to_numeric(data[minimum], errors="coerce"))
                | (pd.to_numeric(data[modal], errors="coerce") > pd.to_numeric(data[maximum], errors="coerce"))
            ).sum()
        )
    categories: dict[str, Any] = {}
    for column in ("crop", "market", "variety"):
        if column in data:
            values = data[column].dropna().astype(str)
            categories[column] = {
                "unique_values": sorted(values.unique().tolist()),
                "blank_values": int((values.str.strip() == "").sum()),
            }

    return {
        "row_counts": {
            "before_cleaning": int(rows_before if rows_before is not None else len(data)),
            "after_cleaning": int(len(data)),
        },
        "required_fields": {
            "expected": list(required_fields),
            "missing": required_missing,
            "passed": not required_missing,
        },
        "data_types": {str(column): str(dtype) for column, dtype in data.dtypes.items()},
        "missing_value_percentages": missing_percentages,
        "duplicates": {
            "exact_duplicates_remaining": int(data.duplicated().sum()),
            "exact_duplicates_removed": int(source_duplicates_removed),
        },
        "invalid_dates": invalid_dates,
        "invalid_numeric_values": invalid_numeric,
        "negative_value_counts": negative_values,
        "price_range": {
            "minimum_greater_than_maximum": price_range_count,
            "modal_outside_minimum_maximum": modal_outside_range,
        },
        "category_consistency": categories,
        "outlier_counts_iqr": _iqr_counts(data),
    }
