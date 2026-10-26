"""Command-line entry point for the Week 2 agribusiness data workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.data_cleaning import CleaningConfig, clean_data, inspect_data, read_csv
from src.data_validation import validate_data


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"


def synthetic_sample() -> pd.DataFrame:
    """Return explicitly hypothetical data for a self-contained demonstration."""
    return pd.DataFrame(
        {
            "Date": [
                "2025-01-05",
                "2025-01-12",
                "2025-02-03",
                "2025-02-18",
                "not-a-date",
                "2025-03-09",
            ],
            "Crop": [" Paddy ", "Rice", "Maize", "maize", None, "Wheat"],
            "Market": ["North Mandi", "North Mandi", "East Market", "East Market", "South Mandi", "Central Market"],
            "Variety": ["Common", "Common", "Hybrid", "Hybrid", "Local", "Durum"],
            "Minimum Price": ["1800", "1820", "2100", "2050", "1900", "2200"],
            "Maximum Price": ["2200", "2210", "2500", "2480", "2300", "2700"],
            "Modal Price": ["2000", "2010", "2300", "2320", "2100", "2400"],
            "Arrivals": [120, 135, 90, 95, 0, 75],
            "Yield": [3.1, 3.2, 2.8, 2.9, 3.0, 3.4],
            "Data Origin": ["Synthetic demonstration"] * 6,
        }
    )


def _json_default(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(data, indent=2, default=_json_default, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def create_charts(before: pd.DataFrame, after: pd.DataFrame, charts_dir: Path) -> None:
    charts_dir.mkdir(parents=True, exist_ok=True)
    missing_before = before.isna().sum()
    missing_after = after.isna().sum().reindex(missing_before.index, fill_value=0)
    missing_summary = pd.DataFrame({"Before cleaning": missing_before, "After cleaning": missing_after})
    axis = missing_summary.plot(kind="bar", figsize=(max(8, len(missing_summary) * 0.8), 5))
    axis.set_title("Missing values before and after cleaning")
    axis.set_ylabel("Number of missing values")
    axis.set_xlabel("Column")
    axis.tick_params(axis="x", rotation=45)
    axis.figure.tight_layout()
    axis.figure.savefig(charts_dir / "missing_values.png", dpi=150)
    plt.close(axis.figure)

    numeric_columns = after.select_dtypes(include="number").columns
    if len(numeric_columns):
        axis = after[numeric_columns].plot(
            kind="box", subplots=True, layout=(-1, 1), figsize=(8, max(4, 3 * len(numeric_columns))),
            sharex=False, sharey=False,
        )
        figure = axis[0][0].figure if getattr(axis, "ndim", 1) > 1 else axis.figure
        figure.suptitle("Numeric feature distributions after cleaning")
        figure.tight_layout()
        figure.savefig(charts_dir / "price_distribution.png", dpi=150)
        plt.close(figure)
    else:
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.text(0.5, 0.5, "No numeric fields available", ha="center", va="center")
        axis.set_axis_off()
        figure.savefig(charts_dir / "price_distribution.png", dpi=150)
        plt.close(figure)

    quality = pd.DataFrame(
        {
            "Before cleaning": [
                len(before),
                int(before.isna().sum().sum()),
                int(before.duplicated().sum()),
            ],
            "After cleaning": [
                len(after),
                int(after.isna().sum().sum()),
                int(after.duplicated().sum()),
            ],
        },
        index=["Rows", "Missing values", "Exact duplicates"],
    )
    axis = quality.plot(kind="bar", figsize=(7, 5))
    axis.set_title("Before-and-after data quality summary")
    axis.set_ylabel("Count")
    axis.tick_params(axis="x", rotation=0)
    axis.figure.tight_layout()
    axis.figure.savefig(charts_dir / "data_quality_summary.png", dpi=150)
    plt.close(axis.figure)


def run(input_path: Path | None, output_dir: Path = OUTPUT_DIR) -> int:
    if input_path is None:
        raw_files = sorted((ROOT / "data" / "raw").glob("*.csv"))
        if raw_files:
            input_path = raw_files[0]
            print(f"No --input supplied; using {input_path}")
        else:
            before = synthetic_sample()
            source = "synthetic demonstration (generated in memory; not real observations)"
            print("No raw CSV found. Using a clearly labelled synthetic demonstration dataset.")
    else:
        source = str(input_path)
    if input_path is not None:
        before = read_csv(input_path)
        source = str(input_path)

    if before.empty:
        print("Warning: input dataset contains zero rows; reports will be generated.")

    before_inspection = inspect_data(before)
    cleaned, cleaning_report = clean_data(before, CleaningConfig())
    validation = validate_data(
        cleaned,
        rows_before=len(before),
        source_duplicates_removed=cleaning_report["exact_duplicates_removed"],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "charts").mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_dir / "cleaned_data.csv", index=False)
    write_json(output_dir / "input_inspection.json", before_inspection)
    write_json(output_dir / "validation_report.json", validation)
    write_json(
        output_dir / "cleaning_log.json",
        {"source": source, **cleaning_report},
    )
    write_json(
        output_dir / "quality_summary.json",
        {
            "before": before_inspection,
            "after": inspect_data(cleaned),
            "rows_removed": len(before) - len(cleaned),
            "outlier_count": cleaning_report["outlier_count"],
        },
    )
    pd.DataFrame(cleaning_report["outliers"], columns=["row_index", "column", "value"]).to_csv(
        output_dir / "outlier_review.csv", index=False
    )
    write_json(
        output_dir / "dataset_metadata.json",
        {"source": source, "synthetic": source.startswith("synthetic demonstration")},
    )
    create_charts(before, cleaned, output_dir / "charts")
    print(f"Input rows: {len(before)}; cleaned rows: {len(cleaned)}")
    print(f"Outputs written to: {output_dir}")
    print(f"Outliers flagged for review (not removed): {cleaning_report['outlier_count']}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean, validate, and report on agribusiness CSV data."
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="CSV file to clean. Without it, use the first data/raw/*.csv or a labelled synthetic demo.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Directory for reports, cleaned data, and charts (default: {OUTPUT_DIR})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        arguments = parse_args()
        raise SystemExit(run(arguments.input, arguments.output_dir))
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error
