# Agribusiness Data Analysis — Week 2

This repository implements a practical data-cleaning and preprocessing workflow for agribusiness CSV datasets. Week 2 builds on the project objective by inspecting, cleaning, validating, and documenting common market-price and production data issues.

> **Demonstration data:** When no CSV is present in `data/raw/` and no `--input` is supplied, the application uses a clearly labelled synthetic dataset. It is hypothetical and must not be interpreted as actual agricultural observations. Use a real public dataset by explicitly providing its CSV file.

## Week 2 objectives

- Inspect dataset shape, columns, types, missing values, duplicates, and numeric summaries.
- Standardize column names; trim and optionally normalize crop and market labels.
- Parse dates and convert common price, arrivals, yield, production, and quantity fields.
- Impute missing numeric values with medians and categorical values with `Unknown` by default.
- Validate non-negative values, price ranges, category values, and required fields.
- Flag invalid dates, suspicious modal prices, possible duplicate market observations, and IQR outliers.
- Create reusable scikit-learn preprocessing and chronological-splitting helpers.
- Save cleaned data, quality/validation reports, an audit log, an outlier review file, and charts.

## Data quality issues and responsible handling

The workflow handles inconsistent headers, surrounding whitespace, configurable crop/market name variants, missing values, invalid dates and numbers, exact duplicates, negative values, minimum/maximum price inconsistencies, modal prices outside the price range, and potential IQR outliers.

Exact duplicate rows are removed after normalization. Possible duplicate market observations and outliers are reported, **not automatically removed**. Agricultural data can contain genuine extreme weather or price events; review source context before excluding any observation. Numeric median imputation can also hide the original missingness in the cleaned CSV, so the input inspection and cleaning log retain counts of missing and invalid values.

The default validation required-field list is `date`, `crop`, and `market`. Missing required fields are reported in JSON but do not make optional source columns a fatal error. Price, arrivals, and yield fields are optional.

## Workflow

1. Read an explicitly supplied CSV, or automatically use the first `data/raw/*.csv`.
2. If no raw CSV exists and no `--input` was given, generate an in-memory synthetic demonstration dataset.
3. Inspect a copy of the input; raw files are never modified.
4. Standardize headers, normalize text, parse/flag dates, convert numeric fields, and record quality issues.
5. Remove exact duplicates; preserve possible duplicates and outliers for review.
6. Generate validation and cleaning reports, cleaned CSV, and plots.
7. For model development, split data first; fit the returned preprocessing pipeline on training data only, then transform test data. Use chronological splitting for time series.

The reusable helpers in `src/preprocessing.py` provide numerical/categorical imputers, one-hot encoding, `StandardScaler`, `MinMaxScaler`, `RobustScaler`, date-derived month/quarter fields, and random or chronological train/test splits.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

Provide a CSV:

```bash
python main.py --input data/raw/market_prices.csv
```

Place a CSV in `data/raw/` and omit `--input` to use the first matching file:

```bash
python main.py
```

With no raw CSV and no `--input`, a clearly labelled synthetic sample is generated for demonstration. A supplied missing file or unsupported extension produces a clear error. Reports can be written elsewhere:

```bash
python main.py --input data/raw/market_prices.csv --output-dir outputs/run_1
```

## Example input format

Column headers are case-insensitive and are standardized to snake_case. Supported fields are optional; typical columns include:

```csv
date,crop,market,variety,min_price,max_price,modal_price,arrivals,yield
2025-01-05,Rice,North Mandi,Common,1800,2200,2000,120,3.1
2025-01-12,Wheat,Central Market,Durum,2100,2600,2350,95,3.4
```

For custom crop/market normalization, pass mapping dictionaries via `CleaningConfig` in Python, for example:

```python
from src.data_cleaning import CleaningConfig, clean_data

cleaned, report = clean_data(
    frame,
    CleaningConfig(crop_mappings={"paddy": "rice", "maize": "corn"}),
)
```

## Outputs

The default output location is `outputs/`:

| File | Description |
|---|---|
| `cleaned_data.csv` | Cleaned dataset; includes flags for invalid dates, invalid numeric inputs, and price checks where applicable. |
| `input_inspection.json` | Input dimensions, columns, types, missing counts, duplicates, and numeric summaries. |
| `validation_report.json` | Required fields, types, missing percentages, duplicates, invalid values, price checks, categories, and IQR counts. |
| `cleaning_log.json` | Source identity and auditable list of transformations and flagged records. |
| `quality_summary.json` | Before/after quality summaries and row/outlier counts. |
| `outlier_review.csv` | IQR outliers identified for human review; no outlier is deleted by this process. |
| `dataset_metadata.json` | Dataset source and synthetic-data indicator. |
| `charts/missing_values.png` | Missing-value counts before and after cleaning. |
| `charts/data_quality_summary.png` | Before/after rows, missing values, and exact-duplicate counts. |
| `charts/price_distribution.png` | Box plots for available numeric fields (including price fields where present). |

Generated outputs and local raw/processed datasets are ignored by Git. Do not commit confidential data, credentials, or large datasets.

## Preprocessing and leakage prevention

```python
from src.preprocessing import build_preprocessor, split_data

x_train, x_test, y_train, y_test = split_data(
    cleaned_frame, target_column="modal_price", time_series=True
)
preprocessor = build_preprocessor(
    numeric_columns=["min_price", "max_price", "arrivals"],
    categorical_columns=["crop", "market"],
    scaler="robust",  # "standard", "minmax", "robust", or "none"
)
x_train_ready = preprocessor.fit_transform(x_train)
x_test_ready = preprocessor.transform(x_test)
```

Fit only on training data. `time_series=True` sorts by date and reserves the latest observations for testing; every row needs a valid date in that mode. This prevents future observations from influencing training transformations.

## Tests

```bash
pytest -q
```

Tests use small in-memory synthetic frames and do not require private or external datasets.

## Limitations

- Input support is CSV only; schemas vary, and all columns other than recognized dates/numerics are treated as categorical/text.
- Crop and market mappings are intentionally configurable rather than exhaustive.
- IQR is a screening heuristic, not proof of erroneous data; flagged extremes require domain review.
- Median imputation is a baseline and may not be suitable for every feature or forecasting task.
- The CLI reports missing required fields but supports optional columns and does not assume one universal schema.
- The synthetic demonstration exists only to exercise the workflow, not to represent real markets or prices.

## Public data references

Potential sources for real datasets (check source licensing, metadata, and definitions before use):

- [Open Government Data Platform India](https://data.gov.in/)
- [Directorate of Economics and Statistics, Department of Agriculture and Farmers Welfare](https://desagri.gov.in/)
- [India Meteorological Department](https://mausam.imd.gov.in/)
