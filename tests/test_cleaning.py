import pandas as pd
import pytest

from src.data_cleaning import CleaningConfig, clean_data, standardize_column_names
from src.data_validation import validate_data
from src.preprocessing import add_date_features, build_preprocessor, split_data


def test_standardizes_column_names_and_rejects_collisions():
    result = standardize_column_names(pd.DataFrame({"Minimum Price": [1], "Crop Name": ["Rice"]}))
    assert list(result.columns) == ["minimum_price", "crop_name"]
    with pytest.raises(ValueError, match="collide"):
        standardize_column_names(pd.DataFrame({"A B": [1], "a-b": [2]}))


def test_missing_numeric_and_categorical_values_are_imputed():
    source = pd.DataFrame(
        {"crop": ["Rice", None, "Wheat"], "modal_price": [10, None, 30]}
    )
    result, _ = clean_data(source)
    assert result["crop"].tolist() == ["Rice", "Unknown", "Wheat"]
    assert result["modal_price"].tolist() == [10, 20, 30]
    assert source.isna().sum().sum() == 2


def test_exact_duplicates_are_removed_and_possible_observations_are_reviewed():
    source = pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-01", "2025-01-01"],
            "crop": ["Rice", "Rice", "Rice"],
            "market": ["A", "A", "A"],
            "modal_price": [10, 10, 11],
        }
    )
    cleaned, report = clean_data(source)
    assert len(cleaned) == 2
    assert report["exact_duplicates_removed"] == 1
    assert report["possible_duplicate_rows"] == 2


def test_date_parsing_flags_invalid_dates():
    cleaned, report = clean_data(pd.DataFrame({"date": ["2025-01-01", "bad"]}))
    assert pd.notna(cleaned.loc[0, "date"])
    assert pd.isna(cleaned.loc[1, "date"])
    assert cleaned["invalid_date"].tolist() == [False, True]
    assert report["invalid_dates"] == {"date": 1}


def test_numeric_conversion_and_invalid_values_are_reported():
    cleaned, report = clean_data(
        pd.DataFrame({"arrivals": ["12", "not-a-number", "18"]}),
        CleaningConfig(numeric_imputation=None),
    )
    assert pd.api.types.is_numeric_dtype(cleaned["arrivals"])
    assert cleaned["arrivals"].iloc[0] == 12
    assert cleaned["invalid_arrivals"].tolist() == [False, True, False]
    assert report["invalid_numeric_values"] == {"arrivals": 1}


def test_price_validation_and_suspicious_modal_price():
    cleaned, report = clean_data(
        pd.DataFrame(
            {
                "min_price": [20, 10],
                "max_price": [10, 30],
                "modal_price": [15, 40],
            }
        )
    )
    assert cleaned["price_range_invalid"].tolist() == [True, False]
    assert cleaned["modal_price_suspicious"].tolist() == [False, True]
    validation = validate_data(cleaned)
    assert validation["price_range"]["minimum_greater_than_maximum"] == 1
    assert validation["price_range"]["modal_outside_minimum_maximum"] == 1
    assert report["price_range_invalid_rows"] == 1


def test_iqr_outliers_are_reported_but_not_deleted():
    source = pd.DataFrame({"arrivals": [1, 2, 2, 3, 100]})
    cleaned, report = clean_data(source)
    assert len(cleaned) == len(source)
    assert report["outlier_count"] == 1
    assert report["outliers"][0]["value"] == 100


def test_optional_columns_and_empty_dataset_are_supported():
    cleaned, report = clean_data(pd.DataFrame(columns=["Crop", "Modal Price"]))
    assert list(cleaned.columns) == ["crop", "modal_price"]
    assert report["rows_before"] == report["rows_after"] == 0
    validation = validate_data(cleaned)
    assert validation["required_fields"]["missing"] == ["date", "crop", "market"]


def test_category_mapping_is_configurable():
    cleaned, _ = clean_data(
        pd.DataFrame({"crop": [" paddy "]}),
        CleaningConfig(crop_mappings={"paddy": "rice"}),
    )
    assert cleaned.loc[0, "crop"] == "rice"


def test_preprocessing_features_and_chronological_split():
    data = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=5),
            "value": [1, 2, 3, 4, 5],
            "crop": ["Rice", "Rice", "Wheat", "Rice", "Wheat"],
            "target": [2, 4, 6, 8, 10],
        }
    )
    featured = add_date_features(data)
    assert featured["month"].tolist() == [1] * 5
    assert featured["quarter"].tolist() == [1] * 5
    x_train, x_test, _, _ = split_data(
        featured, "target", test_size=0.4, time_series=True
    )
    assert x_train["date"].max() < x_test["date"].min()
    preprocessor = build_preprocessor(["value"], ["crop"], scaler="robust")
    transformed_train = preprocessor.fit_transform(x_train[["value", "crop"]])
    transformed_test = preprocessor.transform(x_test[["value", "crop"]])
    assert transformed_train.shape[1] == transformed_test.shape[1]
