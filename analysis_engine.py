"""Safely execute supported, structured analysis requests with Pandas."""

import pandas as pd


SUPPORTED_AGGREGATIONS = {"sum", "mean", "median", "min", "max", "count"}


def _require_column(data, column_name, request_field):
    if not isinstance(column_name, str) or column_name not in data.columns:
        raise ValueError(
            f"The requested {request_field} column {column_name!r} was not found in the dataset."
        )


def _numeric_values(series, column_name):
    """Return numeric values, rejecting non-missing entries that are not numbers."""
    numeric = pd.to_numeric(series, errors="coerce")
    invalid_values = series.notna() & numeric.isna()
    if invalid_values.any():
        raise ValueError(
            f"Column {column_name!r} contains values that cannot be used as numbers."
        )
    return numeric


def _get_aggregation(request):
    aggregation = request.get("aggregation")
    if not isinstance(aggregation, str) or aggregation not in SUPPORTED_AGGREGATIONS:
        supported = ", ".join(sorted(SUPPORTED_AGGREGATIONS))
        raise ValueError(f"Unsupported aggregation. Choose one of: {supported}.")
    return aggregation


def _aggregate(series, aggregation):
    if aggregation == "count":
        return series.count()
    return getattr(series, aggregation)()


def _numeric_series(data, column_name, aggregation):
    series = data[column_name]
    if aggregation == "count":
        return series
    return _numeric_values(series, column_name)


def _execute_groupby(data, request):
    group_column = request.get("group_column")
    value_column = request.get("value_column")
    _require_column(data, group_column, "group")
    _require_column(data, value_column, "value")
    if group_column == value_column:
        raise ValueError("Choose different columns for grouping and values.")

    aggregation = _get_aggregation(request)
    values = _numeric_series(data, value_column, aggregation)
    working = pd.DataFrame({group_column: data[group_column], "_analysis_value": values})
    working = working.dropna(subset=[group_column])
    if aggregation != "count":
        working = working.dropna(subset=["_analysis_value"])
    if working.empty:
        raise ValueError("There are no usable rows for this grouped analysis.")

    result = (
        working.groupby(group_column, dropna=True)["_analysis_value"]
        .agg(aggregation)
        .reset_index(name="Result")
    )

    sort_order = request.get("sort")
    if sort_order is not None:
        if not isinstance(sort_order, str) or sort_order not in {"ascending", "descending"}:
            raise ValueError("Sort must be 'ascending' or 'descending'.")
        result = result.sort_values(
            "Result", ascending=(sort_order == "ascending"), kind="stable"
        )

    limit = request.get("limit")
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("Limit must be a positive whole number.")
        result = result.head(limit)
    return result.reset_index(drop=True)


def _execute_aggregation(data, request):
    value_column = request.get("value_column")
    _require_column(data, value_column, "value")
    aggregation = _get_aggregation(request)
    values = _numeric_series(data, value_column, aggregation).dropna()
    if values.empty:
        raise ValueError(f"Column {value_column!r} has no usable values for this aggregation.")
    result = _aggregate(values, aggregation)
    return result.item() if hasattr(result, "item") else result


def _execute_time_series(data, request):
    date_column = request.get("date_column")
    value_column = request.get("value_column")
    _require_column(data, date_column, "date/time")
    _require_column(data, value_column, "value")
    if date_column == value_column:
        raise ValueError("Choose different columns for dates and values.")

    aggregation = _get_aggregation(request)
    frequency = request.get("frequency", "month")
    frequency_codes = {
        "day": "D", "daily": "D",
        "week": "W", "weekly": "W",
        "month": "M", "monthly": "M",
        "quarter": "Q", "quarterly": "Q",
        "year": "Y", "yearly": "Y",
    }
    if not isinstance(frequency, str) or frequency.lower() not in frequency_codes:
        raise ValueError("Frequency must be day, week, month, quarter, or year.")

    if pd.api.types.is_numeric_dtype(data[date_column]):
        raise ValueError(
            f"Date/time column {date_column!r} contains numbers. Use dates or timestamps instead."
        )
    dates = pd.to_datetime(data[date_column], errors="coerce", utc=True)
    if dates.notna().sum() == 0:
        raise ValueError(f"Column {date_column!r} could not be converted to dates.")
    dates = dates.dt.tz_convert(None)

    values = _numeric_series(data, value_column, aggregation)
    working = pd.DataFrame({"Date": dates, "_analysis_value": values})
    working = working.dropna(subset=["Date"])
    if aggregation != "count":
        working = working.dropna(subset=["_analysis_value"])
    if working.empty:
        raise ValueError("There are no usable date and value pairs for this time-series analysis.")

    period_code = frequency_codes[frequency.lower()]
    working["Date"] = working["Date"].dt.to_period(period_code).dt.to_timestamp()
    result = (
        working.groupby("Date", as_index=False)["_analysis_value"]
        .agg(aggregation)
        .rename(columns={"_analysis_value": "Result"})
        .sort_values("Date")
        .reset_index(drop=True)
    )
    return result


def _execute_relationship(data, request):
    """Execute the correlation request already produced by the planner."""
    columns = request.get("columns")
    if not isinstance(columns, list) or len(columns) != 2 or columns[0] == columns[1]:
        raise ValueError("A relationship request must contain two different column names.")
    for column_name in columns:
        _require_column(data, column_name, "relationship")

    if request.get("method") != "correlation":
        raise ValueError("Only correlation is supported for relationship requests.")

    first = _numeric_values(data[columns[0]], columns[0])
    second = _numeric_values(data[columns[1]], columns[1])
    paired_values = pd.DataFrame({columns[0]: first, columns[1]: second}).dropna()
    if len(paired_values) < 2:
        raise ValueError("At least two complete numeric row pairs are needed for correlation.")
    return paired_values.corr()


def execute_analysis(data, request):
    """Validate and execute a supported request; return a scalar or DataFrame.

    Raises ValueError with a user-friendly message when the request or data is invalid.
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Analysis requires a Pandas DataFrame.")
    if not isinstance(request, dict):
        raise ValueError("The analysis request must be a structured object.")

    analysis_type = request.get("analysis_type")
    if analysis_type == "groupby_aggregation":
        return _execute_groupby(data, request)
    if analysis_type == "aggregation":
        return _execute_aggregation(data, request)
    if analysis_type == "time_series":
        return _execute_time_series(data, request)
    if analysis_type == "relationship":
        return _execute_relationship(data, request)
    raise ValueError(f"Unsupported analysis type: {analysis_type!r}.")
