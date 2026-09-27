"""Turn a business question into a simple, structured analysis plan."""

import re

import pandas as pd


def normalize_text(value):
    """Make column names and question text easier to compare."""
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def find_column_in_question(question, columns, numeric_only=False, excluded=None):
    """Find a column whose name appears in the question, preferring specific names."""
    normalized_question = f" {normalize_text(question)} "
    candidates = []
    excluded = set(excluded or [])

    for column in columns:
        if column in excluded:
            continue
        if numeric_only and not pd.api.types.is_numeric_dtype(columns[column]):
            continue
        normalized_column = normalize_text(column)
        if normalized_column and f" {normalized_column} " in normalized_question:
            candidates.append((len(normalized_column), column))

    if candidates:
        return max(candidates, key=lambda item: item[0])[1]
    return None


def find_date_column(data, question):
    """Find a likely date column using its dtype, values, and optional name mention."""
    mentioned = find_column_in_question(question, data)
    if mentioned is not None and _looks_like_date(mentioned, data[mentioned]):
        return mentioned

    for column in data.columns:
        if _looks_like_date(column, data[column]):
            return column
    return None


def _looks_like_date(column_name, column):
    if pd.api.types.is_datetime64_any_dtype(column):
        return True

    name = normalize_text(column_name)
    if any(word in name.split() for word in ("date", "time", "timestamp")):
        return True

    if pd.api.types.is_object_dtype(column) or pd.api.types.is_string_dtype(column):
        values = column.dropna()
        if not values.empty:
            parsed = pd.to_datetime(values, errors="coerce")
            return parsed.notna().mean() >= 0.8
    return False


def _aggregation(question):
    text = normalize_text(question)
    if any(word in text.split() for word in ("average", "avg", "mean")):
        return "mean"
    if "median" in text.split():
        return "median"
    if any(word in text.split() for word in ("count", "number")):
        return "count"
    if any(word in text.split() for word in ("maximum", "max")):
        return "max"
    if any(word in text.split() for word in ("minimum", "min")):
        return "min"
    return "sum"


def _group_column(question, data, value_column):
    """Use an explicitly mentioned non-value column, then grouping language clues."""
    mentioned = find_column_in_question(
        question, data, excluded={value_column} if value_column is not None else None
    )
    if mentioned is not None:
        return mentioned

    text = normalize_text(question)
    asks_for_groups = any(phrase in text for phrase in (" by ", " per ", " each "))
    if asks_for_groups:
        for column in data.columns:
            if column == value_column or _looks_like_date(column, data[column]):
                continue
            if not pd.api.types.is_numeric_dtype(data[column]):
                return column
    return None


def plan_analysis(question, data):
    """Return (plan, message); plan is None when the question cannot be planned."""
    if not question or not question.strip():
        return None, "Enter a question about this dataset to create an analysis plan."

    text = normalize_text(question)
    aggregation = _aggregation(question)
    value_column = find_column_in_question(question, data, numeric_only=True)

    asks_time_series = any(
        word in text.split() for word in ("monthly", "daily", "weekly", "yearly", "annual")
    ) or "over time" in text
    if asks_time_series:
        date_column = find_date_column(data, question)
        if date_column is None or value_column is None:
            return None, "I couldn't identify both a date/time column and a numerical value column for a time-series plan."
        if "daily" in text.split():
            frequency = "day"
        elif "weekly" in text.split():
            frequency = "week"
        elif any(word in text.split() for word in ("yearly", "annual")):
            frequency = "year"
        else:
            frequency = "month"
        return {
            "analysis_type": "time_series",
            "date_column": date_column,
            "value_column": value_column,
            "aggregation": aggregation,
            "frequency": frequency,
        }, None

    relationship_words = ("affect", "impact", "relationship", "correlate", "correlation")
    if any(word in text.split() for word in relationship_words):
        numeric_columns = [
            column for column in data.columns
            if pd.api.types.is_numeric_dtype(data[column])
        ]
        if value_column is None or len(numeric_columns) < 2:
            return None, "I couldn't identify two numerical columns to describe their relationship."
        other_columns = [column for column in numeric_columns if column != value_column]
        other_column = find_column_in_question(question, data, numeric_only=True)
        if other_column == value_column:
            other_column = next((column for column in other_columns if normalize_text(column) in text), None)
        if other_column is None:
            other_column = other_columns[0]
        return {
            "analysis_type": "relationship",
            "columns": [other_column, value_column],
            "method": "correlation",
        }, None

    asks_rank = any(word in text.split() for word in ("highest", "lowest", "top", "rank"))
    group_column = _group_column(question, data, value_column)
    if group_column is not None and value_column is not None and asks_rank:
        descending = not any(word in text.split() for word in ("lowest", "bottom"))
        return {
            "analysis_type": "groupby_aggregation",
            "group_column": group_column,
            "value_column": value_column,
            "aggregation": aggregation,
            "sort": "descending" if descending else "ascending",
        }, None

    asks_aggregation = any(
        word in text.split()
        for word in ("average", "avg", "mean", "median", "sum", "total", "count", "maximum", "minimum")
    )
    if asks_aggregation and value_column is not None:
        return {
            "analysis_type": "aggregation",
            "value_column": value_column,
            "aggregation": aggregation,
        }, None

    return None, "I couldn't turn that question into a supported analysis plan. Try asking for an average, a ranking by a group, or a trend over time, and mention the relevant column names."
