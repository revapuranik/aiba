"""Prepare analysis results for simple Streamlit charts."""

import pandas as pd


def prepare_chart_data(result, request):
    """Return (chart type, indexed chart data), or None when no chart applies."""
    if not isinstance(result, pd.DataFrame) or "Result" not in result.columns:
        return None

    analysis_type = request.get("analysis_type")
    if analysis_type == "groupby_aggregation":
        group_column = request.get("group_column")
        if group_column in result.columns:
            chart_data = result.set_index(group_column)[["Result"]]
            return "bar", chart_data

    if analysis_type == "time_series" and "Date" in result.columns:
        chart_data = result.set_index("Date")[["Result"]]
        return "line", chart_data

    return None
