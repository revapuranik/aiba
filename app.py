"""Streamlit app for uploading and profiling a business dataset."""

import re

import pandas as pd
import streamlit as st
from analysis_planner import plan_analysis
from analysis_engine import execute_analysis
from charting import prepare_chart_data
from config import PLANNER_MODE
from llm_planner import (
    LLMServiceUnavailable,
    PlannerResponseError,
    PlannerValidationError,
    UnsupportedQuestionError,
    plan_with_llm,
    validate_analysis_request,
)
from insight_generator import generate_business_insight


def load_dataset(uploaded_file):
    """Load a CSV or Excel upload into a pandas DataFrame."""
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    if file_name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)

    raise ValueError("Please upload a CSV or Excel file.")


def looks_like_identifier(column_name, column):
    """Use the column name and value uniqueness to spot likely ID columns."""
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(column_name)).lower()
    name_parts = set(re.findall(r"[a-z0-9]+", name))
    id_name = bool(name_parts.intersection({"id", "ids", "identifier", "uuid", "guid"}))

    non_missing = column.dropna()
    if non_missing.empty:
        return id_name

    uniqueness = non_missing.nunique() / len(non_missing)
    # High uniqueness is a useful clue, but short columns alone are not enough.
    looks_unique = len(non_missing) >= 10 and uniqueness >= 0.9
    return id_name or looks_unique


def looks_like_datetime(column_name, column):
    """Recognize date/time dtypes and string columns that parse mostly as dates."""
    if pd.api.types.is_datetime64_any_dtype(column):
        return True

    name = str(column_name).lower()
    name_suggests_date = any(word in name for word in ("date", "time", "timestamp"))
    if not (pd.api.types.is_object_dtype(column) or pd.api.types.is_string_dtype(column)):
        return False

    non_missing = column.dropna()
    if non_missing.empty:
        return name_suggests_date

    parsed = pd.to_datetime(non_missing, errors="coerce")
    parse_ratio = parsed.notna().mean()
    return parse_ratio >= 0.8 or (name_suggests_date and parse_ratio >= 0.5)


def identify_column_type(column_name, column):
    """Assign a simple, best-effort role to a column."""
    if looks_like_identifier(column_name, column):
        return "Identifier / ID"
    if looks_like_datetime(column_name, column):
        return "Date / time"
    if pd.api.types.is_numeric_dtype(column):
        return "Numerical"
    return "Categorical"


def profile_dataset(data):
    """Build a per-column profile and separate summaries for each column type."""
    rows = len(data)
    profile_rows = []
    numerical_rows = []
    categorical_rows = []

    for column_name in data.columns:
        column = data[column_name]
        missing_count = int(column.isna().sum())
        missing_percent = (missing_count / rows * 100) if rows else 0
        column_type = identify_column_type(column_name, column)

        profile_rows.append(
            {
                "Column": column_name,
                "Data type": str(column.dtype),
                "Detected as": column_type,
                "Missing values": missing_count,
                "Missing (%)": round(missing_percent, 2),
                "Unique values": int(column.nunique(dropna=True)),
            }
        )

        if column_type == "Numerical":
            numerical_rows.append(
                {
                    "Column": column_name,
                    "Count": int(column.count()),
                    "Mean": column.mean(),
                    "Median": column.median(),
                    "Minimum": column.min(),
                    "Maximum": column.max(),
                    "Standard deviation": column.std(),
                }
            )
        elif column_type == "Categorical":
            for value, count in column.value_counts(dropna=True).head(5).items():
                categorical_rows.append(
                    {"Column": column_name, "Value": value, "Count": int(count)}
                )

    return (
        pd.DataFrame(profile_rows),
        pd.DataFrame(numerical_rows),
        pd.DataFrame(categorical_rows),
    )


st.set_page_config(page_title="AI Business Analyst", page_icon="📊", layout="wide")
st.title("AI Business Analyst")
st.write("Upload a business dataset to automatically inspect its structure and contents.")

uploaded_file = st.file_uploader(
    "Choose a CSV or Excel file",
    type=["csv", "xlsx", "xls"],
    help="Supported formats: .csv, .xlsx, and .xls",
)

if uploaded_file is not None:
    try:
        data = load_dataset(uploaded_file)

        st.subheader("Dataset profile")
        row_count, column_count = data.shape
        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("Rows", row_count)
        metric_col2.metric("Columns", column_count)

        profile, numerical_stats, categorical_values = profile_dataset(data)

        st.subheader("Column names, data types, and detected roles")
        if column_count == 0:
            st.info("This file does not contain any columns.")
        else:
            st.dataframe(
                profile[["Column", "Data type", "Detected as"]],
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("Missing values and unique values")
            st.dataframe(
                profile[["Column", "Missing values", "Missing (%)", "Unique values"]],
                use_container_width=True,
                hide_index=True,
            )

        st.subheader("Numerical statistics")
        if numerical_stats.empty:
            st.info("No numerical columns were detected.")
        else:
            st.dataframe(numerical_stats, use_container_width=True, hide_index=True)

        st.subheader("Top values in categorical columns")
        if categorical_values.empty:
            st.info("No categorical columns were detected.")
        else:
            st.dataframe(categorical_values, use_container_width=True, hide_index=True)

        st.subheader("First 10 rows")
        st.dataframe(data.head(10), use_container_width=True, hide_index=True)

        st.subheader("Ask a question about your data")
        question = st.text_input(
            "Business question",
            placeholder="For example: Which region generated the highest profit?",
        )
        if question:
            plan = None
            message = None
            if PLANNER_MODE == "llm":
                try:
                    plan = plan_with_llm(question, data)
                except LLMServiceUnavailable:
                    st.warning("LLM service is unavailable. Using rule-based planner.")
                    plan, message = plan_analysis(question, data)
                except (PlannerResponseError, PlannerValidationError, UnsupportedQuestionError) as error:
                    st.info(str(error))
            elif PLANNER_MODE == "rule_based":
                plan, message = plan_analysis(question, data)
            else:
                st.error("PLANNER_MODE must be set to 'llm' or 'rule_based'.")

            if plan is None:
                if message:
                    st.info(message)
            else:
                try:
                    # Validate fallback requests too, before they reach the engine.
                    validate_analysis_request(plan, data)
                    st.success("Analysis request created and validated.")
                    st.json(plan)
                    result = execute_analysis(data, plan)
                    st.subheader("Analysis result")
                    if isinstance(result, pd.DataFrame):
                        if plan.get("analysis_type") == "groupby_aggregation" and not result.empty:
                            first_row = result.iloc[0]
                            group_column = plan["group_column"]
                            st.write(
                                f"Top match: **{first_row[group_column]}** "
                                f"with **{first_row['Result']}**. The table and chart compare all groups."
                            )
                        st.dataframe(result, use_container_width=True, hide_index=True)
                        chart = prepare_chart_data(result, plan)
                        if chart is not None:
                            chart_type, chart_data = chart
                            st.subheader("Chart")
                            if chart_type == "bar":
                                st.bar_chart(chart_data)
                            elif chart_type == "line":
                                st.line_chart(chart_data)
                    else:
                        st.metric("Result", result)

                    st.subheader("AI Business Insight")
                    try:
                        insight = generate_business_insight(question, plan, data, result)
                        st.markdown(f"**Answer (Fact):** {insight['answer']}")
                        st.markdown(
                            f"**Business Insight (Interpretation):** {insight['business_insight']}"
                        )
                        st.markdown(
                            "**Further Investigation:** "
                            f"{insight['further_investigation']}"
                        )
                    except Exception:
                        st.info(
                            "AI explanation is currently unavailable. The result below was "
                            "calculated directly from the uploaded dataset."
                        )
                except ValueError as error:
                    st.error(f"Could not run this analysis: {error}")
                except (PlannerResponseError, PlannerValidationError, UnsupportedQuestionError) as error:
                    st.info(str(error))

    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError, ValueError) as error:
        st.error("We couldn't read this file as a CSV or Excel dataset.")
        st.caption(f"Details: {error}")
    except Exception as error:
        # Invalid or damaged Excel files can raise different errors depending on the engine.
        st.error("Something went wrong while loading this file. Please check that it is a valid CSV or Excel file.")
        st.caption(f"Details: {error}")
