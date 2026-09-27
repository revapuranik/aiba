"""Generate a grounded, business-friendly explanation of Pandas results."""

import json
import math
import re
import urllib.error
import urllib.request
from numbers import Real

import pandas as pd

from config import VLLM_API_KEY, VLLM_BASE_URL, VLLM_MODEL, VLLM_TIMEOUT_SECONDS


SYSTEM_PROMPT = """You are a business analyst assistant.

Explain the results of an analysis performed by a Python/Pandas analytics
engine. The calculated results provided to you are authoritative. Do not
invent, change, estimate, convert, or recalculate numerical values. Do not
claim to have analyzed data that is not included in the supplied context. Use
only the question, request, dataset details, and calculated result provided.

Return a concise business-focused explanation as one valid JSON object with
exactly these string fields: answer, business_insight, further_investigation.
The answer should directly state the fact shown by the actual result. The
business_insight should clearly mark interpretation as tentative (for example,
"This may suggest..."). The further_investigation field should suggest one or
two questions to examine next, without stating unsupported causes.

Do not make causal claims such as "because", "due to", or "caused by". Do not
introduce numerical values that do not appear in the provided result context.
Return JSON only. Do not return Python code, markdown, or extra fields."""

REQUIRED_FIELDS = {"answer", "business_insight", "further_investigation"}
NUMBER_PATTERN = re.compile(
    r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?(?![\w.])"
)
UNSUPPORTED_CAUSAL_PATTERN = re.compile(
    r"\b(because|due to|caus\w*|driven by|resulted from|resulting from|"
    r"as a result of|as a consequence|the reason|led to|leads to|thanks to|"
    r"attribut\w* to|explain\w* by|influenced by|contributed to)\b",
    re.IGNORECASE,
)


class InsightServiceUnavailable(Exception):
    """Raised when the configured vLLM service cannot be reached."""


class InsightResponseError(Exception):
    """Raised when the LLM response is invalid or not grounded in the result."""


def _native_value(value):
    """Convert common Pandas/Numpy values to JSON-friendly Python values."""
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _relevant_columns(request):
    names = []
    for field in ("group_column", "value_column", "date_column"):
        value = request.get(field)
        if isinstance(value, str) and value not in names:
            names.append(value)
    for value in request.get("columns", []):
        if isinstance(value, str) and value not in names:
            names.append(value)
    return names


def build_result_context(question, request, data, result, sample_rows=20):
    """Build small context from the request and calculated result, not the dataset."""
    relevant_names = _relevant_columns(request)
    dataset_details = {
        "shape": {"rows": int(data.shape[0]), "columns": int(data.shape[1])},
        "relevant_columns": [
            {"name": name, "data_type": str(data[name].dtype)}
            for name in relevant_names
            if name in data.columns
        ],
    }

    if isinstance(result, pd.DataFrame):
        sample = result.head(sample_rows).astype(object)
        sample = sample.where(pd.notna(sample), None)
        numeric_summary = {}
        for column_name in result.select_dtypes(include="number").columns:
            values = result[column_name].dropna()
            if not values.empty:
                numeric_summary[column_name] = {
                    "minimum": _native_value(values.min()),
                    "maximum": _native_value(values.max()),
                    "mean": _native_value(values.mean()),
                }
        actual_result = {
            "result_type": "table",
            "columns": [str(name) for name in result.columns],
            "row_count": int(len(result)),
            "rows_shown": min(len(result), sample_rows),
            "rows": sample.to_dict(orient="records"),
            "numeric_summary": numeric_summary,
        }
    else:
        actual_result = {
            "result_type": "single_value",
            "value": _native_value(result),
        }

    return {
        "user_question": question,
        "analysis_type": request.get("analysis_type"),
        "analysis_request": request,
        "dataset_information": dataset_details,
        "actual_calculated_result": actual_result,
    }


def _numeric_values_in(value):
    """Collect numeric literals the LLM was allowed to see in the result context."""
    if isinstance(value, bool) or value is None:
        return set()
    if isinstance(value, Real):
        return {float(value)}
    if isinstance(value, dict):
        found = set()
        for item in value.values():
            found.update(_numeric_values_in(item))
        return found
    if isinstance(value, (list, tuple)):
        found = set()
        for item in value:
            found.update(_numeric_values_in(item))
        return found
    if isinstance(value, str):
        found = set()
        for match in NUMBER_PATTERN.findall(value):
            token = match.rstrip("% ").replace(",", "")
            try:
                found.add(float(token))
            except ValueError:
                continue
        return found
    return set()


def _validate_response(response, context):
    if not isinstance(response, dict) or set(response) != REQUIRED_FIELDS:
        raise InsightResponseError("The LLM response did not contain the required insight sections.")
    for field in REQUIRED_FIELDS:
        if not isinstance(response[field], str) or not response[field].strip():
            raise InsightResponseError("The LLM returned an empty or invalid insight section.")
        if len(response[field]) > 1500:
            raise InsightResponseError("The LLM response was longer than the allowed length.")
        if UNSUPPORTED_CAUSAL_PATTERN.search(response[field]):
            raise InsightResponseError("The LLM included an unsupported causal explanation.")

    allowed_numbers = _numeric_values_in(context["actual_calculated_result"])
    allowed_numbers.update(_numeric_values_in(context["dataset_information"]["shape"]))
    for field in REQUIRED_FIELDS:
        for claimed_number in _numeric_values_in(response[field]):
            if not any(
                math.isclose(claimed_number, allowed, rel_tol=1e-9, abs_tol=1e-9)
                for allowed in allowed_numbers
            ):
                raise InsightResponseError(
                    "The LLM included a number that was not present in the supplied analysis result."
                )
    return {field: response[field].strip() for field in REQUIRED_FIELDS}


def _request_insight(context):
    if not VLLM_MODEL:
        raise InsightServiceUnavailable("VLLM_MODEL is not configured.")

    url = f"{VLLM_BASE_URL}/chat/completions"
    body = {
        "model": VLLM_MODEL,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(context, default=str, allow_nan=False)},
        ],
    }
    headers = {"Content-Type": "application/json"}
    if VLLM_API_KEY:
        headers["Authorization"] = f"Bearer {VLLM_API_KEY}"
    api_request = urllib.request.Request(
        url,
        data=json.dumps(body, allow_nan=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(api_request, timeout=VLLM_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
        raise InsightServiceUnavailable("The vLLM service could not be reached.") from error

    try:
        content = payload["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        return json.loads(content)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
        raise InsightResponseError("The LLM did not return valid JSON for its explanation.") from error


def generate_business_insight(question, request, data, result):
    """Generate and validate an explanation grounded in an already-computed result."""
    context = build_result_context(question, request, data, result)
    response = _request_insight(context)
    return _validate_response(response, context)
