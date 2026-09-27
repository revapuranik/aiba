# AI Business Analyst

A small Streamlit application for uploading a CSV or Excel business dataset and automatically profiling its structure and contents.

## Run the application

1. Create and activate a Python virtual environment (optional, but recommended).
2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the app:

   ```bash
   streamlit run app.py
   ```

Upload a `.csv`, `.xlsx`, or `.xls` file. The app profiles the dataset, sends a small schema and sample to vLLM (when selected), validates its structured request, and runs valid requests with the Pandas analysis engine. The calculated result and chart remain visible. A separate LLM request then explains the actual result in three sections: Answer, Business Insight, and Further Investigation. The rule-based planner remains available as a mode and as a fallback when vLLM is unavailable. The LLM does not calculate results or execute Python code.

## Configure vLLM

The application expects an already-running OpenAI-compatible vLLM server. It does not download or start a model. Copy `.env.example` to `.env` and set `VLLM_MODEL` to the exact model name served by your vLLM process. The app reads simple `KEY=value` entries from `.env`; environment variables already set in the terminal take precedence.

On a Linux environment supported by vLLM, set the model name and start the server:

```bash
export VLLM_MODEL="your-model-id"
vllm serve "$VLLM_MODEL" --host 0.0.0.0 --port 8000
```

In a second terminal, configure the Streamlit app and run it:

```bash
export PLANNER_MODE=llm
export VLLM_BASE_URL=http://localhost:8000/v1
export VLLM_MODEL="your-model-id"
streamlit run app.py
```

In Windows PowerShell, set the app settings in the terminal where you run Streamlit:

```powershell
$env:PLANNER_MODE = "llm"
$env:VLLM_BASE_URL = "http://localhost:8000/v1"
$env:VLLM_MODEL = "your-model-id"
streamlit run app.py
```

vLLM does not run natively on Windows; use WSL with a compatible Linux distribution and hardware. The model installation steps depend on the GPU or CPU platform, so follow the matching official vLLM installation instructions before running the `vllm serve` command.

Try these questions using column names present in your uploaded file:

- `Which region generated the highest profit?`
- `Which product generated the highest revenue?`
- `What is the average discount?`
- `Show monthly revenue.`
- `Tell me something about the weather.` (should be rejected as unsupported)

After a valid question, the app shows the raw Pandas result and chart first, followed by the three AI explanation sections. If vLLM is unavailable or returns an invalid/ungrounded explanation, the raw result remains available and the app shows an explanation-unavailable notice.

To use the local rule-based planner directly, set `PLANNER_MODE=rule_based`. If vLLM is unavailable while LLM mode is selected, the app displays a notice and uses the rule-based planner.

## Project files

- `app.py` contains the Streamlit interface, CSV/Excel loading logic, reusable profiling functions, and column role detection.
- `analysis_planner.py` converts supported natural-language questions into structured analysis requests without an LLM.
- `analysis_engine.py` validates requests and performs the supported Pandas aggregations and time-series grouping.
- `charting.py` prepares grouped or time-series results for simple charts, separately from the analysis calculations.
- `llm_planner.py` creates a compact dataset context, calls the vLLM chat endpoint, parses JSON, and validates requests before execution.
- `insight_generator.py` sends a compact context containing the question, request, relevant columns, and calculated result to the same configured vLLM service. It checks the response sections and rejects unsupported numeric claims and common causal phrasing.
- `config.py` reads planner and vLLM settings from environment variables.
- `.env.example` lists the settings to configure; copy it as a reference and provide the actual values in your environment.
- `requirements.txt` lists the Python packages needed to run the app. `openpyxl` supports `.xlsx` files, and `xlrd` supports older `.xls` files.
- `README.md` explains how to install and run the application.
