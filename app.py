from datetime import timedelta
import os
import io
import uuid
import re

import numpy as np
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, session

from sklearn.ensemble import RandomForestRegressor
import shap
from causalimpact import CausalImpact

from dotenv import load_dotenv
from openai import OpenAI

# ------------------- APP CONFIG -------------------

load_dotenv()
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret")

app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(hours=1)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

# Keep dataframes in memory keyed by small token, not in cookie
DATAFRAMES: dict[str, pd.DataFrame] = {}


# ------------------- STATE HELPERS -------------------

def get_state():
    state = session.get("state", {})
    if "messages" not in state:
        state["messages"] = []
    if "df_key" not in state:
        state["df_key"] = None
    if "df_meta" not in state:
        state["df_meta"] = {}
    if "config" not in state:
        state["config"] = {
            "mode": "drivers",
            "kpi_col": None,
            "date_col": None,
            "pre_start": None,
            "pre_end": None,
            "post_start": None,
            "post_end": None,
        }
    if "ui" not in state:
        state["ui"] = {"warnings": []}
    session["state"] = state
    return state


def current_df():
    state = get_state()
    key = state.get("df_key")
    if not key:
        return None
    return DATAFRAMES.get(key)


def pick_kpi_column(df: pd.DataFrame):
    preferred = ["revenue", "sales", "kpi", "target", "organic_traffic"]
    for name in preferred:
        for col in df.columns:
            if name in col.lower() and pd.api.types.is_numeric_dtype(df[col]):
                return col
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            return col
    return None


def pick_date_column(df: pd.DataFrame):
    best = None
    best_valid = 0
    for col in df.columns:
        try:
            s = pd.to_datetime(df[col], errors="coerce")
        except Exception:
            continue
        valid = int(s.notna().sum())
        if valid > best_valid and valid >= max(5, int(0.7 * len(df))):
            best = col
            best_valid = valid
    return best


def compute_warnings(df: pd.DataFrame, cfg: dict, meta: dict, user_question: str | None = None):
    warnings = []

    if cfg.get("mode") == "drivers":
        warnings.append(
            "Predictive attribution: Driver Analysis explains model predictions, not real-world causality."
        )
    else:
        warnings.append(
            "Causal estimation: CausalImpact analyzes real interventions at a specific time point, not what-if scenarios."
        )

    if len(df) < 20:
        warnings.append("Small sample: fewer than 20 rows can make both attributions and intervention estimates unstable.")

    num_cols = meta.get("numeric_cols", [])
    if len(num_cols) >= 2:
        corr = df[num_cols].corr(numeric_only=True).abs()
        np.fill_diagonal(corr.values, 0.0)
        if (corr.values > 0.95).any():
            warnings.append("High collinearity: some numeric features are highly correlated, which can destabilize explanations.")

    if cfg.get("mode") == "intervention" and user_question:
        t = user_question.lower()
        if "what if" in t or "zero" in t or "2x" in t or "double" in t:
            warnings.append("Question mismatch: use Driver Analysis for what-if scenarios; Intervention Analysis is for real changes with pre/post periods.")

    return warnings


# ------------------- ROBUST CSV PARSING -------------------

def robust_read_csv(raw_bytes: bytes) -> pd.DataFrame:
    print("UPLOAD STARTED, size:", len(raw_bytes), "bytes", flush=True)

    preview = raw_bytes[:800].decode("utf-8", errors="replace")
    print("RAW PREVIEW (first 3 lines):", flush=True)
    for i, line in enumerate(preview.splitlines()[:3]):
        print(f"  {i}: {line}", flush=True)

    encodings = ["utf-8", "cp1252", "latin-1", "iso-8859-1"]

    # 1) C engine, default separator
    for enc in encodings:
        try:
            df = pd.read_csv(io.BytesIO(raw_bytes), encoding=enc, low_memory=False)
            if not df.empty and len(df.columns) > 0:
                print(f"C ENGINE SUCCESS {df.shape} using {enc}", flush=True)
                return df
        except Exception as e:
            print(f"C ENGINE FAILED {enc}: {type(e).__name__} {e}", flush=True)

    # 2) Python engine with sniffed separator
    for enc in encodings:
        try:
            df = pd.read_csv(
                io.StringIO(raw_bytes.decode(enc, errors="replace")),
                sep=None,
                engine="python",
                on_bad_lines="skip",
            )
            if not df.empty and len(df.columns) > 0:
                print(f"PYTHON ENGINE SUCCESS {df.shape} using {enc}", flush=True)
                return df
        except Exception as e:
            print(f"PYTHON ENGINE FAILED {enc}: {type(e).__name__} {e}", flush=True)

    # 3) Last resort: single-column text
    text = raw_bytes.decode("utf-8", errors="replace")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("All parsers failed and file appears empty.")

    df = pd.DataFrame({"raw_text": lines})
    print("FALLBACK SINGLE COLUMN, rows:", len(df), flush=True)
    return df


# ------------------- DRIVER ANALYSIS (SHAP) -------------------

def run_driver_analysis(df: pd.DataFrame, target_col: str):
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data.")

    num_df = df.select_dtypes(include=[np.number]).copy()
    if target_col not in num_df.columns:
        num_df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    num_df = num_df.dropna(subset=[target_col])

    X = num_df.drop(columns=[target_col], errors="ignore")
    y = num_df[target_col]

    if X.shape[1] < 1:
        X = pd.DataFrame({"row_index": np.arange(len(y))}, index=y.index)

    if len(X) < 10:
        raise ValueError("Need at least 10 valid rows for driver analysis after cleaning the data.")

    X_sample = X.iloc[:200].copy()

    model = RandomForestRegressor(n_estimators=250, random_state=42)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_sample)

    importances = np.abs(shap_vals).mean(axis=0)
    top_features = (
        pd.DataFrame({"feature": X.columns, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(7)
        .to_dict("records")
    )
    return top_features


def llm_summarize_drivers(question: str, kpi: str, top_features):
    bullet_text = "; ".join(
        f"{item['feature']} (importance {item['importance']:.3f})" for item in top_features
    )
    prompt = (
        "You are a BI assistant.\n"
        "Mode: Predictive attribution with SHAP (this explains model predictions, not causal effects).\n"
        f"User question: '{question}'.\n"
        f"KPI: '{kpi}'.\n"
        f"Top driver features (mean absolute SHAP): {bullet_text}.\n"
        "Write 2 to 3 sentences that:\n"
        "1) Name the strongest drivers.\n"
        "2) Explicitly state that this is predictive, not causal.\n"
        "3) Give one concrete business action.\n"
    )
    resp = client.chat.completions.create(
        model="openai/gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=260,
    )
    return resp.choices[0].message.content


# ------------------- COUNTERFACTUAL (WHAT-IF) -------------------

def is_counterfactual_question(text: str) -> bool:
    t = text.lower()
    return bool(
        re.search(r"\bwhat if\b", t)
        or re.search(r"\bif\b.*\bzero\b", t)
        or re.search(r"\bif\b.*\b0\b", t)
        or "double" in t
        or "2x" in t
        or "half" in t
    )


def run_counterfactual_driver_scenario(df: pd.DataFrame, question: str, kpi_col: str) -> str:
    num_df = df.select_dtypes(include=[np.number]).copy()
    if kpi_col not in num_df.columns:
        num_df[kpi_col] = pd.to_numeric(df[kpi_col], errors="coerce")
    num_df = num_df.dropna(subset=[kpi_col])

    X = num_df.drop(columns=[kpi_col], errors="ignore")
    y = num_df[kpi_col]

    if len(X) < 10:
        return "Not enough numeric rows to run a counterfactual driver scenario."

    model = RandomForestRegressor(n_estimators=250, random_state=42)
    model.fit(X, y)

    feature_to_zero = None
    for col in X.columns:
        if col.lower() in question.lower():
            feature_to_zero = col
            break

    if feature_to_zero is None:
        return (
            "Counterfactual question detected, but could not match any feature name "
            "in your question. Please mention the exact column name (for example 'email_spend')."
        )

    y_pred_baseline = model.predict(X)
    baseline_avg = float(y_pred_baseline.mean())

    X_scenario = X.copy()
    X_scenario[feature_to_zero] = 0
    y_pred_scenario = model.predict(X_scenario)
    scenario_avg = float(y_pred_scenario.mean())

    delta = scenario_avg - baseline_avg
    pct_change = (delta / baseline_avg * 100) if baseline_avg != 0 else 0

    direction = "decrease" if delta < 0 else "increase"
    magnitude = abs(delta)

    result = (
        f"Counterfactual scenario: if {feature_to_zero} were zero across your observed data period,\n"
        f"average {kpi_col} would {direction} by about {magnitude:.2f} units ({pct_change:.1f}%).\n\n"
        "Current average (baseline): {:.2f}\n"
        "Predicted if {feature_to_zero}=0: {:.2f}\n\n"
        "Note: This is a predictive what-if based on a RandomForest model trained on your data,\n"
        "not a causal estimate. It shows feature importance via counterfactual simulation."
    ).format(baseline_avg, scenario_avg)

    return result


def llm_summarize_counterfactual(question: str, kpi: str, scenario_result: str):
    prompt = (
        "You are a BI assistant.\n"
        "Mode: Predictive counterfactual what-if via RandomForest.\n"
        f"User question: '{question}'.\n"
        f"KPI: '{kpi}'.\n\n"
        f"Scenario result:\n{scenario_result}\n\n"
        "Write 2-3 sentences that:\n"
        "1) Clearly state what the model predicts would happen.\n"
        "2) Explain this is based on historical patterns in the data, not causal proof.\n"
        "3) Suggest one actionable next step to validate or improve this feature.\n"
    )
    resp = client.chat.completions.create(
        model="openai/gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=280,
    )
    return resp.choices[0].message.content


# ------------------- INTERVENTION ANALYSIS (CAUSALIMPACT) -------------------

def run_intervention_analysis(
    df: pd.DataFrame,
    date_col: str,
    kpi_col: str,
    pre_start: str,
    pre_end: str,
    post_start: str,
    post_end: str,
) -> str:
    if kpi_col not in df.columns:
        return f"KPI column '{kpi_col}' not found in data. Please pick a numeric KPI."

    if date_col not in df.columns:
        return (
            f"Date column '{date_col}' not found. "
            "Intervention Analysis requires a real date/time column. "
            "You can still use Driver Analysis (SHAP) without dates."
        )

    # Try European-style day-first dates like dd-mm-yyyy
    date_series = pd.to_datetime(
    df[date_col],
    errors="coerce",
    dayfirst=True,      # <‑‑ key change
)

    valid_dates = int(date_series.notna().sum())
    if valid_dates < 10:
        sample_raw = df[date_col].dropna().astype(str).head(5).tolist()
        return (
            f"Could not parse enough valid dates in '{date_col}' for intervention analysis "
            f"(only {valid_dates} rows look like dates). "
            f"Examples from that column: {sample_raw}. "
            "Choose a proper date column or switch to Driver Analysis."
        )

    kpi_series = pd.to_numeric(df[kpi_col], errors="coerce")
    valid_kpi = int(kpi_series.notna().sum())
    if valid_kpi < 10:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return (
            f"KPI column '{kpi_col}' has only {valid_kpi} numeric values after cleaning, "
            "which is not enough for intervention analysis. "
            + (
                f"Try one of these numeric columns instead: {numeric_cols[:5]}."
                if numeric_cols
                else "There are no numeric columns available in this dataset."
            )
        )

    numeric_df = df.select_dtypes(include=[np.number]).copy()
    if kpi_col not in numeric_df.columns:
        numeric_df[kpi_col] = kpi_series

    covariates = [c for c in numeric_df.columns if c != kpi_col]

    data = pd.DataFrame({date_col: date_series, kpi_col: numeric_df[kpi_col]})
    if covariates:
        for c in covariates:
            data[c] = numeric_df[c]

    data = (
        data.dropna(subset=[date_col, kpi_col])
        .sort_values(date_col)
        .set_index(date_col)
    )

    if data.empty or data[kpi_col].notna().sum() < 10:
        return (
            "After removing rows with missing dates or KPI values, "
            "there is not enough data left to run intervention analysis. "
            "Try a different KPI or date range, or switch to Driver Analysis."
        )

    if covariates:
        keep_covs = [c for c in covariates if data[c].nunique(dropna=True) > 1]
        if keep_covs:
            data = data[[kpi_col] + keep_covs]
        else:
            data = data[[kpi_col]]
    else:
        data = data[[kpi_col]]

    try:
        pre_start_dt = pd.to_datetime(pre_start)
        pre_end_dt = pd.to_datetime(pre_end)
        post_start_dt = pd.to_datetime(post_start)
        post_end_dt = pd.to_datetime(post_end)

        if not (pre_start_dt < pre_end_dt < post_start_dt < post_end_dt):
            return "Pre and post periods must be in chronological order and non-overlapping."

        data_window = data.loc[pre_start_dt:post_end_dt].copy()
        if data_window[kpi_col].notna().sum() < 10:
            return (
                "Not enough observations between the specified pre and post dates "
                "to run CausalImpact. Please widen the date ranges."
            )

        idx = data_window.index
        try:
            pre0 = idx.get_loc(pre_start_dt)
            pre1 = idx.get_loc(pre_end_dt)
            post0 = idx.get_loc(post_start_dt)
            post1 = idx.get_loc(post_end_dt)
        except KeyError:
            return "Chosen pre/post dates do not exactly match any rows in the date column."

        pre_period = [pre0, pre1]
        post_period = [post0, post1]

        ci = CausalImpact(data_window, pre_period, post_period)
        summary = ci.summary(output="report")
        return summary

    except Exception as e:
        return (
            "CausalImpact could not be run with the current configuration. "
            f"Technical detail: {e}"
        )


def llm_summarize_intervention(
    question: str,
    kpi: str,
    date_col: str,
    pre_start: str,
    pre_end: str,
    post_start: str,
    post_end: str,
    ci_summary: str,
):
    prompt = (
        "You are a BI assistant.\n"
        "Mode: Causal estimation via Bayesian structural time-series (CausalImpact).\n"
        f"User question: '{question}'.\n"
        f"KPI: '{kpi}', indexed by date column '{date_col}'.\n"
        f"Pre-period: {pre_start} to {pre_end}.\n"
        f"Post-period: {post_start} to {post_end}.\n\n"
        "CausalImpact summary:\n"
        f"{ci_summary}\n\n"
        "Write 3 short bullet points:\n"
        "- Whether the intervention likely changed the KPI.\n"
        "- Direction and rough magnitude.\n"
        "- One next action.\n"
        "Also add a final line: 'Assumptions apply: stable pre-trend, no major unobserved confounding.'"
    )
    resp = client.chat.completions.create(
        model="openai/gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=320,
    )
    return resp.choices[0].message.content


# ------------------- SHARED HELPERS & ROUTES -------------------

def infer_kpi_from_question(text, df_columns):
    for col in df_columns:
        if col.lower() in text.lower():
            return col
    return None


@app.route("/", methods=["GET"])
def home():
    state = get_state()
    return render_template(
        "index.html",
        messages=state["messages"],
        meta=state["df_meta"],
        config=state["config"],
        ui=state["ui"],
    )


@app.route("/upload", methods=["POST"])
def upload():
    state = get_state()
    f = request.files.get("file")

    if not f or not f.filename:
        return redirect(url_for("home"))

    try:
        content = f.read()
        df = robust_read_csv(content)

        df_key = str(uuid.uuid4())[:8]
        DATAFRAMES[df_key] = df
        state["df_key"] = df_key

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        state["df_meta"] = {
            "columns": list(df.columns),
            "numeric_cols": numeric_cols,
            "non_numeric_cols": [c for c in df.columns if c not in numeric_cols],
            "kpi_guess": pick_kpi_column(df),
            "date_guess": pick_date_column(df),
            "row_count": len(df),
            "error": None,
        }

        # Pre-fill config with guesses, but UI will not show "suggested" labels
        cfg = state["config"]
        if not cfg.get("kpi_col"):
            cfg["kpi_col"] = state["df_meta"]["kpi_guess"]
        if not cfg.get("date_col"):
            cfg["date_col"] = state["df_meta"]["date_guess"]
        state["config"] = cfg

        print(
            "SUCCESS - columns saved",
            state["df_meta"]["columns"],
            "rows",
            state["df_meta"]["row_count"],
            flush=True,
        )

    except Exception as e:
        print("UPLOAD FAILED", type(e), e, flush=True)
        state["df_meta"] = {"error": str(e), "columns": [], "row_count": 0}
        state["df_key"] = None

    session["state"] = state
    session.modified = True
    return redirect(url_for("home"))


@app.route("/set_config", methods=["POST"])
def set_config():
    state = get_state()
    cfg = state["config"]

    cfg["mode"] = request.form.get("mode", cfg.get("mode", "drivers"))

    kpi_val = request.form.get("kpi_col")
    if kpi_val:
        cfg["kpi_col"] = kpi_val

    cfg["date_col"] = request.form.get("date_col") or cfg.get("date_col")
    cfg["pre_start"] = request.form.get("pre_start") or cfg.get("pre_start")
    cfg["pre_end"] = request.form.get("pre_end") or cfg.get("pre_end")
    cfg["post_start"] = request.form.get("post_start") or cfg.get("post_start")
    cfg["post_end"] = request.form.get("post_end") or cfg.get("post_end")

    state["config"] = cfg

    df = current_df()
    if df is not None and not df.empty:
        state["ui"]["warnings"] = compute_warnings(df, cfg, state["df_meta"])

    session["state"] = state
    session.modified = True
    return redirect(url_for("home"))


@app.route("/chat", methods=["POST"])
def chat():
    state = get_state()
    user_msg = request.form.get("message", "").strip()
    if not user_msg:
        return redirect(url_for("home"))

    state["messages"].append({"id": str(uuid.uuid4()), "role": "user", "content": user_msg})

    df = current_df()
    if df is None:
        state["messages"].append(
            {"id": str(uuid.uuid4()), "role": "assistant", "content": "Please upload a CSV first."}
        )
        session["state"] = state
        session.modified = True
        return redirect(url_for("home"))

    if df.empty:
        state["messages"].append(
            {"id": str(uuid.uuid4()), "role": "assistant", "content": "Uploaded CSV is empty."}
        )
        session["state"] = state
        session.modified = True
        return redirect(url_for("home"))

    cfg = state["config"]
    mode = cfg.get("mode", "drivers")
    counterfactual = is_counterfactual_question(user_msg)

    # Effective mode: force drivers for counterfactual questions, otherwise respect dropdown
    effective_mode = "drivers" if counterfactual else mode

    state["ui"]["warnings"] = compute_warnings(df, cfg, state["df_meta"], user_msg)

    kpi_from_text = infer_kpi_from_question(user_msg, df.columns)
    kpi_col = kpi_from_text or cfg.get("kpi_col") or state["df_meta"].get("kpi_guess")

    try:
        if effective_mode == "drivers":
            if not kpi_col:
                raise ValueError("No KPI column selected. Please pick one in settings.")

            if counterfactual:
                scenario_result = run_counterfactual_driver_scenario(df, user_msg, kpi_col)
                answer = llm_summarize_counterfactual(user_msg, kpi_col, scenario_result)
            else:
                top_features = run_driver_analysis(df, kpi_col)
                answer = llm_summarize_drivers(user_msg, kpi_col, top_features)
        else:
            date_col = cfg.get("date_col")
            pre_start = cfg.get("pre_start")
            pre_end = cfg.get("pre_end")
            post_start = cfg.get("post_start")
            post_end = cfg.get("post_end")

            if not (kpi_col and date_col and pre_start and pre_end and post_start and post_end):
                raise ValueError("Set KPI, date column, and all four pre/post dates in settings for CausalImpact.")

            ci_summary = run_intervention_analysis(
                df, date_col, kpi_col, pre_start, pre_end, post_start, post_end
            )
            answer = llm_summarize_intervention(
                user_msg,
                kpi_col,
                date_col,
                pre_start,
                pre_end,
                post_start,
                post_end,
                ci_summary,
            )

    except Exception as e:
        answer = f"Analysis error: {e}"

    state["messages"].append({"id": str(uuid.uuid4()), "role": "assistant", "content": answer})
    state["messages"] = state["messages"][-80:]
    session["state"] = state
    session.modified = True
    return redirect(url_for("home"))


if __name__ == "__main__":
    app.run(debug=True)
