# Explainable BI & Counterfactual Analysis Assistant

A research-oriented Business Intelligence (BI) assistant that combines **explainable machine learning** with **counterfactual time-series analysis** to answer *why* metrics change — not just *what* changed.

This project was built with a specific goal:  
👉 **to study and demonstrate reliable methods for driver attribution and intervention impact analysis in real-world BI pipelines**, while being explicit about assumptions and tool limitations.

---

## 🔍 What This Project Does

The application supports three complementary analytical modes:

### 1. Driver Analysis (Predictive Attribution)
- Uses a **Random Forest** model trained on historical data.
- Applies **SHAP values** to identify which features most strongly influence a selected KPI.
- Answers questions like:
  - *“What drove revenue this quarter?”*
  - *“Which inputs matter most for sales?”*

⚠️ **Important:** This mode explains *model predictions*, **not causal effects**.

---

### 2. Counterfactual “What-If” Scenarios
- Uses the trained predictive model to simulate hypothetical scenarios.
- Example questions:
  - *“What if email_spend were zero?”*
  - *“What if paid_social_spend doubled?”*

The assistant reports the **predicted change in the KPI**, clearly labeled as *predictive* rather than causal.

---

### 3. Intervention Analysis (Counterfactual Time-Series)
- Estimates whether a real-world change (e.g., campaign launch, pricing update) altered a KPI **beyond trend and seasonality**.
- Implements a **SARIMAX-based counterfactual forecast**:
  - Model is trained on the **pre-intervention period**
  - Forecasts expected KPI values in the **post-intervention period**
  - Compares observed vs forecasted values

This replaces the Python `CausalImpact` library, which was found to be unstable and unreliable for long daily series.

---

## 🧠 Why SARIMAX (and not Python CausalImpact)

During development, the Python implementation of `CausalImpact` was extensively tested and found to:

- Silently fail (`inferences = None`)
- Break under modern Python versions
- Produce non-deterministic errors even with clean data

Rather than forcing a broken tool, this project:
- Diagnoses the limitation explicitly
- Replaces it with **SARIMAX**, a maintained and well-understood state-space model
- Adds convergence checks, frequency inference, and fallback strategies (e.g., weekly aggregation)

This design choice reflects **research integrity over convenience**.

---

## 📊 Methodological Safeguards

The intervention analysis includes several safeguards to prevent misleading results:

- Automatic **frequency inference** (daily vs weekly)
- Fallback to **weekly aggregation** when daily models are unstable
- Explicit **model convergence checks**
- Conservative handling of covariates (never filling the KPI)
- Clear refusal to report impact when the model is unreliable

If the model cannot support a defensible estimate, the system **says so**.

---

## ⚠️ Assumptions & Limitations

All results are subject to standard observational assumptions:

- Stable pre-intervention trend
- No major unobserved confounders
- Correct model specification
- Sufficient pre-period data

This tool **does not claim randomized causal proof**.  
It provides *forecast-based counterfactual evidence* under stated assumptions.

---

## 🛠️ Tech Stack

- **Backend:** Python, Flask  
- **ML / Stats:** scikit-learn, SHAP, statsmodels (SARIMAX)  
- **LLM Layer:** OpenRouter (for natural-language summaries only)  
- **Frontend:** HTML / CSS (lightweight, no JS framework)  
- **Deployment:** HuggingFace Spaces  

---

## 🚀 Typical Use Cases

- BI teams seeking interpretable insights, not black-box dashboards
- Analysts validating whether a campaign outperformed expected trends
- Researchers studying reliability of explainability and causal tools
- Academic demonstrations of applied counterfactual analysis

---

## 🧪 Recommended Robustness Checks

When using the intervention analysis:

1. Re-run with **weekly aggregation**
2. Perform a **placebo test** (fake intervention date)
3. Compare against a simpler univariate model
4. Inspect forecast intervals, not just point estimates

---

## 📌 Project Philosophy

> *If a model cannot produce a reliable answer, the system should say “I don’t know” rather than guess.*

This project prioritizes:
- Transparency over polish
- Correctness over convenience
- Research defensibility over marketing claims

---

## 📬 Notes for Researchers & Collaborators

This codebase is intentionally written to be:
- Readable
- Auditable
- Easy to extend (e.g., EconML, DiD, Prophet baselines)

If you are reviewing this project for research collaboration, the focus should be on:
- Method choice
- Failure-mode handling
- Assumption clarity

Not just outputs.

---

*Assumptions apply: stable pre-trend, correct model specification, no major unobserved confounding.*
