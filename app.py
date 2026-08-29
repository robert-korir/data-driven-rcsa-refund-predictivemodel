from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Data-Driven Multi-KRI RCSA Dashboard",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# FILES
# ============================================================

MODEL_PATH = Path("Final_Multi_KRI_Predictive_Model.pkl")
CONFIG_PATH = Path("RCSA_MultiKRI_Configuration.json")


# ============================================================
# LOAD MODEL / CONFIG
# ============================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_config():
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


model = load_model()
config = load_config()


# ============================================================
# CONFIGURATION
# ============================================================

FEATURES = config.get(
    "features",
    [
        "current_chargebacks",
        "current_refund_operations",
        "chargeback_lag_1",
        "chargeback_lag_2",
        "chargeback_lag_3",
        "chargeback_lag_7",
        "refund_lag_1",
        "refund_lag_2",
        "refund_lag_3",
        "refund_lag_7",
        "chargeback_mean_3",
        "chargeback_mean_7",
        "refund_mean_3",
        "refund_mean_7",
        "chargeback_std_7",
        "refund_std_7",
        "chargeback_change_1",
        "chargeback_change_3",
        "refund_change_1",
        "refund_change_3",
        "day_sin",
        "day_cos",
    ],
)

KRI3_COL = "KRI3_waiting_refunds_after_14_days"
KRI4_COL = "KRI4_chargebacks"
KRI5_COL = "KRI5_historical_refund_operations"

KRI3_Q1 = float(config.get("kri3", {}).get("q1", 35.5))
KRI3_Q3 = float(config.get("kri3", {}).get("q3", 45.0))
KRI4_Q1 = float(config.get("kri4", {}).get("q1", 194.0))
KRI4_Q3 = float(config.get("kri4", {}).get("q3", 1534.75))

KRI3_WEIGHT = float(config.get("kri3", {}).get("weight", 0.50))
KRI4_WEIGHT = float(config.get("kri4", {}).get("weight", 0.50))

GREEN_AMBER_BOUNDARY = float(
    config.get("composite_boundaries", {}).get(
        "green_upper_exclusive", 1.50
    )
)
AMBER_RED_BOUNDARY = float(
    config.get("composite_boundaries", {}).get(
        "red_lower_inclusive", 2.50
    )
)


# ============================================================
# HELPERS
# ============================================================

def classify_higher_is_worse(value, q1, q3):
    if pd.isna(value):
        return np.nan
    if value <= q1:
        return "Green"
    if value <= q3:
        return "Amber"
    return "Red"


def risk_score(risk_class):
    return {"Green": 1, "Amber": 2, "Red": 3}.get(risk_class, np.nan)


def classify_composite(score):
    if pd.isna(score):
        return np.nan
    if score < GREEN_AMBER_BOUNDARY:
        return "Green"
    if score < AMBER_RED_BOUNDARY:
        return "Amber"
    return "Red"


def predictive_monitoring_signal(z):
    if pd.isna(z):
        return "Insufficient variability information"
    if z >= 2:
        return "Substantially Above Recent Range"
    if z >= 1:
        return "Above Recent Range"
    if z <= -2:
        return "Substantially Below Recent Range"
    if z <= -1:
        return "Below Recent Range"
    return "Within Recent Range"


def integrated_rcsa_signal(composite_class, predictive_signal):
    elevated_forecast = predictive_signal in {
        "Above Recent Range",
        "Substantially Above Recent Range",
    }

    if composite_class == "Red":
        if elevated_forecast:
            return (
                "Priority RCSA review supported. Current multi-KRI evidence is Red "
                "and the predictive component indicates higher-than-recent refund "
                "operational activity. Investigate the underlying drivers, review "
                "control effectiveness, and determine whether the current RCSA "
                "assessment remains appropriate."
            )
        return (
            "RCSA review or escalation supported by current Red multi-KRI evidence. "
            "Investigate the risk drivers and review control effectiveness."
        )

    if composite_class == "Amber":
        if elevated_forecast:
            return (
                "Enhanced monitoring and RCSA review supported. Current multi-KRI "
                "evidence is Amber and the forecast indicates increasing operational "
                "activity. Review refund-processing conditions, backlog indicators, "
                "chargebacks, and relevant controls."
            )
        return (
            "Enhanced monitoring supported. Current multi-KRI evidence is Amber. "
            "Continue trend monitoring and assess whether the existing RCSA rating "
            "remains appropriate."
        )

    if composite_class == "Green":
        if elevated_forecast:
            return (
                "Current multi-KRI evidence remains Green, but the predictive "
                "component indicates higher-than-recent refund operational activity. "
                "Maintain the current RCSA assessment while increasing short-term "
                "monitoring."
            )
        return (
            "Routine monitoring supported. Current multi-KRI evidence is Green and "
            "the predictive component does not indicate an unusual increase in "
            "refund operational activity."
        )

    return "Insufficient evidence for integrated RCSA decision support."


def validate_master_data(data):
    required = {"date", KRI3_COL, KRI4_COL, KRI5_COL}
    missing = required.difference(data.columns)

    if missing:
        raise ValueError(
            "Uploaded CSV is missing required column(s): "
            + ", ".join(sorted(missing))
        )

    work = data[["date", KRI3_COL, KRI4_COL, KRI5_COL]].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")

    for col in [KRI3_COL, KRI4_COL, KRI5_COL]:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    if work["date"].isna().any():
        raise ValueError("One or more date values could not be parsed.")

    if work["date"].duplicated().any():
        raise ValueError("Duplicate dates were found in the uploaded data.")

    if (work[[KRI3_COL, KRI4_COL, KRI5_COL]] < 0).any(axis=None):
        raise ValueError("Negative KRI values were found.")

    return work.sort_values("date").reset_index(drop=True)


def create_predictive_frame(master):
    daily = (
        master[["date", KRI4_COL, KRI5_COL]]
        .dropna()
        .sort_values("date")
        .reset_index(drop=True)
    )

    if len(daily) < 8:
        raise ValueError(
            "At least 8 rows with both KRI4 chargebacks and KRI5 refund "
            "operations are required for the predictive feature set."
        )

    recent_dates = daily["date"].tail(8)
    if len(recent_dates) >= 2:
        gaps = recent_dates.diff().dropna().dt.days
        if not (gaps == 1).all():
            raise ValueError(
                "The latest 8 KRI4/KRI5 observations must be consecutive daily "
                "dates because the trained model uses daily lags."
            )

    work = daily.copy()

    work["current_chargebacks"] = work[KRI4_COL]
    work["current_refund_operations"] = work[KRI5_COL]

    for lag in [1, 2, 3, 7]:
        work[f"chargeback_lag_{lag}"] = work[KRI4_COL].shift(lag)
        work[f"refund_lag_{lag}"] = work[KRI5_COL].shift(lag)

    work["chargeback_mean_3"] = work[KRI4_COL].rolling(3).mean()
    work["chargeback_mean_7"] = work[KRI4_COL].rolling(7).mean()
    work["refund_mean_3"] = work[KRI5_COL].rolling(3).mean()
    work["refund_mean_7"] = work[KRI5_COL].rolling(7).mean()

    work["chargeback_std_7"] = work[KRI4_COL].rolling(7).std()
    work["refund_std_7"] = work[KRI5_COL].rolling(7).std()

    work["chargeback_change_1"] = work[KRI4_COL].diff(1)
    work["chargeback_change_3"] = work[KRI4_COL].diff(3)
    work["refund_change_1"] = work[KRI5_COL].diff(1)
    work["refund_change_3"] = work[KRI5_COL].diff(3)

    work["day_of_week"] = work["date"].dt.dayofweek
    work["day_sin"] = np.sin(2 * np.pi * work["day_of_week"] / 7)
    work["day_cos"] = np.cos(2 * np.pi * work["day_of_week"] / 7)

    latest = work.iloc[[-1]][FEATURES].copy()

    if latest.isna().any(axis=None):
        raise ValueError(
            "The latest observation does not have enough historical context "
            "to construct all predictive features."
        )

    return work, latest


def latest_composite_record(master):
    snapshot = (
        master[["date", KRI3_COL, KRI4_COL]]
        .dropna()
        .sort_values("date")
        .copy()
    )

    if snapshot.empty:
        raise ValueError(
            "No date contains both KRI3 Waiting Refunds After 14 Days and "
            "KRI4 Chargebacks. A composite RCSA score cannot be generated."
        )

    row = snapshot.iloc[-1].copy()

    kri3_class = classify_higher_is_worse(
        row[KRI3_COL], KRI3_Q1, KRI3_Q3
    )
    kri4_class = classify_higher_is_worse(
        row[KRI4_COL], KRI4_Q1, KRI4_Q3
    )

    kri3_score = int(risk_score(kri3_class))
    kri4_score = int(risk_score(kri4_class))

    composite_score = (
        kri3_score * KRI3_WEIGHT
        + kri4_score * KRI4_WEIGHT
    )
    composite_class = classify_composite(composite_score)

    return {
        "date": row["date"],
        "kri3_value": float(row[KRI3_COL]),
        "kri3_class": kri3_class,
        "kri3_score": kri3_score,
        "kri4_value": float(row[KRI4_COL]),
        "kri4_class": kri4_class,
        "kri4_score": kri4_score,
        "composite_score": float(composite_score),
        "composite_class": composite_class,
    }


def make_evidence_id(parts):
    raw = "|".join(str(x) for x in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("Model specification")
    st.write("**Predictive model:** " + config.get("selected_model", "Ridge Regression"))
    st.write("**Prediction target:** Next-day Historical Refund Operations")
    st.write("**Current-risk KRIs:** KRI3 + KRI4")
    st.write("**Predictive context:** KRI4 + KRI5 history")

    st.divider()
    st.write("**KRI3 — Waiting Refunds >14 days**")
    st.write(f"Green: ≤ {KRI3_Q1:,.2f}")
    st.write(f"Amber: > {KRI3_Q1:,.2f} to ≤ {KRI3_Q3:,.2f}")
    st.write(f"Red: > {KRI3_Q3:,.2f}")
    st.write(f"Weight: {KRI3_WEIGHT:.2f}")

    st.divider()
    st.write("**KRI4 — Daily Chargebacks**")
    st.write(f"Green: ≤ {KRI4_Q1:,.2f}")
    st.write(f"Amber: > {KRI4_Q1:,.2f} to ≤ {KRI4_Q3:,.2f}")
    st.write(f"Red: > {KRI4_Q3:,.2f}")
    st.write(f"Weight: {KRI4_WEIGHT:.2f}")

    st.divider()
    if model is None:
        st.error(
            "Model file not found. Add `Final_Multi_KRI_Predictive_Model.pkl` "
            "to the same GitHub directory as app.py."
        )
    else:
        st.success("Predictive model loaded")


# ============================================================
# HEADER
# ============================================================

st.title("Data-Driven Multi-KRI RCSA Dashboard")
st.caption(
    "Integrated deterministic KRI scoring, composite RCSA evidence, "
    "and next-day predictive operational monitoring."
)

st.info(
    "Proof-of-concept thresholds and KRI weights are illustrative and require "
    "organizational validation before production use."
)


tabs = st.tabs(
    [
        "Integrated RCSA",
        "KRI Monitoring",
        "Model Performance",
        "Model Governance",
    ]
)


# ============================================================
# TAB 1 — INTEGRATED RCSA
# ============================================================

with tabs[0]:
    st.subheader("Generate integrated RCSA decision support")

    uploaded = st.file_uploader(
        "Upload the multi-KRI history CSV",
        type=["csv"],
        help=(
            "Required columns: date, KRI3_waiting_refunds_after_14_days, "
            "KRI4_chargebacks, KRI5_historical_refund_operations. "
            "KRI3 may be blank on non-snapshot dates."
        ),
        key="integrated_upload",
    )

    if uploaded is None:
        st.info(
            "Upload a CSV to calculate the latest KRI3/KRI4 composite score "
            "and generate the KRI5 next-day forecast."
        )
    else:
        try:
            master = validate_master_data(pd.read_csv(uploaded))
            current = latest_composite_record(master)
            predictive_history, latest_features = create_predictive_frame(master)

            if model is None:
                st.error(
                    "Prediction cannot run because the saved model file is missing."
                )
            else:
                prediction = float(model.predict(latest_features)[0])
                latest_predictive_date = predictive_history["date"].iloc[-1]
                forecast_date = latest_predictive_date + pd.Timedelta(days=1)

                recent_refunds = predictive_history[KRI5_COL].tail(7)
                recent_mean = float(recent_refunds.mean())
                recent_std = float(recent_refunds.std())

                if recent_std > 0:
                    forecast_z = (prediction - recent_mean) / recent_std
                else:
                    forecast_z = np.nan

                predictive_signal = predictive_monitoring_signal(forecast_z)
                decision_signal = integrated_rcsa_signal(
                    current["composite_class"],
                    predictive_signal,
                )

                evidence_id = make_evidence_id(
                    [
                        current["date"],
                        current["composite_score"],
                        current["composite_class"],
                        forecast_date,
                        round(prediction, 6),
                    ]
                )

                st.write(
                    f"Latest composite-evidence date: **{current['date'].date()}** | "
                    f"Forecast date: **{forecast_date.date()}**"
                )

                c1, c2, c3, c4 = st.columns(4)
                c1.metric(
                    "KRI3 state",
                    current["kri3_class"],
                )
                c2.metric(
                    "KRI4 state",
                    current["kri4_class"],
                )
                c3.metric(
                    "Composite risk score",
                    f"{current['composite_score']:.2f}",
                )
                c4.metric(
                    "Composite state",
                    current["composite_class"],
                )

                p1, p2, p3 = st.columns(3)
                p1.metric(
                    "Predicted refund operations",
                    f"{prediction:,.2f}",
                )
                p2.metric(
                    "Recent 7-day mean",
                    f"{recent_mean:,.2f}",
                )
                p3.metric(
                    "Predictive context",
                    predictive_signal,
                )

                if current["composite_class"] == "Green":
                    st.success(decision_signal)
                elif current["composite_class"] == "Amber":
                    st.warning(decision_signal)
                else:
                    st.error(decision_signal)

                evidence = pd.DataFrame(
                    {
                        "Evidence ID": [evidence_id],
                        "Observed Date": [current["date"].date()],
                        "KRI3 Value": [current["kri3_value"]],
                        "KRI3 Class": [current["kri3_class"]],
                        "KRI3 Score": [current["kri3_score"]],
                        "KRI4 Value": [current["kri4_value"]],
                        "KRI4 Class": [current["kri4_class"]],
                        "KRI4 Score": [current["kri4_score"]],
                        "Composite Score": [current["composite_score"]],
                        "Composite Class": [current["composite_class"]],
                        "Forecast Date": [forecast_date.date()],
                        "Predicted Refund Operations": [round(prediction, 2)],
                        "Forecast Z-Score": [
                            None if pd.isna(forecast_z) else round(float(forecast_z), 3)
                        ],
                        "Predictive Context": [predictive_signal],
                        "RCSA Decision Support": [decision_signal],
                    }
                )

                st.subheader("Integrated evidence trail")
                st.dataframe(
                    evidence,
                    use_container_width=True,
                    hide_index=True,
                )

                csv_bytes = evidence.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download evidence trail",
                    data=csv_bytes,
                    file_name=f"RCSA_Evidence_{evidence_id}.csv",
                    mime="text/csv",
                )

                with st.expander("Latest predictive feature vector"):
                    feature_view = latest_features.T.reset_index()
                    feature_view.columns = ["Feature", "Value"]
                    st.dataframe(
                        feature_view,
                        use_container_width=True,
                        hide_index=True,
                    )

        except Exception as exc:
            st.error(str(exc))


# ============================================================
# TAB 2 — KRI MONITORING
# ============================================================

with tabs[1]:
    st.subheader("Historical multi-KRI monitoring")

    monitoring_upload = st.file_uploader(
        "Upload multi-KRI history for monitoring",
        type=["csv"],
        key="monitor_upload",
    )

    if monitoring_upload is None:
        st.info("Upload the same multi-KRI CSV to inspect historical patterns.")
    else:
        try:
            master = validate_master_data(pd.read_csv(monitoring_upload))

            daily = master[["date", KRI4_COL, KRI5_COL]].dropna().copy()

            m1, m2, m3 = st.columns(3)
            m1.metric("Rows", f"{len(master)}")
            m2.metric("KRI4 daily observations", f"{daily[KRI4_COL].count()}")
            m3.metric("KRI5 daily observations", f"{daily[KRI5_COL].count()}")

            st.subheader("Daily Chargebacks")
            st.line_chart(daily.set_index("date")[[KRI4_COL]])

            st.subheader("Historical Refund Operations")
            st.line_chart(daily.set_index("date")[[KRI5_COL]])

            snapshots = master[["date", KRI3_COL]].dropna().copy()
            st.subheader("Waiting Refunds After 14 Days")
            if snapshots.empty:
                st.info("No KRI3 snapshot observations are present.")
            else:
                st.line_chart(snapshots.set_index("date")[[KRI3_COL]])

            composite_rows = master[["date", KRI3_COL, KRI4_COL]].dropna().copy()
            if not composite_rows.empty:
                composite_rows["KRI3 Class"] = composite_rows[KRI3_COL].apply(
                    lambda v: classify_higher_is_worse(v, KRI3_Q1, KRI3_Q3)
                )
                composite_rows["KRI4 Class"] = composite_rows[KRI4_COL].apply(
                    lambda v: classify_higher_is_worse(v, KRI4_Q1, KRI4_Q3)
                )
                composite_rows["KRI3 Score"] = composite_rows["KRI3 Class"].map(
                    {"Green": 1, "Amber": 2, "Red": 3}
                )
                composite_rows["KRI4 Score"] = composite_rows["KRI4 Class"].map(
                    {"Green": 1, "Amber": 2, "Red": 3}
                )
                composite_rows["Composite Score"] = (
                    composite_rows["KRI3 Score"] * KRI3_WEIGHT
                    + composite_rows["KRI4 Score"] * KRI4_WEIGHT
                )
                composite_rows["Composite Class"] = composite_rows[
                    "Composite Score"
                ].apply(classify_composite)

                st.subheader("Composite RCSA history")
                st.line_chart(
                    composite_rows.set_index("date")[["Composite Score"]]
                )
                st.dataframe(
                    composite_rows.tail(30),
                    use_container_width=True,
                    hide_index=True,
                )

        except Exception as exc:
            st.error(str(exc))


# ============================================================
# TAB 3 — MODEL PERFORMANCE
# ============================================================

with tabs[2]:
    st.subheader("Predictive model validation")

    validation = config.get("validation", {})
    ridge = validation.get("ridge_regression", {})
    rf = validation.get("random_forest_regression", {})

    st.caption(
        validation.get(
            "split_method",
            "Chronological holdout validation",
        )
    )

    performance_df = pd.DataFrame(
        [
            {
                "Model": "Ridge Regression",
                "MAE": ridge.get("mae"),
                "RMSE": ridge.get("rmse"),
                "R²": ridge.get("r2"),
                "MAPE (%)": ridge.get("mape_percent"),
            },
            {
                "Model": "Random Forest Regression",
                "MAE": rf.get("mae"),
                "RMSE": rf.get("rmse"),
                "R²": rf.get("r2"),
                "MAPE (%)": rf.get("mape_percent"),
            },
        ]
    )

    st.dataframe(
        performance_df.style.format(
            {
                "MAE": "{:.2f}",
                "RMSE": "{:.2f}",
                "R²": "{:.3f}",
                "MAPE (%)": "{:.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    if ridge:
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Selected-model MAE", f"{ridge.get('mae', 0):.2f}")
        p2.metric("Selected-model RMSE", f"{ridge.get('rmse', 0):.2f}")
        p3.metric("Selected-model R²", f"{ridge.get('r2', 0):.3f}")
        p4.metric("Selected-model MAPE", f"{ridge.get('mape_percent', 0):.2f}%")

    st.info(
        "The regression metrics evaluate next-day Historical Refund Operations "
        "during the proof-of-concept holdout period. They do not measure the "
        "accuracy of the complete organizational RCSA process."
    )


# ============================================================
# TAB 4 — GOVERNANCE
# ============================================================

with tabs[3]:
    st.subheader("Model governance and interpretation")

    st.markdown(
        """
**Purpose**
- Convert observed KRI3 and KRI4 values into standardized Green/Amber/Red evidence.
- Aggregate current KRI scores using configurable proof-of-concept weights.
- Forecast next-day Historical Refund Operations using KRI4/KRI5 daily history.
- Combine current risk evidence with forward-looking operational context.
- Preserve a traceable evidence trail for RCSA review.

**Current proof-of-concept limitations**
- KRI3 and KRI4 thresholds are empirical quartile thresholds, not formally approved risk-appetite limits.
- KRI3 and KRI4 use illustrative 0.50/0.50 weights.
- KRI5 is treated as predictive operational context; a high forecast is not automatically interpreted as high risk.
- KRI1 and KRI2 are not included because their official business definitions and risk directions are not yet confirmed.
- Production use requires governed source-system feeds, approved thresholds and weights, access controls, model-version controls, stakeholder validation, and ongoing performance monitoring.
- The model supports professional judgement; it does not replace accountable risk-owner decisions.
        """
    )

    with st.expander("Predictive feature set"):
        st.write(FEATURES)

    with st.expander("Saved configuration"):
        st.json(config)
