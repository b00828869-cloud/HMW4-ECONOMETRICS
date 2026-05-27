import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px

from statsmodels.tsa.stattools import grangercausalitytests

from src.data_loader import list_csv_files, load_stock_csv
from src.preprocessing import create_realized_variance


COLOR_ACTUAL = "#9CA3AF"
COLOR_HAR = "#7C3AED"
COLOR_HAR_X = "#F97316"


st.set_page_config(
    page_title="Q11-Q12 - Granger causality",
    page_icon="🔗",
    layout="wide"
)

st.title("🔗 Questions 11 & 12 — Granger causality and richer forecasting models")

st.markdown("""
### Objective

The objective is to test whether the past volatility of one stock helps predict the volatility of another stock.

### Granger causality idea

A stock X is said to Granger-cause stock Y if past values of X contain useful information for predicting Y, beyond the past values of Y itself.

### Hypotheses

$$
H_0: X \\text{ does not Granger-cause } Y
$$

$$
H_1: X \\text{ Granger-causes } Y
$$

If the p-value is below 0.05, we reject the null hypothesis.

### Forecasting extension

If Granger causality is detected, we compare:

- HAR model using only the target stock,
- HAR-X model using the target stock plus the explanatory stock.
""")

csv_files = list_csv_files()

if not csv_files:
    st.error("No CSV files found.")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    target_file = st.selectbox(
        "Target stock to forecast",
        csv_files,
        format_func=lambda x: x.stem,
        index=0
    )

with col2:
    source_file = st.selectbox(
        "Potential explanatory stock",
        csv_files,
        format_func=lambda x: x.stem,
        index=1
    )

max_lag = st.slider(
    "Maximum lag for Granger causality test",
    min_value=1,
    max_value=10,
    value=5
)

if target_file == source_file:
    st.warning("Please select two different stocks.")
    st.stop()


def prepare_lnrv(file):
    df = load_stock_csv(file)
    _, rv_df = create_realized_variance(df)

    out = rv_df[["day_id", "ln_rv"]].copy()
    out = out.rename(columns={"ln_rv": file.stem})

    return out


target_lnrv = prepare_lnrv(target_file)
source_lnrv = prepare_lnrv(source_file)

min_len = min(len(target_lnrv), len(source_lnrv))

target_lnrv = target_lnrv.tail(min_len).reset_index(drop=True)
source_lnrv = source_lnrv.tail(min_len).reset_index(drop=True)

combined_df = pd.DataFrame({
    "Target": target_lnrv[target_file.stem],
    "Source": source_lnrv[source_file.stem]
}).dropna()

st.subheader("Granger causality tests")

test_data_source_to_target = combined_df[["Target", "Source"]]
test_data_target_to_source = combined_df[["Source", "Target"]]

result_source_to_target = grangercausalitytests(
    test_data_source_to_target,
    maxlag=max_lag,
    verbose=False
)

result_target_to_source = grangercausalitytests(
    test_data_target_to_source,
    maxlag=max_lag,
    verbose=False
)


def extract_granger_results(results, direction):
    rows = []

    for lag, output in results.items():
        p_value = output[0]["ssr_ftest"][1]

        rows.append({
            "Direction": direction,
            "Lag": lag,
            "p-value": p_value,
            "Reject H0 at 5%": p_value < 0.05
        })

    return rows


granger_rows = []

granger_rows += extract_granger_results(
    result_source_to_target,
    f"{source_file.stem} → {target_file.stem}"
)

granger_rows += extract_granger_results(
    result_target_to_source,
    f"{target_file.stem} → {source_file.stem}"
)

granger_df = pd.DataFrame(granger_rows)

st.dataframe(
    granger_df,
    use_container_width=True
)

detected_df = granger_df[
    granger_df["Reject H0 at 5%"]
]

if detected_df.empty:
    st.warning("""
No Granger causality is detected at the 5% level for the selected pair and lag range.
""")
    st.stop()

st.success("""
At least one significant Granger causality relationship is detected at the 5% level.
""")

st.subheader("Out-of-sample forecasting with richer model")

st.markdown(f"""
We now test whether adding **{source_file.stem}** improves the forecast of **{target_file.stem}**.
""")

model_df = pd.DataFrame({
    "target_y": combined_df["Target"],
    "source_y": combined_df["Source"]
})

model_df["target_lag_1"] = model_df["target_y"].shift(1)
model_df["target_weekly_avg"] = model_df["target_y"].rolling(5).mean().shift(1)
model_df["target_monthly_avg"] = model_df["target_y"].rolling(22).mean().shift(1)

model_df["source_lag_1"] = model_df["source_y"].shift(1)
model_df["source_weekly_avg"] = model_df["source_y"].rolling(5).mean().shift(1)
model_df["source_monthly_avg"] = model_df["source_y"].rolling(22).mean().shift(1)

model_df = model_df.dropna().reset_index(drop=True)

split_index = len(model_df) // 2


def rolling_forecast(data, split_idx, x_cols):
    predictions = []
    actuals = []

    for i in range(len(data) - split_idx):
        expanding_df = data.iloc[:split_idx + i]
        current_obs = data.iloc[split_idx + i]

        X_train = expanding_df[x_cols]
        X_train = sm.add_constant(X_train)

        y_train = expanding_df["target_y"]

        model = sm.OLS(y_train, X_train).fit()

        X_test = pd.DataFrame(
            [current_obs[x_cols].values],
            columns=x_cols
        )

        X_test = sm.add_constant(X_test, has_constant="add")

        forecast = model.predict(X_test).iloc[0]

        predictions.append(forecast)
        actuals.append(current_obs["target_y"])

    return np.array(actuals), np.array(predictions)


har_cols = [
    "target_lag_1",
    "target_weekly_avg",
    "target_monthly_avg"
]

har_x_cols = [
    "target_lag_1",
    "target_weekly_avg",
    "target_monthly_avg",
    "source_lag_1",
    "source_weekly_avg",
    "source_monthly_avg"
]

actuals, har_pred = rolling_forecast(
    model_df,
    split_index,
    har_cols
)

_, har_x_pred = rolling_forecast(
    model_df,
    split_index,
    har_x_cols
)


def compute_metrics(actual, pred):
    error = actual - pred
    mfe = error.mean()
    rmse = np.sqrt(np.mean(error ** 2))

    return mfe, rmse


har_mfe, har_rmse = compute_metrics(actuals, har_pred)
har_x_mfe, har_x_rmse = compute_metrics(actuals, har_x_pred)

metrics_df = pd.DataFrame({
    "Model": ["HAR", "HAR-X with Granger stock"],
    "MFE": [har_mfe, har_x_mfe],
    "RMSE": [har_rmse, har_x_rmse]
})

st.subheader("Forecast performance comparison")

st.dataframe(
    metrics_df,
    use_container_width=True
)

st.subheader("Actual vs forecasted lnRV")

forecast_df = pd.DataFrame({
    "Observation": range(len(actuals)),
    "Actual lnRV": actuals,
    "HAR forecast": har_pred,
    "HAR-X forecast": har_x_pred
})

MAX_POINTS = 1200

plot_df = forecast_df.iloc[
    ::max(1, len(forecast_df) // MAX_POINTS)
]

chart_option = st.radio(
    "Choose forecast comparison",
    [
        "HAR vs actual",
        "HAR-X vs actual",
        "HAR vs HAR-X vs actual"
    ],
    horizontal=True
)

if chart_option == "HAR vs actual":
    y_cols = ["Actual lnRV", "HAR forecast"]
    colors = [COLOR_ACTUAL, COLOR_HAR]

elif chart_option == "HAR-X vs actual":
    y_cols = ["Actual lnRV", "HAR-X forecast"]
    colors = [COLOR_ACTUAL, COLOR_HAR_X]

else:
    y_cols = ["Actual lnRV", "HAR forecast", "HAR-X forecast"]
    colors = [COLOR_ACTUAL, COLOR_HAR, COLOR_HAR_X]

fig = px.line(
    plot_df,
    x="Observation",
    y=y_cols,
    title=f"Forecast comparison — {source_file.stem} helping {target_file.stem}",
    render_mode="webgl",
    color_discrete_sequence=colors
)

fig.update_layout(
    template="plotly_white",
    height=500,
    xaxis_title="Out-of-sample observation",
    yaxis_title="lnRV",
    legend_title="Series",
    hovermode="x unified"
)

fig.data[0].opacity = 0.15
fig.data[0].line.width = 1

for trace in fig.data[1:]:
    trace.opacity = 1
    trace.line.width = 3

st.plotly_chart(
    fig,
    use_container_width=True
)

st.subheader("Conclusion")

if har_x_rmse < har_rmse:
    st.success(f"""
For this pair, the richer HAR-X model improves forecasting performance.

The RMSE decreases from **{har_rmse:.6f}** to **{har_x_rmse:.6f}**.

This suggests that past volatility from **{source_file.stem}** contains useful information for forecasting **{target_file.stem}**.
""")
else:
    st.warning(f"""
For this pair, the richer HAR-X model does not improve forecasting performance.

The RMSE increases from **{har_rmse:.6f}** to **{har_x_rmse:.6f}**.

This suggests that Granger causality in-sample does not necessarily translate into better out-of-sample forecasts.
""")