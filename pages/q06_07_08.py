import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px

from src.data_loader import list_csv_files, load_stock_csv
from src.preprocessing import create_realized_variance


COLOR_ACTUAL = "#9CA3AF"
COLOR_HAR = "#7C3AED"
COLOR_AR1 = "#2563EB"
COLOR_RW = "#F97316"


st.set_page_config(
    page_title="Q6-Q8 - Forecast comparison",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Questions 6, 7 & 8 — Forecast comparison")

st.markdown("""
### Objective

The objective is to perform the same out-of-sample forecasting exercise with three models:

- HAR model
- AR(1) model
- Random Walk model

### Forecasting setup

For all models:

- the first 50% of observations are used for estimation,
- the last 50% are used for out-of-sample forecasting,
- the forecasting horizon is \(h = 1\),
- the model is re-estimated after every forecast when parameters need to be estimated.

### Models

HAR model:

$$
y_t =
\\beta_0
+
\\beta_1 y_{t-1}
+
\\beta_2 y^{(5)}_{t-1}
+
\\beta_3 y^{(22)}_{t-1}
+
\\varepsilon_t
$$

AR(1) model:

$$
y_t =
\\beta_0
+
\\beta_1 y_{t-1}
+
\\varepsilon_t
$$

Random Walk model:

$$
\\hat{y}_{t+1} = y_t
$$

### Evaluation metrics

$$
MFE =
\\frac{1}{N}
\\sum_{t=1}^{N}
(y_t - \\hat{y}_t)
$$

$$
RMSE =
\\sqrt{
\\frac{1}{N}
\\sum_{t=1}^{N}
(y_t - \\hat{y}_t)^2
}
$$
""")

csv_files = list_csv_files()

if not csv_files:
    st.error("No CSV files found.")
    st.stop()

selected_file = st.selectbox(
    "Select a stock",
    csv_files,
    format_func=lambda x: x.stem
)

df = load_stock_csv(selected_file)

_, rv_df = create_realized_variance(df)

base_df = rv_df.copy()
base_df["y"] = base_df["ln_rv"]

base_df["lag_1"] = base_df["y"].shift(1)

base_df["weekly_avg"] = (
    base_df["y"]
    .rolling(5)
    .mean()
    .shift(1)
)

base_df["monthly_avg"] = (
    base_df["y"]
    .rolling(22)
    .mean()
    .shift(1)
)

model_df = base_df.dropna().reset_index(drop=True)

split_index = len(model_df) // 2


def rolling_forecast(data, split_idx, model_type):
    predictions = []
    actuals = []

    for i in range(len(data) - split_idx):
        expanding_df = data.iloc[:split_idx + i]
        current_obs = data.iloc[split_idx + i]

        actuals.append(current_obs["y"])

        if model_type == "HAR":
            x_cols = ["lag_1", "weekly_avg", "monthly_avg"]

            X_train = expanding_df[x_cols]
            X_train = sm.add_constant(X_train)

            y_train = expanding_df["y"]

            model = sm.OLS(y_train, X_train).fit()

            X_test = pd.DataFrame(
                [current_obs[x_cols].values],
                columns=x_cols
            )

            X_test = sm.add_constant(X_test, has_constant="add")

            forecast = model.predict(X_test).iloc[0]

        elif model_type == "AR(1)":
            x_cols = ["lag_1"]

            X_train = expanding_df[x_cols]
            X_train = sm.add_constant(X_train)

            y_train = expanding_df["y"]

            model = sm.OLS(y_train, X_train).fit()

            X_test = pd.DataFrame(
                [current_obs[x_cols].values],
                columns=x_cols
            )

            X_test = sm.add_constant(X_test, has_constant="add")

            forecast = model.predict(X_test).iloc[0]

        elif model_type == "Random Walk":
            forecast = current_obs["lag_1"]

        else:
            raise ValueError("Unknown model type")

        predictions.append(forecast)

    return predictions, actuals


har_predictions, actuals = rolling_forecast(
    model_df,
    split_index,
    "HAR"
)

ar1_predictions, _ = rolling_forecast(
    model_df,
    split_index,
    "AR(1)"
)

rw_predictions, _ = rolling_forecast(
    model_df,
    split_index,
    "Random Walk"
)

forecast_df = pd.DataFrame({
    "Observation": range(len(actuals)),
    "Actual lnRV": actuals,
    "HAR forecast": har_predictions,
    "AR(1) forecast": ar1_predictions,
    "Random Walk forecast": rw_predictions
})

forecast_df["HAR error"] = (
    forecast_df["Actual lnRV"] - forecast_df["HAR forecast"]
)

forecast_df["AR(1) error"] = (
    forecast_df["Actual lnRV"] - forecast_df["AR(1) forecast"]
)

forecast_df["Random Walk error"] = (
    forecast_df["Actual lnRV"] - forecast_df["Random Walk forecast"]
)


def compute_metrics(error_series):
    mfe = error_series.mean()
    rmse = np.sqrt(np.mean(error_series ** 2))
    return mfe, rmse


har_mfe, har_rmse = compute_metrics(forecast_df["HAR error"])
ar1_mfe, ar1_rmse = compute_metrics(forecast_df["AR(1) error"])
rw_mfe, rw_rmse = compute_metrics(forecast_df["Random Walk error"])


st.subheader("Forecast performance comparison")

metrics_df = pd.DataFrame({
    "Model": ["HAR", "AR(1)", "Random Walk"],
    "MFE": [har_mfe, ar1_mfe, rw_mfe],
    "RMSE": [har_rmse, ar1_rmse, rw_rmse]
})

st.dataframe(
    metrics_df,
    use_container_width=True
)

st.subheader("Actual vs forecasted lnRV")

chart_option = st.radio(
    "Choose the forecast comparison to display",
    [
        "HAR vs actual",
        "AR(1) vs actual",
        "Random Walk vs actual",
        "All models vs actual"
    ],
    horizontal=True
)

MAX_POINTS = 1200

plot_df = forecast_df.iloc[
    ::max(1, len(forecast_df) // MAX_POINTS)
]

if chart_option == "HAR vs actual":
    y_cols = ["Actual lnRV", "HAR forecast"]
    colors = [COLOR_ACTUAL, COLOR_HAR]

elif chart_option == "AR(1) vs actual":
    y_cols = ["Actual lnRV", "AR(1) forecast"]
    colors = [COLOR_ACTUAL, COLOR_AR1]

elif chart_option == "Random Walk vs actual":
    y_cols = ["Actual lnRV", "Random Walk forecast"]
    colors = [COLOR_ACTUAL, COLOR_RW]

else:
    y_cols = [
        "Actual lnRV",
        "HAR forecast",
        "AR(1) forecast",
        "Random Walk forecast"
    ]
    colors = [
        COLOR_ACTUAL,
        COLOR_HAR,
        COLOR_AR1,
        COLOR_RW
    ]

fig = px.line(
    plot_df,
    x="Observation",
    y=y_cols,
    title=f"Out-of-sample forecast comparison — {selected_file.stem}",
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

st.subheader("Comment")

best_model = metrics_df.loc[
    metrics_df["RMSE"].idxmin(),
    "Model"
]

st.markdown(f"""
For **{selected_file.stem}**, the model with the lowest RMSE is **{best_model}**.

The RMSE is used to compare overall forecasting accuracy, while the MFE indicates whether a model tends to overestimate or underestimate future \(lnRV\).
""")