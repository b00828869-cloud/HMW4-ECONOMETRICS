import streamlit as st
import pandas as pd
import statsmodels.api as sm
import plotly.express as px

from src.data_loader import list_csv_files, load_stock_csv
from src.preprocessing import create_realized_variance
from src.style import COLOR_PRIMARY


COLOR_SECONDARY = "#111827"


st.set_page_config(
    page_title="Q5 - HAR model",
    page_icon="📘",
    layout="wide"
)

st.title("📘 Question 5 — HAR model estimation")

st.markdown("""
### Objective

The objective is to estimate the HAR model of Corsi et al. for the \(lnRV_t\) series on the full available sample.

### Why HAR models?

The HAR model is designed to capture volatility persistence over several horizons:
- daily volatility,
- weekly volatility,
- monthly volatility.

This is useful because financial volatility often depends not only on yesterday's volatility, but also on average volatility over longer recent periods.

### Model specification

The estimated model is:

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

with:

$$
y^{(n)}_{t-1}
=
\\frac{
y_{t-1} + y_{t-2} + ... + y_{t-n}
}{n}
$$

where \(y_t = lnRV_t\).

### Interpretation

- \(y_{t-1}\) captures the daily component.
- \(y^{(5)}_{t-1}\) captures the weekly component.
- \(y^{(22)}_{t-1}\) captures the monthly component.
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

har_df = rv_df.copy()
har_df["y"] = har_df["ln_rv"]

har_df["lag_1"] = har_df["y"].shift(1)

har_df["weekly_avg"] = (
    har_df["y"]
    .rolling(5)
    .mean()
    .shift(1)
)

har_df["monthly_avg"] = (
    har_df["y"]
    .rolling(22)
    .mean()
    .shift(1)
)

har_df = har_df.dropna()

X = har_df[["lag_1", "weekly_avg", "monthly_avg"]]
X = sm.add_constant(X)

y = har_df["y"]

model = sm.OLS(y, X).fit()

st.subheader("HAR coefficient estimates")

coef_df = pd.DataFrame({
    "Parameter": ["β0", "β1", "β2", "β3"],
    "Variable": [
        "Constant",
        "Daily component: y(t-1)",
        "Weekly component: y(t-1)^5",
        "Monthly component: y(t-1)^22"
    ],
    "Estimate": model.params.values,
    "t-statistic": model.tvalues.values,
    "p-value": model.pvalues.values
})

st.dataframe(
    coef_df,
    use_container_width=True
)

st.subheader("Actual vs fitted lnRV")

har_df["fitted_lnrv"] = model.fittedvalues

plot_df = har_df[["day_id", "y", "fitted_lnrv"]].copy()

plot_df = plot_df.rename(
    columns={
        "y": "Actual lnRV",
        "fitted_lnrv": "Fitted lnRV"
    }
)

MAX_POINTS = 5000

plot_df = plot_df.iloc[
    ::max(1, len(plot_df) // MAX_POINTS)
]

fig_fit = px.line(
    plot_df,
    x="day_id",
    y=["Actual lnRV", "Fitted lnRV"],
    title=f"Actual vs fitted lnRV — {selected_file.stem}",
    render_mode="webgl",
    color_discrete_sequence=[COLOR_PRIMARY, COLOR_SECONDARY]
)

fig_fit.update_layout(
    template="plotly_white",
    height=450,
    xaxis_title="Artificial trading day",
    yaxis_title="lnRV",
    legend_title="Series"
)

fig_fit.data[0].opacity = 0.35
fig_fit.data[0].line.width = 1.2

fig_fit.data[1].opacity = 1
fig_fit.data[1].line.width = 2.8

st.plotly_chart(fig_fit, use_container_width=True)

st.subheader("Model fit")

fit_df = pd.DataFrame({
    "Metric": [
        "R-squared",
        "Adjusted R-squared"
    ],
    "Value": [
        model.rsquared,
        model.rsquared_adj
    ]
})

st.dataframe(
    fit_df,
    use_container_width=True
)

st.subheader("Comment")

st.markdown(f"""
For **{selected_file.stem}**, the coefficient table reports the estimated parameters of the HAR model.

The comparison between actual and fitted \(lnRV\) shows how well the model reproduces the dynamics of realized volatility on the estimation sample.
""")