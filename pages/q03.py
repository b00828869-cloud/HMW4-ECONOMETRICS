import streamlit as st
import plotly.express as px
import pandas as pd

from scipy.stats import skew, kurtosis
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt

from src.data_loader import list_csv_files, load_stock_csv
from src.preprocessing import create_realized_variance
from src.style import COLOR_PRIMARY


st.set_page_config(
    page_title="Q3 - lnRV analysis",
    page_icon="📉",
    layout="wide"
)

st.title("📉 Question 3 — Log-realized variance analysis")

st.markdown("""
### Objective

The objective is to study the statistical properties of the logarithm of realized variance.

### Why use logarithms?

Realized variance is highly skewed and contains extreme values.

Applying the natural logarithm stabilizes the variance and produces a series that is easier to model statistically.

### Methodology

We transform realized variance using:

$$
lnRV_t = \\ln(RV_t)
$$

We then study the autocorrelation structure of the series using the sample autocorrelation function:

$$
ACF(k) =
\\frac{
Cov(lnRV_t, lnRV_{t-k})
}{
Var(lnRV_t)
}
$$

where:
- \(k\) is the lag,
- \(Cov\) is the covariance,
- \(Var\) is the variance of the series.

### Interpretation

A slowly decaying autocorrelation function indicates volatility persistence, meaning that periods of high volatility tend to be followed by high volatility.
""")
csv_files = list_csv_files()

selected_file = st.selectbox(
    "Select a stock",
    csv_files,
    format_func=lambda x: x.stem
)

df = load_stock_csv(selected_file)

_, rv_df = create_realized_variance(df)

lnrv_df = rv_df.copy()

MAX_POINTS = 5000

lnrv_plot = lnrv_df.iloc[
    ::max(1, len(lnrv_df) // MAX_POINTS)
]

# ---------------------------------------------------
# LN RV TIME SERIES
# ---------------------------------------------------

st.subheader("lnRV time series")

fig_lnrv = px.line(
    lnrv_plot,
    x="day_id",
    y="ln_rv",
    title=f"lnRV time series — {selected_file.stem}",
    render_mode="webgl",
    color_discrete_sequence=[COLOR_PRIMARY]
)

fig_lnrv.update_layout(
    template="plotly_white",
    height=450,
    xaxis_title="Artificial trading day",
    yaxis_title="ln(RV)"
)

st.plotly_chart(fig_lnrv, use_container_width=True)

# ---------------------------------------------------
# AUTOCORRELATION PLOT
# ---------------------------------------------------
st.subheader("Sample autocorrelation function")

fig_acf, ax = plt.subplots(figsize=(10, 4))

plot_acf(
    lnrv_df["ln_rv"],
    lags=40,
    ax=ax
)

ax.set_facecolor("white")

for line in ax.lines:
    line.set_color(COLOR_PRIMARY)

for collection in ax.collections:
    try:
        collection.set_color(COLOR_PRIMARY)
    except:
        pass

ax.set_title(
    "Sample autocorrelation function",
    fontsize=18,
    fontweight="bold"
)

ax.grid(
    alpha=0.2
)

fig_acf.patch.set_facecolor("white")

st.pyplot(fig_acf)
# ---------------------------------------------------
# DESCRIPTIVE STATISTICS
# ---------------------------------------------------

st.subheader("Descriptive statistics — first four moments")

lnrv_series = lnrv_df["ln_rv"]

stats_df = pd.DataFrame({
    "Statistic": [
        "Mean",
        "Variance",
        "Skewness",
        "Kurtosis"
    ],
    "Value": [
        lnrv_series.mean(),
        lnrv_series.var(),
        skew(lnrv_series),
        kurtosis(lnrv_series)
    ]
})

st.dataframe(
    stats_df,
    use_container_width=True
)