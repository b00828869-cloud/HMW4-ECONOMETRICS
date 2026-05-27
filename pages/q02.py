import streamlit as st
import plotly.express as px
import pandas as pd
from scipy.stats import skew, kurtosis

from src.data_loader import list_csv_files, load_stock_csv
from src.preprocessing import create_realized_variance
from src.style import COLOR_PRIMARY


st.set_page_config(
    page_title="Q2 - RV statistics",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Question 2 — RV time series and descriptive statistics")

st.markdown("""
### Objective

The objective is to analyze the realized variance series created in Question 1.

### Why is this useful?

Before forecasting volatility, we need to understand how realized variance behaves over time.

This helps us identify:
- volatility spikes,
- asymmetry in the distribution,
- extreme values,
- and whether the series is stable or highly irregular.

### Methodology

We use the daily realized variance series:

$$
RV_t = \\sum_{i=1}^{78} r_{t,i}^2
$$

Then, we compute the first four moments:

1. Mean  
2. Variance  
3. Skewness  
4. Kurtosis  

### Interpretation

- The **mean** gives the average level of realized variance.
- The **variance** shows how unstable RV is over time.
- The **skewness** indicates whether extreme values are mostly on the right side.
- The **kurtosis** captures the presence of extreme volatility events.
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

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Stock", selected_file.stem)

with col2:
    st.metric("RV observations", f"{len(rv_df):,}")

with col3:
    st.metric("Average RV", f"{rv_df['rv'].mean():.8f}")

st.divider()

MAX_POINTS = 5000

rv_plot = rv_df.iloc[
    ::max(1, len(rv_df) // MAX_POINTS)
]

st.subheader("Realized variance time series")

fig_rv = px.line(
    rv_plot,
    x="day_id",
    y="rv",
    title=f"Realized variance time series — {selected_file.stem}",
    render_mode="webgl",
    color_discrete_sequence=[COLOR_PRIMARY]
)

fig_rv.update_layout(
    template="plotly_white",
    height=450,
    xaxis_title="Artificial trading day",
    yaxis_title="Realized variance"
)

st.plotly_chart(fig_rv, use_container_width=True)

st.subheader("Distribution of realized variance")

fig_hist = px.histogram(
    rv_df,
    x="rv",
    nbins=80,
    title=f"Distribution of realized variance — {selected_file.stem}",
    color_discrete_sequence=[COLOR_PRIMARY]
)

fig_hist.update_layout(
    template="plotly_white",
    height=420,
    xaxis_title="Realized variance",
    yaxis_title="Frequency"
)

st.plotly_chart(fig_hist, use_container_width=True)

st.subheader("Descriptive statistics — first four moments")

rv_series = rv_df["rv"]

stats_df = pd.DataFrame({
    "Statistic": [
        "Mean",
        "Variance",
        "Skewness",
        "Kurtosis"
    ],
    "Value": [
        rv_series.mean(),
        rv_series.var(),
        skew(rv_series),
        kurtosis(rv_series)
    ]
})

st.dataframe(
    stats_df,
    use_container_width=True
)

st.subheader("Interpretation")

skew_val = skew(rv_series)
kurt_val = kurtosis(rv_series)

if skew_val > 1:
    skew_text = "The distribution is strongly right-skewed, meaning that most days have low volatility but a few days exhibit very high realized variance."
else:
    skew_text = "The distribution is not strongly skewed."

if kurt_val > 3:
    kurt_text = "The high kurtosis indicates heavy tails, meaning that extreme volatility events are present in the data."
else:
    kurt_text = "The kurtosis is relatively moderate, suggesting fewer extreme volatility events."

st.info(f"""
For **{selected_file.stem}**, the realized variance series shows an average RV of **{rv_series.mean():.8f}**.

{skew_text}

{kurt_text}
""")

st.success("""
Q2 completed.

The app displays the RV time series and reports the first four moments of the realized variance distribution.
""")