import streamlit as st
import plotly.express as px

from src.data_loader import list_csv_files, load_stock_csv
from src.preprocessing import create_realized_variance
from src.style import COLOR_PRIMARY


st.set_page_config(
    page_title="Q1 - Realized variance",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Question 1 — Five-minute returns and realized variance")

st.markdown("""
### Objective

The objective is to transform high-frequency stock prices into a daily volatility measure called **realized variance**.

### Why five-minute returns?

Five-minute returns measure short-term price changes during the trading day.

They are useful because they capture intraday volatility much better than one single daily return.

### Methodology

We use the close price and compute log returns:

$$
r_t = \\ln(P_t) - \\ln(P_{t-1})
$$

Then, for each trading day, we compute realized variance:

$$
RV_t = \\sum_{i=1}^{78} r_{t,i}^2
$$

Since the raw files do not contain timestamps, we assume that observations are ordered and regularly spaced.

We use **78 five-minute intervals per trading day**, which corresponds to a standard US trading day.

### Interpretation

A high RV means that the stock had strong intraday price movements.

A low RV means that the stock was relatively stable during the day.
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

returns_df, rv_df = create_realized_variance(df)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Stock", selected_file.stem)

with col2:
    st.metric("Raw observations used", f"{len(df.tail(300000)):,}")

with col3:
    st.metric("5-minute returns", f"{len(returns_df):,}")

with col4:
    st.metric("RV observations", f"{len(rv_df):,}")

st.divider()

MAX_POINTS = 5000

returns_plot = returns_df.iloc[
    ::max(1, len(returns_df) // MAX_POINTS)
]

rv_plot = rv_df.iloc[
    ::max(1, len(rv_df) // MAX_POINTS)
]

st.subheader("Five-minute return time series")

fig_returns = px.line(
    returns_plot.reset_index(),
    x="index",
    y="return_5min",
    title=f"Five-minute log returns — {selected_file.stem}",
    render_mode="webgl",
    color_discrete_sequence=[COLOR_PRIMARY]
)

fig_returns.update_layout(
    template="plotly_white",
    height=420,
    xaxis_title="Observation index",
    yaxis_title="Five-minute log return"
)

st.plotly_chart(fig_returns, use_container_width=True)

st.subheader("Realized variance time series")

fig_rv = px.line(
    rv_plot,
    x="day_id",
    y="rv",
    title=f"Daily realized variance — {selected_file.stem}",
    render_mode="webgl",
    color_discrete_sequence=[COLOR_PRIMARY]
)

fig_rv.update_layout(
    template="plotly_white",
    height=420,
    xaxis_title="Artificial trading day",
    yaxis_title="Realized variance"
)

st.plotly_chart(fig_rv, use_container_width=True)

st.subheader("Output samples")

tab1, tab2 = st.tabs(["Five-minute returns", "Realized variance"])

with tab1:
    st.dataframe(
        returns_df[["price", "log_price", "return_5min", "day_id"]].head(20),
        use_container_width=True
    )

with tab2:
    st.dataframe(
        rv_df.head(20),
        use_container_width=True
    )

st.success("""
Q1 completed.

The app creates five-minute log returns from close prices and aggregates their squared values into realized variances.
""")