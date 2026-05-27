import streamlit as st
import pandas as pd

from statsmodels.tsa.stattools import adfuller

from src.data_loader import list_csv_files, load_stock_csv
from src.preprocessing import create_realized_variance


st.set_page_config(
    page_title="Q4 - ADF stationarity test",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Question 4 — Stationarity analysis using ADF tests")

st.markdown("""
### Objective

The objective is to determine whether the lnRV series are stationary.

### Why is stationarity important?

Many econometric forecasting models such as:
- AR models,
- HAR models,
- and several machine learning approaches

assume that the statistical properties of the series remain stable over time.

A non-stationary series may lead to misleading regressions and unstable forecasts.

### Methodology

We use the Augmented Dickey-Fuller (ADF) test.

The null hypothesis is:

$$
H_0 : \\text{The series contains a unit root (non-stationary)}
$$

The alternative hypothesis is:

$$
H_1 : \\text{The series is stationary}
$$

### Decision rule

- If the p-value < 0.05 → reject \(H_0\) → the series is stationary.
- If the p-value ≥ 0.05 → fail to reject \(H_0\).
""")

csv_files = list_csv_files()

selected_file = st.selectbox(
    "Select a stock",
    csv_files,
    format_func=lambda x: x.stem
)

df = load_stock_csv(selected_file)

_, rv_df = create_realized_variance(df)

lnrv_series = rv_df["ln_rv"]

# ---------------------------------------------------
# ADF TEST
# ---------------------------------------------------

adf_result = adfuller(lnrv_series)

adf_stat = adf_result[0]
p_value = adf_result[1]
lags_used = adf_result[2]
n_obs = adf_result[3]

critical_values = adf_result[4]

# ---------------------------------------------------
# RESULTS
# ---------------------------------------------------

st.subheader("ADF test results")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("ADF statistic", f"{adf_stat:.4f}")

with col2:
    st.metric("p-value", f"{p_value:.6f}")

with col3:
    st.metric("Lags used", lags_used)

with col4:
    st.metric("Observations", n_obs)

# ---------------------------------------------------
# CRITICAL VALUES
# ---------------------------------------------------

st.subheader("Critical values")

critical_df = pd.DataFrame({
    "Confidence level": critical_values.keys(),
    "Critical value": critical_values.values()
})

st.dataframe(
    critical_df,
    use_container_width=True
)

# ---------------------------------------------------
# DECISION
# ---------------------------------------------------

st.subheader("Conclusion")

if p_value < 0.05:
    st.success(f"""
The p-value is below 0.05.

We reject the null hypothesis of a unit root.

The lnRV series for {selected_file.stem} appears to be stationary.
""")
else:
    st.warning(f"""
The p-value is above 0.05.

We fail to reject the null hypothesis.

The lnRV series for {selected_file.stem} may be non-stationary.
""")