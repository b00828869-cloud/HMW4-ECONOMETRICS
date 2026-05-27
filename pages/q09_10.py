import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

from src.data_loader import list_csv_files, load_stock_csv
from src.preprocessing import create_realized_variance


COLOR_PRIMARY = "#7C3AED"


st.set_page_config(
    page_title="Q9-Q10 - Model comparison",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Questions 9 & 10 — Machine Learning vs Linear Models")

st.markdown("""
### Objective

The objective is to test whether machine learning models can outperform the previous linear models.

We compare:

- HAR
- AR(1)
- Random Walk
- Random Forest
- Gradient Boosting

### Forecasting setup

For each stock:

- the first 50% of observations are used for estimation,
- the last 50% are used for out-of-sample forecasting,
- the forecast horizon is \(h = 1\),
- models are evaluated using MFE and RMSE.

### Machine learning models

Machine learning models are tested because they can capture non-linear relationships between volatility features.
""")

csv_files = list_csv_files()

if not csv_files:
    st.error("No CSV files found.")
    st.stop()

selected_files = st.multiselect(
    "Select stocks to include",
    csv_files,
    default=csv_files[:5],
    format_func=lambda x: x.stem
)

if not selected_files:
    st.warning("Please select at least one stock.")
    st.stop()


def prepare_model_data(file):
    df = load_stock_csv(file)
    _, rv_df = create_realized_variance(df)

    model_df = rv_df.copy()
    model_df["y"] = model_df["ln_rv"]

    model_df["lag_1"] = model_df["y"].shift(1)
    model_df["weekly_avg"] = model_df["y"].rolling(5).mean().shift(1)
    model_df["monthly_avg"] = model_df["y"].rolling(22).mean().shift(1)

    model_df = model_df.dropna().reset_index(drop=True)

    return model_df


def compute_metrics(actual, pred):
    error = actual - pred
    mfe = np.mean(error)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    return mfe, rmse


def evaluate_models(model_df):
    split_index = len(model_df) // 2

    train_df = model_df.iloc[:split_index].copy()
    test_df = model_df.iloc[split_index:].copy()

    y_test = test_df["y"].values

    results = {}

    # HAR rolling expanding forecast
    har_pred = []

    for i in range(len(test_df)):
        expanding_df = model_df.iloc[:split_index + i]

        X_train = expanding_df[["lag_1", "weekly_avg", "monthly_avg"]]
        X_train = sm.add_constant(X_train)

        y_train = expanding_df["y"]

        model = sm.OLS(y_train, X_train).fit()

        current_obs = test_df.iloc[i]

        X_test = pd.DataFrame({
            "const": [1],
            "lag_1": [current_obs["lag_1"]],
            "weekly_avg": [current_obs["weekly_avg"]],
            "monthly_avg": [current_obs["monthly_avg"]]
        })

        har_pred.append(model.predict(X_test).iloc[0])

    results["HAR"] = compute_metrics(y_test, np.array(har_pred))

    # AR(1) rolling expanding forecast
    ar1_pred = []

    for i in range(len(test_df)):
        expanding_df = model_df.iloc[:split_index + i]

        X_train = expanding_df[["lag_1"]]
        X_train = sm.add_constant(X_train)

        y_train = expanding_df["y"]

        model = sm.OLS(y_train, X_train).fit()

        current_obs = test_df.iloc[i]

        X_test = pd.DataFrame({
            "const": [1],
            "lag_1": [current_obs["lag_1"]]
        })

        ar1_pred.append(model.predict(X_test).iloc[0])

    results["AR(1)"] = compute_metrics(y_test, np.array(ar1_pred))

    # Random Walk
    rw_pred = test_df["lag_1"].values
    results["Random Walk"] = compute_metrics(y_test, rw_pred)

    # ML features
    x_cols = ["lag_1", "weekly_avg", "monthly_avg"]

    X_train_ml = train_df[x_cols]
    y_train_ml = train_df["y"]

    X_test_ml = test_df[x_cols]

    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train_ml, y_train_ml)
    rf_pred = rf.predict(X_test_ml)

    results["Random Forest"] = compute_metrics(y_test, rf_pred)

    # Gradient Boosting
    gb = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )

    gb.fit(X_train_ml, y_train_ml)
    gb_pred = gb.predict(X_test_ml)

    results["Gradient Boosting"] = compute_metrics(y_test, gb_pred)

    return results


all_results = []

with st.spinner("Running model comparison..."):
    for file in selected_files:
        model_df = prepare_model_data(file)

        if len(model_df) < 100:
            continue

        results = evaluate_models(model_df)

        for model_name, (mfe, rmse) in results.items():
            all_results.append({
                "Stock": file.stem,
                "Model": model_name,
                "MFE": mfe,
                "RMSE": rmse
            })


results_df = pd.DataFrame(all_results)

if results_df.empty:
    st.error("No results could be computed.")
    st.stop()


st.subheader("Full model comparison table")

st.dataframe(
    results_df,
    use_container_width=True
)


st.subheader("Best model by stock")

best_df = (
    results_df
    .sort_values("RMSE")
    .groupby("Stock")
    .first()
    .reset_index()
    .rename(columns={
        "Model": "Best model",
        "RMSE": "Best RMSE",
        "MFE": "Best MFE"
    })
)

st.dataframe(
    best_df,
    use_container_width=True
)


st.subheader("RMSE comparison by model")

selected_stock = st.selectbox(
    "Select a stock for the RMSE chart",
    sorted(results_df["Stock"].unique())
)

chart_df = results_df[
    results_df["Stock"] == selected_stock
].copy()

# Sort by RMSE
chart_df = chart_df.sort_values(
    "RMSE",
    ascending=False
)

fig = px.bar(
    chart_df,
    x="Model",
    y="RMSE",
    title=f"RMSE comparison — {selected_stock}",
    color="Model",
    text="RMSE"
)

fig.update_traces(
    texttemplate="%{text:.3f}",
    textposition="outside"
)

fig.update_layout(
    template="plotly_white",
    height=450,
    xaxis_title="Model",
    yaxis_title="RMSE",
    showlegend=False,

    # IMPORTANT
    bargap=0.25,
    bargroupgap=0.05
)

st.plotly_chart(
    fig,
    use_container_width=True
)


st.subheader("Summary table for all stocks")

summary_df = results_df.pivot(
    index="Stock",
    columns="Model",
    values="RMSE"
).reset_index()

summary_df["Best model"] = summary_df.drop(columns=["Stock"]).idxmin(axis=1)

st.dataframe(
    summary_df,
    use_container_width=True
)


st.subheader("Discussion")

overall_best = (
    best_df["Best model"]
    .value_counts()
    .idxmax()
)

st.markdown(f"""
Across the selected stocks, the most frequent best-performing model is **{overall_best}**.

The comparison shows whether machine learning models improve forecasting accuracy relative to the linear benchmarks.

The final decision is based on RMSE, because RMSE measures the overall size of the forecasting errors.
""")