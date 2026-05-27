import numpy as np
from streamlit import cache_data


OBS_PER_DAY = 78
MAX_OBSERVATIONS = 300000


@cache_data
def create_five_min_returns(df):
    df = df.tail(MAX_OBSERVATIONS).copy()

    df["price"] = df["close"]
    df["log_price"] = np.log(df["price"])
    df["return_5min"] = df["log_price"].diff()

    df = df.dropna(subset=["return_5min"]).reset_index(drop=True)

    return df


@cache_data
def create_realized_variance(df):
    returns_df = create_five_min_returns(df)

    returns_df["day_id"] = returns_df.index // OBS_PER_DAY

    rv_df = (
        returns_df
        .groupby("day_id")["return_5min"]
        .apply(lambda x: np.sum(x ** 2))
        .reset_index(name="rv")
    )

    rv_df["ln_rv"] = np.log(rv_df["rv"])

    return returns_df, rv_df