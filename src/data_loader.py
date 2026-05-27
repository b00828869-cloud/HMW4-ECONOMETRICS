from pathlib import Path
import pandas as pd

DATA_FOLDER = Path("data/HF_csvformat")


def list_csv_files():
    return sorted([
        f for f in DATA_FOLDER.rglob("*.csv")
        if not f.name.startswith("._")
    ])


def load_stock_csv(file_path):
    df = pd.read_csv(file_path, header=None)

    df = df.dropna(how="all")
    df = df.reset_index(drop=True)

    df.columns = ["open", "high", "low", "close", "volume"]

    return df