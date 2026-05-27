import streamlit as st
from src.data_loader import list_csv_files, load_stock_csv

st.set_page_config(
    page_title="Q1 - Data connection",
    page_icon="🔌",
    layout="wide"
)

st.title("🔌 Question 1 — Data connection")

st.markdown("""
This first page checks that the app is correctly connected to the high-frequency CSV files.

The objective is to verify:
- how many stock files are available,
- whether each file can be loaded correctly,
- what columns are included,
- and whether the data structure is ready for the next steps.
""")

csv_files = list_csv_files()

col1, col2 = st.columns(2)

with col1:
    st.metric("CSV files detected", len(csv_files))

with col2:
    st.metric("Data source", "Local folder")

if not csv_files:
    st.error("No CSV files found. Check that your files are inside data/HF_csvformat.")
    st.stop()

st.success("Data connection successful.")

selected_file = st.selectbox(
    "Select a stock file",
    csv_files,
    format_func=lambda x: x.stem
)

df = load_stock_csv(selected_file)

st.divider()

st.subheader(f"Preview of {selected_file.stem}")

st.dataframe(
    df.head(20),
    use_container_width=True
)

st.subheader("Dataset structure")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Rows", df.shape[0])

with col2:
    st.metric("Columns", df.shape[1])

with col3:
    st.metric("Selected stock", selected_file.stem)

st.subheader("Columns detected")
st.write(list(df.columns))

st.subheader("Data types")
st.dataframe(
    df.dtypes.astype(str).reset_index().rename(
        columns={"index": "Column", 0: "Data type"}
    ),
    use_container_width=True
)

st.subheader("Descriptive overview")

numeric_summary = df.describe().T

st.dataframe(
    numeric_summary,
    use_container_width=True
)

st.info("""
This confirms that the CSV files are correctly loaded.  
The next step will be to identify the timestamp and price columns, then create 5-minute returns.
""")