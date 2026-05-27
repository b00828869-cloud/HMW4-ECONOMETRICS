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

st.subheader("Raw data preview")

st.dataframe(
    df.head(20),
    use_container_width=True,
    height=400
)
st.divider()

st.subheader("Structure consistency across all stocks")

sample_results = []

for file in csv_files[:10]:
    try:
        temp_df = load_stock_csv(file)

        sample_results.append({
            "Stock": file.stem,
            "Rows": temp_df.shape[0],
            "Columns": temp_df.shape[1],
            "Column names": str(list(temp_df.columns))
        })

    except Exception as e:
        sample_results.append({
            "Stock": file.stem,
            "Rows": "ERROR",
            "Columns": "ERROR",
            "Column names": str(e)
        })

st.dataframe(sample_results, use_container_width=True)