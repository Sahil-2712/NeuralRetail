import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="👥",
    layout="wide"
)

# ---------------- TITLE ----------------
st.title("👥 Customer Segmentation Dashboard")
st.markdown("Customer segmentation analysis using Machine Learning clustering")

# ---------------- BACK BUTTON ----------------
if st.button("⬅ Back"):
    st.switch_page("main_app.py")

# ---------------- BASE DIRECTORY ----------------
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)

# ---------------- FILE PATH ----------------
segmentation_data_path = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "customer_segmentation.csv"
)

# ---------------- LOAD DATA ----------------
df = pd.read_csv(segmentation_data_path)

# ---------------- OVERVIEW ----------------
st.subheader("📊 Segmentation Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Customers", len(df))

with col2:
    if 'Cluster' in df.columns:
        st.metric(
            "Total Segments",
            df['Cluster'].nunique()
        )
    else:
        st.metric("Total Segments", "N/A")

with col3:
    if 'Annual Income' in df.columns:
        st.metric(
            "Average Income",
            f"{df['Annual Income'].mean():.2f}"
        )
    else:
        st.metric("Average Income", "N/A")

# ---------------- DATA PREVIEW ----------------
st.subheader("📋 Customer Segmentation Data")

st.dataframe(
    df.head(20),
    width='stretch'
)

# ---------------- CLUSTER DISTRIBUTION ----------------
if 'Cluster' in df.columns:

    st.subheader("📈 Customer Cluster Distribution")

    fig1, ax1 = plt.subplots(figsize=(8, 5))

    cluster_counts = (
        df['Cluster']
        .value_counts()
        .sort_index()
    )

    ax1.bar(
        cluster_counts.index.astype(str),
        cluster_counts.values
    )

    ax1.set_xlabel("Cluster")
    ax1.set_ylabel("Number of Customers")

    st.pyplot(fig1)

# ---------------- SEGMENT VISUALIZATION ----------------
if (
    'Annual Income' in df.columns and
    'Spending Score' in df.columns and
    'Cluster' in df.columns
):

    st.subheader("🎯 Customer Segments Visualization")

    fig2, ax2 = plt.subplots(figsize=(10, 6))

    scatter = ax2.scatter(
        df['Annual Income'],
        df['Spending Score'],
        c=df['Cluster']
    )

    ax2.set_xlabel("Annual Income")
    ax2.set_ylabel("Spending Score")

    st.pyplot(fig2)

# ---------------- SEGMENT INSIGHTS ----------------
if 'Cluster' in df.columns:

    st.subheader("🧠 Segment Insights")

    cluster_summary = df.groupby(
        'Cluster'
    ).mean(numeric_only=True)

    st.dataframe(
        cluster_summary,
        width='stretch'
    )

# ---------------- DOWNLOAD BUTTON ----------------
csv = df.to_csv(index=False)

st.download_button(
    label="⬇ Download Segmentation Results",
    data=csv,
    file_name="customer_segmentation_results.csv",
    mime="text/csv"
)

# ---------------- SUCCESS MESSAGE ----------------
st.success("Customer Segmentation Dashboard Loaded Successfully!")