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
st.markdown(
    "Customer segmentation, churn prediction and CLV intelligence dashboard"
)

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
    "dashboard_master.csv"
)

# ---------------- LOAD DATA ----------------
df = pd.read_csv(segmentation_data_path)
# ---------------- LOAD DATA ----------------
df = pd.read_csv(segmentation_data_path)

# ---------------- BUSINESS OVERVIEW ----------------
st.subheader("📊 Business Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Customers",
        df['CustomerID'].nunique()
    )

with col2:
    st.metric(
        "KMeans Segments",
        df['KMeans_Cluster'].nunique()
    )

with col3:
    st.metric(
        "Average Revenue",
        f"{df['Monetary'].mean():.2f}"
    )

with col4:
    st.metric(
        "Average Frequency",
        f"{df['Frequency'].mean():.2f}"
    )

# ---------------- KMEANS CLUSTER DISTRIBUTION ----------------
st.subheader("📈 KMeans Cluster Distribution")

fig1, ax1 = plt.subplots(figsize=(10, 5))

cluster_counts = (
    df['KMeans_Cluster']
    .value_counts()
    .sort_index()
)

ax1.bar(
    cluster_counts.index.astype(str),
    cluster_counts.values
)

ax1.set_xlabel("KMeans Cluster")
ax1.set_ylabel("Customers")
ax1.set_title("KMeans Customer Segments")

st.pyplot(fig1)

# ---------------- RFM CUSTOMER SEGMENTATION ----------------
st.subheader("🎯 RFM Customer Segmentation")

fig2, ax2 = plt.subplots(figsize=(12, 6))

scatter = ax2.scatter(
    df['Frequency'],
    df['Monetary'],
    c=df['KMeans_Cluster']
)

ax2.set_xlabel("Frequency")
ax2.set_ylabel("Monetary")
ax2.set_title(
    "Customer Segments based on Frequency and Monetary"
)

st.pyplot(fig2)

# ---------------- DBSCAN OUTLIER DETECTION ----------------
if 'DBSCAN_Cluster' in df.columns:

    st.subheader("🔍 DBSCAN Outlier Detection")

    fig3, ax3 = plt.subplots(figsize=(12, 6))

    scatter = ax3.scatter(
        df['Recency'],
        df['Monetary'],
        c=df['DBSCAN_Cluster']
    )

    ax3.set_xlabel("Recency")
    ax3.set_ylabel("Monetary")
    ax3.set_title("DBSCAN Outlier Clusters")

    st.pyplot(fig3)
# ---------------- GMM CONFIDENCE ANALYSIS ----------------
if 'Max_Probability' in df.columns:

    st.subheader("🤖 GMM Probability Analysis")

    fig4, ax4 = plt.subplots(figsize=(10, 5))

    ax4.hist(
        df['Max_Probability'],
        bins=20
    )

    ax4.set_xlabel("Max Probability")
    ax4.set_ylabel("Customers")
    ax4.set_title("GMM Confidence Distribution")

    st.pyplot(fig4)

# ---------------- CHURN ANALYSIS ----------------
if 'Churn' in df.columns:

    st.subheader("⚠ Customer Churn Analysis")

    churn_counts = df['Churn'].value_counts()

    fig5, ax5 = plt.subplots(figsize=(6, 6))

    ax5.pie(
        churn_counts.values,
        labels=churn_counts.index,
        autopct='%1.1f%%'
    )

    ax5.set_title("Customer Churn Distribution")

    st.pyplot(fig5)
# ---------------- CLV ANALYSIS ----------------
st.subheader("💰 Customer Lifetime Value Analysis")

top_clv = df.sort_values(
    by='CLV',
    ascending=False
).head(10)

fig6, ax6 = plt.subplots(figsize=(12, 5))

ax6.bar(
    top_clv['CustomerID'].astype(str),
    top_clv['CLV']
)

ax6.set_xlabel("Customer ID")
ax6.set_ylabel("CLV")
ax6.set_title("Top 10 Customers by CLV")

plt.xticks(rotation=45)

st.pyplot(fig6)

# ---------------- RETENTION STRATEGY ----------------
if 'Retention_Action' in df.columns:

    st.subheader("🧠 Retention Strategy Analysis")

    retention_counts = (
        df['Retention_Action']
        .value_counts()
    )

    fig7, ax7 = plt.subplots(figsize=(10, 5))

    ax7.bar(
        retention_counts.index,
        retention_counts.values
    )

    ax7.set_xlabel("Retention Action")
    ax7.set_ylabel("Customers")
    ax7.set_title("Retention Strategy Distribution")

    plt.xticks(rotation=15)

    st.pyplot(fig7)

# ---------------- SEGMENT INSIGHTS ----------------
st.subheader("📋 Segment Insights")

cluster_summary = df.groupby(
    'KMeans_Cluster'
).mean(numeric_only=True)

st.dataframe(
    cluster_summary,
    width='stretch'
)

# ---------------- TOP CUSTOMERS ----------------
st.subheader("🏆 High Value Customers")

top_customers = df.sort_values(
    by='Monetary',
    ascending=False
).head(10)

st.dataframe(
    top_customers[
        [
            'CustomerID',
            'Monetary',
            'Frequency',
            'CLV',
            'Retention_Action'
        ]
    ],
    width='stretch'
)

# ---------------- ML MODEL OUTPUTS ----------------
st.subheader("🤖 ML Model Outputs")

ml_col1, ml_col2, ml_col3 = st.columns(3)

with ml_col1:
    if 'KMeans_Cluster' in df.columns:
        st.info(
            f"KMeans Clusters Created: "
            f"{df['KMeans_Cluster'].nunique()}"
        )

with ml_col2:
    if 'DBSCAN_Cluster' in df.columns:
        st.info(
            f"DBSCAN Clusters Detected: "
            f"{df['DBSCAN_Cluster'].nunique()}"
        )

with ml_col3:
    if 'Churn_Probability' in df.columns:
        st.info(
            f"Average Churn Probability: "
            f"{df['Churn_Probability'].mean():.2f}"
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
st.success(
    "✅ Customer Segmentation Dashboard Loaded Successfully!"
)