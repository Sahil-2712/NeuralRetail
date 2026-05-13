import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="⚠",
    layout="wide"
)

# ---------------- TITLE ----------------
st.title("⚠ Customer Churn Prediction Dashboard")
st.markdown(
    "ANN based churn prediction and retention intelligence dashboard"
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
churn_data_path = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "churn_results.csv"
)

# ---------------- LOAD DATA ----------------
df = pd.read_csv(churn_data_path)

# ---------------- BUSINESS OVERVIEW ----------------
st.subheader("📊 Churn Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Customers",
        len(df)
    )

with col2:
    st.metric(
        "Predicted Churn Customers",
        int(df['Predicted_Churn'].sum())
    )

with col3:
    st.metric(
        "Average Churn Probability",
        f"{df['Churn_Probability'].mean():.2f}"
    )

with col4:
    churn_rate = (
        df['Predicted_Churn'].mean() * 100
    )

    st.metric(
        "Predicted Churn Rate",
        f"{churn_rate:.2f}%"
    )

# ---------------- CHURN DISTRIBUTION ----------------
st.subheader("📈 Churn Distribution")

fig1, ax1 = plt.subplots(figsize=(6, 6))

churn_counts = (
    df['Predicted_Churn']
    .value_counts()
)

ax1.pie(
    churn_counts.values,
    labels=["No Churn", "Churn"],
    autopct='%1.1f%%'
)

ax1.set_title("Predicted Customer Churn")

st.pyplot(fig1)

# ---------------- CHURN PROBABILITY ----------------
st.subheader("📊 Churn Probability Distribution")

fig2, ax2 = plt.subplots(figsize=(10, 5))

ax2.hist(
    df['Churn_Probability'],
    bins=20
)

ax2.set_xlabel("Churn Probability")
ax2.set_ylabel("Customers")
ax2.set_title("Customer Churn Probability")

st.pyplot(fig2)

# ---------------- HIGH RISK CUSTOMERS ----------------
st.subheader("⚠ High Risk Customers")

high_risk = df.sort_values(
    by='Churn_Probability',
    ascending=False
).head(10)

st.dataframe(
    high_risk,
    width='stretch'
)



# ---------------- MODEL OUTPUTS ----------------
st.subheader("🤖 ANN Model Outputs")# ---------------- CHURN RISK VISUALIZATION ----------------
st.subheader("🎯 Customer Churn Risk Levels")

risk_counts = {
    "Low Risk": (df['Churn_Probability'] < 0.4).sum(),

    "Medium Risk": (
        (df['Churn_Probability'] >= 0.4) &
        (df['Churn_Probability'] < 0.7)
    ).sum(),

    "High Risk": (
        df['Churn_Probability'] >= 0.7
    ).sum()
}

fig3, ax3 = plt.subplots(figsize=(8, 5))

ax3.bar(
    risk_counts.keys(),
    risk_counts.values()
)

ax3.set_xlabel("Risk Level")
ax3.set_ylabel("Customers")
ax3.set_title("Customer Churn Risk Categories")

st.pyplot(fig3)

ml_col1, ml_col2, ml_col3 = st.columns(3)

with ml_col1:
    st.info(
        f"Total Predictions: {len(df)}"
    )

with ml_col2:
    st.info(
        f"Average Churn Risk: "
        f"{df['Churn_Probability'].mean():.2f}"
    )

with ml_col3:
    st.info(
        f"High Risk Customers: "
        f"{(df['Churn_Probability'] > 0.7).sum()}"
    )

# ---------------- DOWNLOAD BUTTON ----------------
csv = df.to_csv(index=False)

st.download_button(
    label="⬇ Download Churn Results",
    data=csv,
    file_name="customer_churn_results.csv",
    mime="text/csv"
)

# ---------------- SUCCESS MESSAGE ----------------
st.success(
    "✅ Customer Churn Dashboard Loaded Successfully!"
)