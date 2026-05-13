import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import mean_absolute_error, r2_score
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Sales Forecast Dashboard",
    page_icon="📈",
    layout="wide"
)

# ---------------- TITLE ----------------
st.title("📈 Sales Forecast Dashboard")

if st.button("⬅ Back"):
    st.switch_page("main_app.py")

st.markdown(
    "Machine Learning based sales forecasting dashboard"
)

# ---------------- BASE DIRECTORY ----------------
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)

# ---------------- FILE PATHS ----------------

forecast_data_path = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "forecast_data.csv"
)

forecast_features_path = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "forecast_features.csv"
)

forecast_results_path = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "forecast_results.csv"
)

model_path = os.path.join(
    BASE_DIR,
    "src",
    "models",
    "forecast_model.pkl"
)

# ---------------- LOAD DATA ----------------

forecast_df = pd.read_csv(forecast_data_path)

features_df = pd.read_csv(forecast_features_path)

results_df = pd.read_csv(forecast_results_path)

# ---------------- DATE CONVERSION ----------------

forecast_df['Date'] = pd.to_datetime(
    forecast_df['Date'],
    format='mixed',
    dayfirst=True
)

features_df['Date'] = pd.to_datetime(
    features_df['Date'],
    format='mixed',
    dayfirst=True
)

# ---------------- LOAD MODEL ----------------

model = joblib.load(model_path)

# ---------------- PREPARE FEATURES ----------------

X = features_df[
    ['lag_1', 'lag_7', 'rolling_7']
]

y = features_df['Sales']

# ---------------- MAKE PREDICTIONS ----------------

predictions = model.predict(X)

# ---------------- EVALUATION METRICS ----------------

mae = mean_absolute_error(y, predictions)

r2 = r2_score(y, predictions)

# ---------------- METRICS DISPLAY ----------------

st.subheader("📊 Forecast Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Mean Absolute Error",
        f"{mae:.2f}"
    )

with col2:
    st.metric(
        "R² Score",
        f"{r2:.2f}"
    )

with col3:
    st.metric(
        "Average Forecasted Sales",
        f"{predictions.mean():.2f}"
    )

with col4:
    st.metric(
        "Maximum Forecasted Sales",
        f"{predictions.max():.2f}"
    )

# ---------------- RESULTS TABLE ----------------

st.subheader("📋 Forecast Prediction Results")

st.dataframe(
    results_df.head(20),
    width='stretch'
)

# ---------------- ACTUAL VS PREDICTED ----------------

st.subheader("📈 Actual vs Predicted Sales")

results_df['Index'] = range(len(results_df))

fig1 = go.Figure()

# Actual Sales
fig1.add_trace(
    go.Scatter(
        x=results_df["Index"],
        y=results_df["Actual"],
        mode='lines',
        name='Actual Sales',
        line=dict(width=3)
    )
)

# Predicted Sales
fig1.add_trace(
    go.Scatter(
        x=results_df["Index"],
        y=results_df["Predicted"],
        mode='lines',
        name='Predicted Sales',
        line=dict(width=3, dash='dash')
    )
)

fig1.update_layout(
    template="plotly_white",
    height=500,
    xaxis_title="Records",
    yaxis_title="Sales",
    hovermode="x unified"
)

st.plotly_chart(
    fig1,
    width='stretch'
)

# ---------------- FORECAST TREND ----------------

st.subheader("📅 Monthly Sales Trend")

monthly_sales = forecast_df.groupby(
    pd.Grouper(key='Date', freq='ME')
)["Sales"].sum().reset_index()

fig2 = px.line(
    monthly_sales,
    x="Date",
    y="Sales",
    markers=True,
    title="Monthly Sales Forecast Trend"
)

fig2.update_layout(
    template="plotly_white",
    height=500,
    xaxis_title="Month",
    yaxis_title="Total Sales"
)

st.plotly_chart(
    fig2,
    width='stretch'
)

# ---------------- ERROR DISTRIBUTION ----------------

st.subheader("📉 Forecast Error Distribution")

results_df["Error"] = (
    results_df["Actual"] -
    results_df["Predicted"]
)

fig3 = px.histogram(
    results_df,
    x="Error",
    nbins=30,
    title="Prediction Error Distribution"
)

fig3.update_layout(
    template="plotly_white",
    height=500
)

st.plotly_chart(
    fig3,
    width='stretch'
)

# ---------------- FORECAST PERFORMANCE ----------------

st.subheader("🎯 Forecast Performance")

performance_counts = {
    "Accurate": (
        abs(results_df["Error"]) < 50
    ).sum(),

    "Moderate Error": (
        (abs(results_df["Error"]) >= 50) &
        (abs(results_df["Error"]) < 200)
    ).sum(),

    "High Error": (
        abs(results_df["Error"]) >= 200
    ).sum()
}

fig4 = px.bar(
    x=list(performance_counts.keys()),
    y=list(performance_counts.values()),
    title="Forecast Accuracy Categories"
)

fig4.update_layout(
    template="plotly_white",
    xaxis_title="Performance Category",
    yaxis_title="Records"
)

st.plotly_chart(
    fig4,
    width='stretch'
)

# ---------------- ML MODEL OUTPUTS ----------------

st.subheader("🤖 ML Model Outputs")

ml_col1, ml_col2, ml_col3 = st.columns(3)

with ml_col1:
    st.info(
        f"Total Forecast Records: "
        f"{len(results_df)}"
    )

with ml_col2:
    st.info(
        f"Average Forecasted Sales: "
        f"{predictions.mean():.2f}"
    )

with ml_col3:
    st.info(
        f"Best R² Score: "
        f"{r2:.2f}"
    )

# ---------------- DOWNLOAD BUTTON ----------------

csv = results_df.to_csv(index=False)

st.download_button(
    label="⬇ Download Forecast Results",
    data=csv,
    file_name='forecast_results.csv',
    mime='text/csv'
)

# ---------------- SUCCESS MESSAGE ----------------

st.success(
    "✅ Forecast Dashboard Loaded Successfully!"
)