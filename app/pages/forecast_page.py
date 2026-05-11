import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import mean_absolute_error, r2_score
import streamlit.components.v1 as components

# ---------------- PAGE TITLE ----------------
# Back Button
st.title("📈 Sales Forecast Dashboard")

if st.button("⬅ Back"):
    st.switch_page("main_app.py")

st.markdown("Forecasting future sales using Machine Learning")

# ---------------- LOAD DATA ----------------

# Forecast data
forecast_data_path = "data/processed/forecast_data.csv"

# Feature engineered data
forecast_features_path = "data/processed/forecast_features.csv"

# Trained model
model_path = "src/models/forecast_model.pkl"

# Read datasets
forecast_df = pd.read_csv(forecast_data_path)
features_df = pd.read_csv(forecast_features_path)

# Convert Date column
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

X = features_df[['lag_1', 'lag_7', 'rolling_7']]
y = features_df['Sales']

# ---------------- MAKE PREDICTIONS ----------------

predictions = model.predict(X)

# ---------------- EVALUATION METRICS ----------------

mae = mean_absolute_error(y, predictions)
r2 = r2_score(y, predictions)

# ---------------- METRICS DISPLAY ----------------

col1, col2 = st.columns(2)

with col1:
    st.metric("Mean Absolute Error", f"{mae:.2f}")

with col2:
    st.metric("R² Score", f"{r2:.2f}")

# ---------------- RESULTS DATAFRAME ----------------

results_df = pd.DataFrame({
    'Date': features_df['Date'],
    'Actual Sales': y,
    'Predicted Sales': predictions
})

st.subheader("📊 Prediction Results")

st.dataframe(results_df.head(20), use_container_width=True)

# ---------------- CLEAN ACTUAL VS PREDICTED GRAPH ----------------

st.subheader("📈 Actual vs Predicted Sales")

# Grouping data for cleaner visualization
clean_df = results_df.groupby("Date").mean(numeric_only=True).reset_index()

fig1 = go.Figure()

# Actual Sales
fig1.add_trace(
    go.Scatter(
        x=clean_df["Date"],
        y=clean_df["Actual Sales"],
        mode='lines',
        name='Actual Sales',
        line=dict(width=3)
    )
)

# Predicted Sales
fig1.add_trace(
    go.Scatter(
        x=clean_df["Date"],
        y=clean_df["Predicted Sales"],
        mode='lines',
        name='Predicted Sales',
        line=dict(width=3, dash='dash')
    )
)

fig1.update_layout(
    template="plotly_white",
    height=500,
    xaxis_title="Date",
    yaxis_title="Sales",
    hovermode="x unified"
)

st.plotly_chart(fig1, use_container_width=True)

# ---------------- MONTHLY SALES TREND ----------------

st.subheader("📅 Monthly Sales Trend")

monthly_sales = forecast_df.groupby(
    pd.Grouper(key='Date', freq='ME')
)["Sales"].sum().reset_index()

fig2 = px.line(
    monthly_sales,
    x="Date",
    y="Sales",
    markers=True
)

fig2.update_layout(
    template="plotly_white",
    height=500,
    xaxis_title="Month",
    yaxis_title="Total Sales"
)

st.plotly_chart(fig2, use_container_width=True)

# ---------------- DOWNLOAD RESULTS ----------------

csv = results_df.to_csv(index=False)

st.download_button(
    label="⬇ Download Prediction Results",
    data=csv,
    file_name='forecast_results.csv',
    mime='text/csv'
)

st.success("Forecast Dashboard Loaded Successfully!")