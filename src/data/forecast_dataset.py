import pandas as pd

# Read cleaned compressed dataset
df = pd.read_csv(
    "data/processed/cleaned_online_retail.csv.gz",
    compression='gzip'
)

print(df.head())

# Create a new DataFrame with only the 'InvoiceDate' and 'Revenue' columns
forecast_df = df[['InvoiceDate', 'Revenue']].copy()
forecast_df['InvoiceDate'] = pd.to_datetime(
    forecast_df['InvoiceDate']
)

daily_sales = (
    forecast_df
    .groupby(forecast_df['InvoiceDate'].dt.date)['Revenue']
    .sum()
    .reset_index()
)

daily_sales.columns = ['Date', 'Sales']
daily_sales.to_csv(
    "data/processed/forecast_data.csv",
    index=False
)

print("Forecast dataset created successfully!")