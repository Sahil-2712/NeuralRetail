import pandas as pd
#LOAD FORECAST DATASET
df = pd.read_csv("data/processed/forecast_data.csv")

#print(df.head())
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
# Create lag features
df['lag_1'] = df['Sales'].shift(1)

df['lag_7'] = df['Sales'].shift(7)

# Create rolling mean features
df['rolling_7'] = df['Sales'].rolling(window=7).mean()
df = df.dropna()
print(df.head(10))
# Save the features to a new CSV file
df.to_csv(
    "data/processed/forecast_features.csv",
    index=False
)

print("Feature engineering completed!")