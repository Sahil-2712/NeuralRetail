import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
# Load the features dataset 
df = pd.read_csv(
    "data/processed/forecast_features.csv"
)

#print(df.head())
# Define features and target variable
#input features
X = df[['lag_1', 'lag_7', 'rolling_7']]
#target variable
y = df['Sales']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)

#create the model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
# Train the model
model.fit(X_train, y_train)
joblib.dump(
    model,
    "src/models/forecast_model.pkl"
)

print("Model saved successfully!")
# Make predictions
predictions = model.predict(X_test)
# Evaluate the model
mae = mean_absolute_error(y_test, predictions)

print("Mean Absolute Error:", mae)#MAE = average prediction error.
r2 = r2_score(y_test, predictions)

print("R2 Score:", r2)

#view the predictions vs actual values
results = pd.DataFrame({
    'Actual': y_test,
    'Predicted': predictions
})
print(results.head(10))
# Save forecast results for dashboard
results.to_csv(
    "data/processed/forecast_results.csv",
    index=False
)

print("Forecast results exported successfully!")