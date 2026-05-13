
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
url = "https://github.com/IamArunavaSamanta/NeuralRetailProject/raw/main/data/processed/cleaned_online_retail.csv.gz"
df = pd.read_csv("../data/processed/cleaned_online_retail.csv.gz", compression="gzip")
df.head(2)  
# This handles the missing values without crashing
df['CustomerID'] = df['CustomerID'].astype(float).astype('Int64')
df['TotalAmount'] = df['Quantity'] * df['Price']
df['Description'] = df['Description'].fillna('Unknown')
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

# Aggregate by Customer
customer_df = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days, # Recency
    'InvoiceNo': 'nunique',                                   # Frequency
    'TotalAmount': 'sum'                                   # Monetary
}).reset_index()

customer_df.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
# 1 if Recency > 90 days, else 0
customer_df['Churn'] = (customer_df['Recency'] > 90).astype(int)
print(customer_df['Churn'].value_counts())
X = customer_df[['Frequency', 'Monetary']] # Don't use Recency, it's too tied to the target!
y = customer_df['Churn']
X = customer_df[['Frequency', 'Monetary']] # Don't use Recency, it's too tied to the target!
y = customer_df['Churn']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Churn Prediction Confusion Matrix')
plt.show()
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Initialize the ANN
model = Sequential()

# Input layer and first hidden layer
model.add(
    Dense(
        units=100,
        activation='relu',
        input_dim=X_train.shape[1]
    )
)

model.add(
    Dropout(0.2)
)  # Prevents overfitting

# Second hidden layer
model.add(
    Dense(
        units=50,
        activation='relu'
    )
)

model.add(
    Dropout(0.2)
)  # Prevents overfitting

# Output layer
model.add(
    Dense(
        units=1,
        activation='sigmoid'
    )
)

# Compile the ANN
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
# Early stopping to prevent overfitting
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Train the ANN model
history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[early_stop]
)
# Predict churn probabilities
y_pred_prob = model.predict(X_test)

# Convert probabilities into binary predictions
y_pred = (y_pred_prob > 0.5).astype(int)
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("Accuracy:", accuracy)
model.save(
    "../src/models/customer_churn_model.keras"
)

print("Churn model saved successfully!")
import pandas as pd

# Convert X_test to DataFrame
results_df = pd.DataFrame(X_test)

# Add prediction columns
results_df['Actual_Churn'] = y_test
results_df['Churn_Probability'] = y_pred_prob.flatten()
results_df['Predicted_Churn'] = y_pred.flatten()

# Export CSV
results_df.to_csv(
    "../data/processed/churn_results.csv",
    index=False
)

print("Churn results exported successfully!")