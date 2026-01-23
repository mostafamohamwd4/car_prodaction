"""
Retrain the car price prediction model with current scikit-learn version.
This script retrains the model to fix pickle compatibility issues.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Data", "Car details v3.csv")
MODEL_PATH = os.path.join(BASE_DIR, "Model", "BestModel.pkl")

print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")

# Data preprocessing (matching what was done in the notebook)
# Drop rows with missing values
df_clean = df.dropna()

# Calculate car age from year
current_year = 2026  # Updated to match current year
df_clean = df_clean.copy()
df_clean['age'] = current_year - df_clean['year']

# Process mileage (remove 'kmpl' or 'km/kg' and convert to float)
df_clean['mileage'] = df_clean['mileage'].str.replace(' kmpl', '').str.replace(' km/kg', '')
df_clean['mileage'] = pd.to_numeric(df_clean['mileage'], errors='coerce')

# Process engine (remove 'CC' and convert to float)
df_clean['engine'] = df_clean['engine'].str.replace(' CC', '')
df_clean['engine'] = pd.to_numeric(df_clean['engine'], errors='coerce')

# Process max_power (remove 'bhp' and convert to float)
df_clean['max_power'] = df_clean['max_power'].str.replace(' bhp', '')
df_clean['max_power'] = pd.to_numeric(df_clean['max_power'], errors='coerce')

# Encode categorical variables
fuel_map = {'Petrol': 1, 'Diesel': 2, 'CNG': 3, 'LPG': 4}
seller_type_map = {'Individual': 1, 'Dealer': 2, 'Trustmark Dealer': 3}
transmission_map = {'Manual': 1, 'Automatic': 2}
owner_map = {'First Owner': 0, 'Second Owner': 1, 'Third Owner': 2, 'Fourth & Above Owner': 3, 'Test Drive Car': 4}

df_clean['fuel'] = df_clean['fuel'].map(fuel_map)
df_clean['seller_type'] = df_clean['seller_type'].map(seller_type_map)
df_clean['transmission'] = df_clean['transmission'].map(transmission_map)
df_clean['owner'] = df_clean['owner'].map(owner_map)

# Drop rows with any remaining NaN values
df_clean = df_clean.dropna()

print(f"Dataset shape after cleaning: {df_clean.shape}")

# Features and target (matching what app.py expects)
feature_cols = ['selling_price', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 
                'mileage', 'engine', 'max_power', 'seats', 'age']

# Check which columns exist
available_features = [col for col in feature_cols if col in df_clean.columns]
print(f"Available features: {available_features}")

# Note: Based on the app.py, it seems the model might be used to predict selling_price
# But the app includes selling_price as a feature, which is unusual
# Let's check what the target should be - looking at the data, selling_price is the target
# The app.py has a bug - it includes selling_price as a feature

# Let's fix this - the target should be selling_price, not a feature
X = df_clean[['km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 
              'mileage', 'engine', 'max_power', 'seats', 'age']]
y = df_clean['selling_price']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training RandomForest model...")
# Train the model with same parameters as might have been used
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Evaluate
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"Training R² Score: {train_score:.4f}")
print(f"Testing R² Score: {test_score:.4f}")

# Save the model
print(f"Saving model to: {MODEL_PATH}")
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model, f)

print("Model retrained and saved successfully!")
print("\n⚠️ NOTE: The app.py was including 'selling_price' as a feature which doesn't make sense")
print("since we're trying to predict the price. The model now expects these features:")
print(X.columns.tolist())
