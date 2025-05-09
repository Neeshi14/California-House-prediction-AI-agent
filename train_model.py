from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
import numpy as np

# Load the California housing dataset
housing = fetch_california_housing()

# Select the features we want to use (matching our application's features)
features = np.column_stack([
    housing.data[:, housing.feature_names.index('AveRooms')],  # rooms
    housing.data[:, housing.feature_names.index('AveBedrms')] * 1000,  # size (approximated)
    housing.data[:, housing.feature_names.index('MedInc')] * 10000,  # income
    housing.data[:, housing.feature_names.index('Population')]  # population
])

# Scale target values to current housing prices (original data is from 1990)
target = housing.target * 1000000  # Convert to current price levels

# Train a Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(features, target)

# Save the model
with open('california_house_price_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved successfully!")