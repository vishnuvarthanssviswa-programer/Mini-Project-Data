# 🌍 Climate Change Analysis - Acre, Brazil (Enhanced Version)
# 👨‍💻 Author: Vishnu Varthan S.S

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1️⃣ Load the dataset
data = pd.read_csv("GlobalLandTemperaturesByState.csv")

# 2️⃣ Display first few rows
print("📊 First 5 rows:")
print(data.head())

# 3️⃣ Convert date column to datetime
data['dt'] = pd.to_datetime(data['dt'])

# 4️⃣ Handle missing values (replace with mean)
data['AverageTemperature'] = data['AverageTemperature'].fillna(data['AverageTemperature'].mean())
data['AverageTemperatureUncertainty'] = data['AverageTemperatureUncertainty'].fillna(data['AverageTemperatureUncertainty'].mean())

# 5️⃣ Filter data for Acre, Brazil
data = data[(data['State'] == 'Acre') & (data['Country'] == 'Brazil')]

# 6️⃣ Extract year and month
data['Year'] = data['dt'].dt.year
data['Month'] = data['dt'].dt.month

# 7️⃣ Smooth data (Yearly average to reduce congestion)
yearly_avg = data.groupby('Year')['AverageTemperature'].mean().reset_index()

# 8️⃣ Plot Temperature Trend (Clean & Clear)
plt.figure(figsize=(12,6))
plt.plot(yearly_avg['Year'], yearly_avg['AverageTemperature'], 
         color='red', linewidth=2.5, marker='o', markersize=5, label='Average Temperature (°C)')

plt.title("🌡️ Climate Change Trend - Acre, Brazil (Yearly Average)", fontsize=14, pad=15)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Temperature (°C)", fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# 9️⃣ Machine Learning: Predict temperature using uncertainty
X = data[['AverageTemperatureUncertainty']]
y = data['AverageTemperature']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# 🔟 Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📈 Model Evaluation:")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# 11️⃣ Plot Actual vs Predicted
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, color='purple', alpha=0.7)
plt.title("Actual vs Predicted Temperature (°C)", fontsize=13)
plt.xlabel("Actual Temperature", fontsize=11)
plt.ylabel("Predicted Temperature", fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
