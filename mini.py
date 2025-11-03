# 🌍 Climate Change Analysis - Global Dataset
# 👨‍💻 Author: Vishnu Varthan S.S

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1️⃣ Load Dataset
data = pd.read_csv("GlobalTemperatures.csv")

# 2️⃣ Display first few rows
print("📊 First 5 rows:")
print(data.head())

# 3️⃣ Convert 'dt' column to datetime format
data['dt'] = pd.to_datetime(data['dt'])

# 4️⃣ Handle missing values by filling with mean
for col in ['LandAverageTemperature', 'LandAverageTemperatureUncertainty',
            'LandMaxTemperature', 'LandMinTemperature', 
            'LandAndOceanAverageTemperature']:
    if col in data.columns:
        data[col] = data[col].fillna(data[col].mean())

# 5️⃣ Extract Year and Month for analysis
data['Year'] = data['dt'].dt.year
data['Month'] = data['dt'].dt.month

# 6️⃣ Calculate yearly average temperature
yearly_temp = data.groupby('Year')['LandAverageTemperature'].mean().reset_index()

# 7️⃣ Plot: Global Average Temperature Trend
plt.figure(figsize=(12,6))
plt.plot(yearly_temp['Year'], yearly_temp['LandAverageTemperature'], 
         color='darkred', linewidth=2.5, marker='o', markersize=4)
plt.title("🌡️ Global Average Land Temperature Trend (1750 - Present)", fontsize=14, pad=15)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Average Temperature (°C)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# 8️⃣ Compare Land and Land+Ocean temperatures (if available)
if 'LandAndOceanAverageTemperature' in data.columns:
    yearly_combo = data.groupby('Year')[['LandAverageTemperature', 'LandAndOceanAverageTemperature']].mean().reset_index()
    plt.figure(figsize=(12,6))
    plt.plot(yearly_combo['Year'], yearly_combo['LandAverageTemperature'], label='Land Temperature', color='orange')
    plt.plot(yearly_combo['Year'], yearly_combo['LandAndOceanAverageTemperature'], label='Land + Ocean Temperature', color='blue')
    plt.title("🌍 Land vs Land+Ocean Temperature Comparison", fontsize=14, pad=15)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Temperature (°C)", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# 9️⃣ Machine Learning: Predict Land Temperature using Uncertainty
X = data[['LandAverageTemperatureUncertainty']]
y = data['LandAverageTemperature']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Linear Regression Model
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

# 11️⃣ Plot Actual vs Predicted Temperatures
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, color='purple', alpha=0.6)
plt.title("Actual vs Predicted Land Temperature (°C)", fontsize=13)
plt.xlabel("Actual Temperature (°C)", fontsize=11)
plt.ylabel("Predicted Temperature (°C)", fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
