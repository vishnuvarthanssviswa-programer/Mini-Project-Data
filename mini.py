# Climate Change Temperature Trend Analysis
# SDG Goal 13: Climate Action

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# 1️⃣ Load the dataset
# You can download data from: https://www.ncei.noaa.gov/access/monitoring/global-temperature-anomalies
# Example CSV: 'https://datahub.io/core/global-temp/r/annual.csv'

url = 'https://datahub.io/core/global-temp/r/annual.csv'
df = pd.read_csv(url)

# 2️⃣ View first few rows
print("Sample Data:\n", df.head())

# 3️⃣ Extract relevant columns
df = df.rename(columns={'Source': 'source', 'Year': 'year', 'Mean': 'mean_temp_anomaly'})

# Filter only 'GCAG' (Global Combined Land and Ocean)
data = df[df['source'] == 'GCAG']

# 4️⃣ Trend analysis
X = data['year'].values.reshape(-1, 1)
y = data['mean_temp_anomaly'].values

model = LinearRegression()
model.fit(X, y)
trend = model.predict(X)

# 5️⃣ Display warming rate per decade
slope = model.coef_[0]
warming_rate = slope * 10
print(f"\n🌡️ Estimated Warming Rate: {warming_rate:.3f} °C per decade")

# 6️⃣ Plot temperature trend
plt.figure(figsize=(10,6))
plt.scatter(data['year'], y, color='skyblue', label='Observed Anomalies')
plt.plot(data['year'], trend, color='red', linewidth=2, label='Trend Line')

plt.title('Global Temperature Trend Analysis (NOAA / GCAG)')
plt.xlabel('Year')
plt.ylabel('Temperature Anomaly (°C)')
plt.legend()
plt.grid(True)
plt.show()
