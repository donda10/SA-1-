# Program developed by: Simon Malachi S
# Register Number: 212224040318


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv("FuelConsumption.csv")

# Rename columns for consistency
df.rename(columns={
    'CYLINDERS': 'Cylinders',
    'ENGINESIZE': 'EngineSize',
    'FUELCONSUMPTION_COMB': 'FuelConsumption_comb',
    'CO2EMISSIONS': 'CO2Emissions'
}, inplace=True)

# Q1: Scatter plot - Cylinders vs CO2Emissions (green)
plt.figure(figsize=(6, 4))
plt.scatter(df['Cylinders'], df['CO2Emissions'], color='green')
plt.xlabel("Cylinders")
plt.ylabel("CO2 Emissions")
plt.title("Q1: Cylinders vs CO2 Emissions")
plt.grid(True)
plt.show()

# Q2: Scatter plot - Cylinders vs CO2 and EngineSize vs CO2
plt.figure(figsize=(6, 4))
plt.scatter(df['Cylinders'], df['CO2Emissions'], color='blue', label='Cylinders vs CO2')
plt.scatter(df['EngineSize'], df['CO2Emissions'], color='red', label='EngineSize vs CO2')
plt.xlabel("Feature Value")
plt.ylabel("CO2 Emissions")
plt.title("Q2: Cylinders & EngineSize vs CO2 Emissions")
plt.legend()
plt.grid(True)
plt.show()

# Q3: Add FuelConsumption_comb vs CO2 to previous plot
plt.figure(figsize=(6, 4))
plt.scatter(df['Cylinders'], df['CO2Emissions'], color='blue', label='Cylinders')
plt.scatter(df['EngineSize'], df['CO2Emissions'], color='red', label='Engine Size')
plt.scatter(df['FuelConsumption_comb'], df['CO2Emissions'], color='purple', label='Fuel Consumption')
plt.xlabel("Feature Value")
plt.ylabel("CO2 Emissions")
plt.title("Q3: Multiple Features vs CO2 Emissions")
plt.legend()
plt.grid(True)
plt.show()

# Q4: Model with Cylinders
X1 = df[['Cylinders']]
y = df['CO2Emissions']
X1_train, X1_test, y_train, y_test = train_test_split(X1, y, test_size=0.2, random_state=1)
model1 = LinearRegression()
model1.fit(X1_train, y_train)
y_pred1 = model1.predict(X1_test)
acc1 = r2_score(y_test, y_pred1)
print(f"Q4: R2 Score using Cylinders = {acc1:.4f}")

# Q5: Model with FuelConsumption_comb
X2 = df[['FuelConsumption_comb']]
X2_train, X2_test, _, _ = train_test_split(X2, y, test_size=0.2, random_state=1)
model2 = LinearRegression()
model2.fit(X2_train, y_train)
y_pred2 = model2.predict(X2_test)
acc2 = r2_score(y_test, y_pred2)
print(f"Q5: R2 Score using FuelConsumption_comb = {acc2:.4f}")

# Q6: Train-test split accuracy analysis
ratios = [0.1, 0.2, 0.3, 0.4]
print("\nQ6: R2 Scores at different train-test splits:")
for ratio in ratios:
    X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=ratio, random_state=1)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = r2_score(y_test, y_pred)
    print(f"Train-Test Split {1-ratio:.1f}-{ratio:.1f}: R2 Score = {acc:.4f}")
