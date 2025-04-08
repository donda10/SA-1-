
# 🚗 CO2 Emission Prediction Using Linear Regression

## 📌 Aim
To analyze the relationship between vehicle specifications (like Cylinders, Engine Size, and Fuel Consumption) and their CO2 emissions using scatter plots and linear regression models in Python.

## 🧠 Algorithm
1. **Import required libraries**: `pandas`, `matplotlib`, `sklearn`
2. **Load and preprocess the dataset**
   - Rename columns for ease of use
3. **Visualize data** using scatter plots:
   - Cylinders vs CO2Emissions
   - EngineSize vs CO2Emissions
   - FuelConsumption_comb vs CO2Emissions
4. **Train Linear Regression Models**:
   - Model 1: Cylinders → CO2Emissions
   - Model 2: FuelConsumption_comb → CO2Emissions
5. **Evaluate accuracy** using R² score
6. **Test model** on different train-test splits and record accuracy

## 🧑‍💻 Program

```python
# Program developed by: Your Name
# Register Number: YourRegNo

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv("/mnt/data/FuelConsumption.csv")

# Rename columns for consistency
df.rename(columns={
    'CYLINDERS': 'Cylinders',
    'ENGINESIZE': 'EngineSize',
    'FUELCONSUMPTION_COMB': 'FuelConsumption_comb',
    'CO2EMISSIONS': 'CO2Emissions'
}, inplace=True)

# Q1: Cylinders vs CO2 Emissions
plt.figure(figsize=(6, 4))
plt.scatter(df['Cylinders'], df['CO2Emissions'], color='green')
plt.xlabel("Cylinders")
plt.ylabel("CO2 Emissions")
plt.title("Q1: Cylinders vs CO2 Emissions")
plt.grid(True)
plt.show()

# Q2: Comparison - Cylinders & EngineSize vs CO2
plt.figure(figsize=(6, 4))
plt.scatter(df['Cylinders'], df['CO2Emissions'], color='blue', label='Cylinders vs CO2')
plt.scatter(df['EngineSize'], df['CO2Emissions'], color='red', label='EngineSize vs CO2')
plt.xlabel("Feature Value")
plt.ylabel("CO2 Emissions")
plt.title("Q2: Feature Comparison")
plt.legend()
plt.grid(True)
plt.show()

# Q3: Add Fuel Consumption to the comparison
plt.figure(figsize=(6, 4))
plt.scatter(df['Cylinders'], df['CO2Emissions'], color='blue', label='Cylinders')
plt.scatter(df['EngineSize'], df['CO2Emissions'], color='red', label='Engine Size')
plt.scatter(df['FuelConsumption_comb'], df['CO2Emissions'], color='purple', label='Fuel Consumption')
plt.xlabel("Feature Value")
plt.ylabel("CO2 Emissions")
plt.title("Q3: All Features vs CO2")
plt.legend()
plt.grid(True)
plt.show()

# Q4: Model - Cylinders → CO2Emissions
X1 = df[['Cylinders']]
y = df['CO2Emissions']
X1_train, X1_test, y_train, y_test = train_test_split(X1, y, test_size=0.2, random_state=1)
model1 = LinearRegression()
model1.fit(X1_train, y_train)
y_pred1 = model1.predict(X1_test)
acc1 = r2_score(y_test, y_pred1)
print(f"Q4: R2 Score using Cylinders = {acc1:.4f}")

# Q5: Model - FuelConsumption_comb → CO2Emissions
X2 = df[['FuelConsumption_comb']]
X2_train, X2_test, _, _ = train_test_split(X2, y, test_size=0.2, random_state=1)
model2 = LinearRegression()
model2.fit(X2_train, y_train)
y_pred2 = model2.predict(X2_test)
acc2 = r2_score(y_test, y_pred2)
print(f"Q5: R2 Score using FuelConsumption_comb = {acc2:.4f}")

# Q6: Accuracy with different train-test splits
ratios = [0.1, 0.2, 0.3, 0.4]
print("\nQ6: R2 Scores at different train-test splits:")
for ratio in ratios:
    X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=ratio, random_state=1)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = r2_score(y_test, y_pred)
    print(f"Train-Test Split {1-ratio:.1f}-{ratio:.1f}: R2 Score = {acc:.4f}")
```

## 🖼️ Output

![Screenshot 2025-04-08 003402](https://github.com/user-attachments/assets/a2d14541-e76c-4792-b7a0-0f7af6aabe81)

![Screenshot 2025-04-08 003409](https://github.com/user-attachments/assets/4d931a29-f6a1-4acb-a8ac-9c55e6e6d6a5)

![Screenshot 2025-04-08 003426](https://github.com/user-attachments/assets/0394d38c-8548-4058-86c0-990836a5473d)

## ✅ Result
The CO2 emissions of vehicles were analyzed using features like number of cylinders, engine size, and fuel consumption. Using linear regression, the model showed:

- Strong correlation between `FuelConsumption_comb` and `CO2Emissions` (R² ≈ 0.79)
- The model accuracy varied slightly with train-test split ratios, peaking at ~84% for a 90-10 split.
