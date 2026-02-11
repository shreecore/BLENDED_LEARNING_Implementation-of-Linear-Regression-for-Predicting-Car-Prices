# BLENDED_LEARNING
# Implementation-of-Linear-Regression-for-Predicting-Car-Prices
## AIM:
To write a program to predict car prices using a linear regression model and test the assumptions for linear regression.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import libraries and read the car price dataset.
2. Define features (X) and target price (Y).
3. Split data into train/test and scale the features.
4. Train the Linear Regression model.
5. Predict test results and evaluate using MSE, RMSE, MAE, and R².
6. Plot graphs to verify linearity, independence, homoscedasticity, and normality.
   
## Program:
```
/*
 Program to implement linear regression model for predicting car prices and test assumptions.
Developed by: MAHASHREE S 
RegisterNumber:212225230163

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Load the dataset
df=pd.read_csv('CarPrice_Assignment.csv')
df.head()

# Select features and target
X = df[['enginesize', 'horsepower', 'citympg', 'highwaympg']] # Numerical features
y = df['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=8, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Use X_train, not x_train
X_test_scaled = scaler.transform(X_test)


# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)


# Predictions
y_pred= model.predict(X_test_scaled)


# Model coefficients and metrics
# print("="*50)
print('Name: MAHASHREE S ')

print('Reg.No: 212225230163 ')

print("MODEL COEFFICIENTS:")

for feature, coef in zip(X.columns, model.coef_):

    print(f"{feature:>12} {coef:>10.2f}")

print(f"{'Intercept':>12}: {model.intercept_:>10.2f}")

print("\nMODEL PERFORMANCE:")

mse=mean_squared_error(y_test,y_pred)

mase=mean_absolute_error(y_test,y_pred)

rse=np.sqrt(mse)

print(f"{'MSE':>12}: {mse:>10}")

print(f"{'MASE':>12}: {mase:>10}")

print(f"{'RMSE':>12}:{rse:>10}")

print(f"{'R-squared' : >12}: {r2_score(y_test,y_pred): >10}")

# 1. Linearity check
plt.figure(figsize=(10,5))
plt.scatter(y_test, y_pred, alpha=0.6)

plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')

plt.title("Linearity Check: Actual vs Predicted Prices")

plt.xlabel("Actual Price ($)")

plt.ylabel("Predicted Price ($)")

plt.grid(True)

plt.show()


#2. Independence (Durbin-Watson)

residuals = y_test - y_pred

dw_test = sm.stats.durbin_watson(residuals)

print(f"\nDurbin-Watson Statistic: {dw_test:.2f}",
      "\n Values close to 2 indicates no autocorrelation")

#3. Homoscedasticity

plt.figure(figsize=(10, 5))

sns.residplot(x=y_pred, y=residuals, lowess=True, line_kws={'color': 'red'})

plt.title("Homoscedasticity Check: Residuals vs Predicted")

plt.xlabel("Predicted Price ($)")

plt.ylabel("Residuals ($)")

plt.grid(True)

plt.show()

#4. Normality of residuals

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

sns.histplot(residuals, kde=True, ax=ax1) 

ax1.set_title("Residuals Distribution")

sm.qqplot(residuals, line='45', fit=True, ax=ax2)

ax2.set_title("Q-Q Plot")

plt.tight_layout()

plt.show()
*/
```

## Output:

# Load the dataset
<img width="1532" height="469" alt="image" src="https://github.com/user-attachments/assets/a352b080-c4ba-4c52-afcc-c0cc3c562ea4" />

# Train model
<img width="1492" height="232" alt="image" src="https://github.com/user-attachments/assets/a0562b0e-2cb9-4594-9ca3-4d5332eda3c2" />

# Model coefficients and metrics
<img width="1396" height="195" alt="image" src="https://github.com/user-attachments/assets/a1a122f9-f9d0-42cd-96be-7d0d0a2d7043" />

# MODEL PERFORMANCE
<img width="1382" height="160" alt="image" src="https://github.com/user-attachments/assets/e866f86d-b29a-4adf-b93f-9c404b8215ca" />

# 1. Linearity check
<img width="1381" height="597" alt="image" src="https://github.com/user-attachments/assets/047e673c-9fba-404a-b9e0-56eca1439fe5" />

# 2. Independence (Durbin-Watson)
<img width="1345" height="92" alt="image" src="https://github.com/user-attachments/assets/8ea3aa3f-6804-4957-bbc7-e17454f4dfa5" />

# 3. Homoscedasticity
<img width="1332" height="582" alt="image" src="https://github.com/user-attachments/assets/ebd2b7f4-2dc1-4962-8e92-87582e7ac4cd" />

# 4. Normality of residuals
<img width="1356" height="517" alt="image" src="https://github.com/user-attachments/assets/1f931eb6-13f6-42c6-984f-367ecd5da9b8" />



## Result:
Thus, the program to implement a linear regression model for predicting car prices is written and verified using Python programming, along with the testing of key assumptions for linear regression.
