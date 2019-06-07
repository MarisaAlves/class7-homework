#Ridge Regression
#Ridge regression is a technique to reduce the complexity and to avoid over-fitting
#in a linear regression model. It does so by imppsing a penalty on the size of the coefficients
#via minimizing the residual sum of squares. The greater the value of alpha, the greater
#the amount of shrinkage.


from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from math import sqrt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

os.makedirs('plots/', exist_ok=True)

diabetes = load_diabetes()
feature_names = diabetes.feature_names
X = diabetes.data
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

rr = Ridge(alpha=100)
rr.fit(X_train, y_train)

print(f"Intercept: {rr.intercept_}\n")
print(f"Coeficients: {rr.coef_}\n")
print(f"Named Coeficients: {pd.DataFrame(rr.coef_, feature_names)}")

predicted_values = rr.predict(X_test)

for (real, predicted) in list(zip(y_test, predicted_values)):
    print(f"Value: {real:.2f}, pred: {predicted:.2f}, diff: {(real - predicted):.2f}")

sns.set(palette="Paired")
residuals = y_test - predicted_values

sns.scatterplot(y_test, predicted_values, marker="X")
plt.plot([0, 300], [0, 300], '--')
plt.xlabel('Real Value')
plt.ylabel('Predicted Value')
plt.title('Ridge Real Value vs Predicted Values')
plt.savefig('plots/Ridge_Predicted.png')
plt.clf()

sns.scatterplot(y_test, residuals, marker=5)
plt.plot([300, 0], [0, 0], '--')
plt.xlabel('Real Value')
plt.ylabel('Residuals')
plt.title('Ridge Real Value vs Residuals')
plt.savefig('plots/Ridge_Residuals.png')
plt.clf()

sns.distplot(residuals, bins=20, kde=False)
plt.plot([0, 0], [50, 0], '--')
plt.title('Ridge Residual Distribution')
plt.savefig('plots/Ridge_Residual_Distn.png')
plt.clf()

print(f"MAE error(avg abs residual): {metrics.mean_absolute_error(y_test, predicted_values)}")
print(f"MSE error: {metrics.mean_squared_error(y_test, predicted_values)}")
print(f"RMSE error: {np.sqrt(metrics.mean_squared_error(y_test, predicted_values))}")


