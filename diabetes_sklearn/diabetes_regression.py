from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt

diabetes = load_diabetes()
feature_names = diabetes.feature_names
X = diabetes.data
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)

print(f"Intercept: {lm.intercept_}\n")
print(f"Coeficients: {lm.coef_}\n")
print(f"Named Coeficients: {pd.DataFrame(lm.coef_, feature_names)}")

predicted_values = lm.predict(X_test)

for (real, predicted) in list(zip(y_test, predicted_values)):
    print(f"Value: {real:.2f}, pred: {predicted:.2f}, diff: {(real - predicted):.2f}")


os.makedirs('plots/', exist_ok=True)

sns.set(palette="Set2")
residuals = y_test - predicted_values

sns.scatterplot(y_test, predicted_values, marker="H")
plt.plot([0, 300], [0, 300], '--')
plt.xlabel('Real Value')
plt.ylabel('Predicted Value')
plt.title('Real Value vs Predicted Values')
plt.savefig('plots/Predicted.png')
plt.clf()

sns.scatterplot(y_test, residuals, marker="*")
plt.plot([300, 0], [0, 0], '--')
plt.xlabel('Real Value')
plt.ylabel('Residuals')
plt.title('Real Value vs Residuals')
plt.savefig('plots/Residuals.png')
plt.clf()

sns.distplot(residuals, bins=20, kde=False)
plt.plot([0, 0], [50, 0], '--')
plt.title('Residual Distribution')
plt.savefig('plots/Residual_Distn.png')
plt.clf()

print(f"MAE error(avg abs residual): {metrics.mean_absolute_error(y_test, predicted_values)}")
print(f"MSE error: {metrics.mean_squared_error(y_test, predicted_values)}")
print(f"RMSE error: {np.sqrt(metrics.mean_squared_error(y_test, predicted_values))}")
