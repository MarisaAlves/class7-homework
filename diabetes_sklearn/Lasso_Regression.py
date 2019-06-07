#Lasso Regression
#Another technique to reduce model complexity and prevent over-fitting
#It estimates sparse coefficients, that is, it reduces the number of features that
#will be used in the regression model, by preferring a solution with fewer coefficients.
#The lasso estimate is added to the objective function of a linear model and the objective
#is to minimize said function via least-squares.

from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn import metrics
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

lasso = Lasso()
lasso.fit(X_train, y_train)

print(f"Intercept: {lasso.intercept_}\n")
print(f"Coeficients: {lasso.coef_}\n")
print(f"Named Coeficients: {pd.DataFrame(lasso.coef_, feature_names)}")

predicted_values = lasso.predict(X_test)

for (real, predicted) in list(zip(y_test, predicted_values)):
    print(f"Value: {real:.2f}, pred: {predicted:.2f}, diff: {(real - predicted):.2f}")

sns.set(palette="hls")
residuals = y_test - predicted_values

sns.scatterplot(y_test, predicted_values, marker="+")
plt.plot([0, 300], [0, 300], '--')
plt.xlabel('Real Value')
plt.ylabel('Predicted Value')
plt.title('Lasso Real Value vs Predicted Values')
plt.savefig('plots/Lasso_Predicted.png')
plt.clf()

sns.scatterplot(y_test, residuals, marker="s")
plt.plot([200, 0], [0, 0], '--')
plt.xlabel('Real Value')
plt.ylabel('Residuals')
plt.title('Lasso Real Value vs Residuals')
plt.savefig('plots/Lasso_Residuals.png')
plt.clf()

sns.distplot(residuals, bins=20, kde=False)
plt.plot([0, 0], [50, 0], '--')
plt.title('Ridge Residual Distribution')
plt.savefig('plots/Ridge_Residual_Distn.png')
plt.clf()

print(f"MAE error(avg abs residual): {metrics.mean_absolute_error(y_test, predicted_values)}")
print(f"MSE error: {metrics.mean_squared_error(y_test, predicted_values)}")
print(f"RMSE error: {np.sqrt(metrics.mean_squared_error(y_test, predicted_values))}")