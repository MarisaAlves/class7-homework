#Nearest Neighbors Regression
#A regression analysis by which the label for a point is computed based on the mean
#of its k closest neighbors.
#By default, each neighbor is given an equal weight to the calculation of the point.

from sklearn.datasets import load_diabetes
from sklearn.neighbors import KNeighborsRegressor
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

neigh = KNeighborsRegressor(n_neighbors=5)
neigh.fit(X_train, y_train)
predicted_values= neigh.predict(X_test)

for (real, predicted) in list(zip(y_test, predicted_values)):
    print(f"Value: {real:.2f}, pred: {predicted:.2f}, diff: {(real - predicted):.2f}")

sns.set(palette="Set1")
residuals = y_test - predicted_values

sns.scatterplot(y_test, predicted_values, marker="p")
plt.plot([0, 300], [0, 300], '--')
plt.xlabel('Real Value')
plt.ylabel('Predicted Value')
plt.title('Real Value vs Predicted Values for 5 Neighbors')
plt.savefig('plots/Neighbors_5_Predicted.png')
plt.clf()

sns.scatterplot(y_test, residuals, marker="^")
plt.plot([300, 0], [0, 0], '--')
plt.xlabel('Real Value')
plt.ylabel('Residuals')
plt.title('Real Value vs Residuals for 5 Neighbors')
plt.savefig('plots/Neighbors_5_Residuals.png')
plt.clf()

sns.distplot(residuals, bins=20, kde=False)
plt.plot([0, 0], [300, 0], '--')
plt.title('Residual Distribution for 5 Neighbors')
plt.savefig('plots/Neighbors_5_Residual_Distn.png')
plt.clf()

print(f"MAE error(avg abs residual): {metrics.mean_absolute_error(y_test, predicted_values)}")
print(f"MSE error: {metrics.mean_squared_error(y_test, predicted_values)}")
print(f"RMSE error: {np.sqrt(metrics.mean_squared_error(y_test, predicted_values))}")

#Looking at the affect of changing the number of neighbors on the error for the model
rmse_val=[]
for K in range(20):
    K = K+1
    neigh = KNeighborsRegressor(n_neighbors=K)
    neigh.fit(X_train, y_train)
    predicted_values = neigh.predict(X_test)
    error = sqrt(mean_squared_error(y_test, predicted_values))
    rmse_val.append(error)
    print('RMSE value for k = ', K, 'is: ', error)

kvalues = pd.DataFrame(rmse_val)
kvalues.plot()
plt.xlabel('Neighbors')
plt.ylabel('RMSE')
plt.title('Change of Number of Neighbors Affecting RMSE')
plt.savefig('plots/Neighbors_vs_RMSE.png')
plt.clf()

neigh = KNeighborsRegressor(n_neighbors=13)
neigh.fit(X_train, y_train)
predicted_values= neigh.predict(X_test)

for (real, predicted) in list(zip(y_test, predicted_values)):
    print(f"Value: {real:.2f}, pred: {predicted:.2f}, diff: {(real - predicted):.2f}")

sns.set(palette="cubehelix")
residuals = y_test - predicted_values

sns.scatterplot(y_test, predicted_values, marker="+")
plt.plot([0, 300], [0, 300], '--')
plt.xlabel('Real Value')
plt.ylabel('Predicted Value')
plt.title('Real Value vs Predicted Values for 13 Neighbors')
plt.savefig('plots/Neighbors_13_Predicted.png')
plt.clf()

sns.scatterplot(y_test, residuals, marker=9)
plt.plot([300, 0], [0, 0], '--')
plt.xlabel('Real Value')
plt.ylabel('Residuals')
plt.title('Real Value vs Residuals for 13 Neighbors')
plt.savefig('plots/Neighbors_13_Residuals.png')
plt.clf()

sns.distplot(residuals, bins=20, kde=False)
plt.plot([0, 0], [50, 0], '--')
plt.title('Residual Distribution for 13 Neighbors')
plt.savefig('plots/Neighbors_13_Residual_Distn.png')
plt.clf()

print(f"MAE error(avg abs residual): {metrics.mean_absolute_error(y_test, predicted_values)}")
print(f"MSE error: {metrics.mean_squared_error(y_test, predicted_values)}")
print(f"RMSE error: {np.sqrt(metrics.mean_squared_error(y_test, predicted_values))}")