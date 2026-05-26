# Implementation of Multivariate Linear Regression
## Aim
To write a python program to implement multivariate linear regression and predict the output.
## Equipment’s required:
1.	Hardware – PCs
2.	Anaconda – Python 3.7 Installation / Moodle-Code Runner
## Algorithm:
Step1:import pandas as pd.

Step2:Read the csv file.

Step3:Get the value of X and y variables.

Step4:Create the linear regression model and fit.

Step5:Predict the CO2 emission of a car where the weight is 2300kg, and the volume is 1300cm cube.

## Program:
```
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model, metrics
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Load California Housing dataset
housing = fetch_california_housing()
X = housing.data
y = housing.target

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

# Create and train linear regression model
reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)

# Results
print('Coefficients:', reg.coef_)
print('Variance score: {}'.format(reg.score(X_test, y_test)))

# Residual error plot
plt.style.use('fivethirtyeight')
plt.scatter(reg.predict(X_train), reg.predict(X_train) - y_train, color="green", s=10, label='Train data')
plt.scatter(reg.predict(X_test), reg.predict(X_test) - y_test, color="blue", s=10, label='Test data')
plt.hlines(y=0, xmin=0, xmax=50, linewidth=2)
plt.legend(loc='upper right')
plt.title("Residual errors")
plt.show()
```
## Output:
<img width="1017" height="815" alt="WhatsApp Image 2026-05-26 at 7 09 17 PM" src="https://github.com/user-attachments/assets/58c8fa89-42e3-4637-9175-1a10fe12a136" />

### Insert your output

<br>

## Result
Thus the multivariate linear regression is implemented and predicted the output using python program.
