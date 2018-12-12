#####################################################################################
"""
------------------
DATA PREPROCESSING
------------------
"""
#####################################################################################

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values # We use 1:2 for the variable to be consider a matrix (X) - this will not include column with index 2 since the upper bound of a range in python is excluded, dooh!
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""
NO NEED for this because:
	1. Our number of observations is really low (just 10), not much data
	2. We want an accurate prediction so we can't miss any data, we need as much data as possible

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

"""

# Feature Scaling
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

"""


#######################################################################################################################
"""
----------------------------
POLYNOMIAL LINEAR REGRESSION
----------------------------

---------------------------------------------
|					                        |
|   y = b0 + b1x1 + b2x1ˆ2 + ... + bnx1ˆn   |
|					                        |
---------------------------------------------

• Graph explanation: https://i.imgur.com/Wi8143G.png

• Use case: 
	- Used to describe how diseases spread

• Assumptions: https://i.imgur.com/ZmKxD7O.png

"""
#######################################################################################################################

# FITTING LINEAR REGRESSION TO THE DATASET
# THIS IS USED ONLY FOR REFERENCE (SO THAT WE CAN COMPARE THE RESULTS WITH THE POLYNOMIAL REGRESSION RESULTS)
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# FITTING POLYNOMIAL REGRESSION TO THE DATASET
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4) # Degree param is important and best to decide by visualising the graph: https://udemy-images.s3.amazonaws.com/redactor/raw/2018-12-03_03-05-07-72d59355384f3715c8cf3c68a93d1808.png
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# VISUALISING THE LINEAR REGRESSION RESULTS
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# VISUALISING THE POLYNOMIAL REGRESSION RESULTS
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# VISUALISING THE POLYNOMIAL REGRESSION RESULTS (FOR HIGHER RESOLUTION AND SMOOTHER CURVE)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# PREDICTING A NEW RESULT WITH LINEAR REGRESSION
lin_reg.predict(6.5)

# PREDICTING A NEW RESULT WITH POLYNOMIAL REGRESSION
lin_reg_2.predict(poly_reg.fit_transform(6.5))