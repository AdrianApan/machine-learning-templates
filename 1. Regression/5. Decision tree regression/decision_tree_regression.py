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
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

"""

# Feature Scaling
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

"""


#####################################################################################
"""
------------------------
DECISION TREE REGRESSION
------------------------

• You can apply a decision tree to linear or non-linear data as well as classification
  or regression. Decision trees are very versatile.

• Graph: https://i.imgur.com/tpF32rc.png

• Each split is called a leaf, it's splits until it can't add any more information or
  when you reach less than 5% of your total points in a leaf

• Bookmark: https://www.udemy.com/machinelearning/learn/v4/t/lecture/5732730?start=351

"""
#####################################################################################


# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0) # random_state is used just so that we have the same results with the course tutorial
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict(6.5)

# Visualising the Decision Tree Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()