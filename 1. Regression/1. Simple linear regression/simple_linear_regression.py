#######################################################################################################################
"""
------------------
DATA PREPROCESSING
------------------
"""
#######################################################################################################################

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)

"""


#######################################################################################################################
"""
------------------------
SIMPLE LINEAR REGRESSION
------------------------

------------------------
|					   |
|   y = b0 + b1 * x1   |
|					   |
------------------------

• Used for 1 independent variable

• Linear regression has many practical uses. Most applications fall into one of the following two broad categories:
  If the goal is prediction, or forecasting, or error reduction, linear regression can be used to fit a predictive
  model to an observed data set of values of the response and explanatory variables.

• https://www.quora.com/What-are-some-real-world-applications-of-simple-linear-regression

• https://en.wikipedia.org/wiki/Linear_regression

• Assumptions: https://i.imgur.com/ZmKxD7O.png

"""
#######################################################################################################################


# FITTING THE SIMPLE LINEAR REGRESSION TO THE TRAINING SET
# Learn on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# PREDICTING THE TEST SET RESULTS
# Check what the model has learned (using the test set)
y_pred = regressor.predict(X_test) #

# VISUALISING THE TRAINING SET RESULTS
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# VISUALISING THE TEST SET RESULTS
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue') #regression line plot
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Superimposed charts
# plt.scatter(X_train,y_train, color='red')
# plt.plot(X_train, regressor.predict(X_train),color ='purple')
# plt.title("Salary vs experience(Training set)")
# plt.xlabel('Years of experience')
# plt.ylabel("Salary")
# plt.scatter(X_test, y_test, color = "green")
# plt.show()

# Predicting a single value (ex: salary for 21 years experience)
# singlePrediction = regressor.predict(np.asarray([21]).reshape(1, -1))

# Model accuracy
# from sklearn.metrics import mean_squared_error
# print(mean_squared_error(y_test, y_pred)
# or with a simple calculation
# np.sum((y_pred - y_test)**2)/len(y_test)