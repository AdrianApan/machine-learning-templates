##################
"""
DATA PREPROCESSING
"""
##################

# IMPORTING THE LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# IMPORTING THE DATASET
dataset = pd.read_csv('Data.csv')

# Independent variable
X = dataset.iloc[:, :-1].values # all lines(:) and all columns except the last one (:-1)
# Dependent variable
y = dataset.iloc[:, 3].values # all lines (:) and just the last column (column with index 3)

"""
Q: Why uppercase X instead of lowercase x? (or why y and not Y)
A: Naming convention... A capital letter is the convention for a matrix, whereas a lower case letter is convention for a vector.
"""

# TAKING CARE OF THE MISSING DATA
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3]) # all rows and columns with index 1 and 2 (upper is excluded - 3 is excluded)
X[:, 1:3] = imputer.transform(X[:, 1:3])

# ENCODING CATEGORICAL DATA

"""
We need to get rid of the categorical variable (not numbers) - in our case Country and Purchased
https://www.udemy.com/machinelearning/learn/v4/t/lecture/5683428?start=0
"""

# Encoding the independent variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0]) # We encoded the country values (France = 0, Germany = 1, Spain = 2)

"""
We can't compare countries we can't say Germany > France, makes no sense
If we would be using something like SIZE (S, M, L, XL etc) then it's fine
To prevent this we use dummy variables by splitting the Country columns in 3 colums (France | Germany | Spain)
https://i.imgur.com/TbUFCTj.png


COUNTRIES                 FRANCE |    | SPAIN |    | GERMANY |
----------                --------    ---------     ----------
France                      1             0              0
Spain                       0             1              0
Germany                     0             0              1

"""

onehotencoder = OneHotEncoder(categorical_features = [0]) # which column index are we talking about (0)
X = onehotencoder.fit_transform(X).toarray()

# Encoding the dependent variable
# Since it's a dependent variable the ML model will know that it's a category (0 = no, 1 = yes)
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


# SPLITTING THE DATASET INTO THE TRAINING SET AND TEST SET
from sklearn.model_selection import train_test_split
# Test size should be usually 20% or 25%,
# This means that in our case out of 10 entries 2 will go into test set and 8 to the train set
# random_state is used just for the course so we get the same results, not really needed but 42 is a good value to use
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

# FEATURE SCALING
# We need to standardise or normalise the values else the small values might be ignore - we need to bring them in the same range like -1 to 1
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train= sc_X.fit_transform(X_train)
X_test= sc_X.transform(X_test)

"""
The fit function will fit data to an equation or algorithm. The transform funciton will apply that fitting equation to the data.
There are some cases where we want to apply a the fit of one variable to another, and there are some cases where we want to fit and transform the same data.
For example in the scaling sections in out code we use fit_transform() X_train and we use the fit to transform the X_test data in order to not bias our model.
"""