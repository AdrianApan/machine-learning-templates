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
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
"""
• https://i.imgur.com/XRBkXlt.jpg
• https://i.imgur.com/TuLRoaM.jpg 

It's enough to use just one dummy column / variable since you can instantly if NY is 0 then it's clearly Cali

!!! ALWAYS OMIT ONE DUMMY VARIABLE (if you have 10 include 9, if you have 5 include 4 etc.)

"""
X = X[:, 1:] # although python multiple linear regression lib is already taking care of it

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""
NO NEED for this as the multiple linear regresssion library will take care of it

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)

"""

####################################################################################
"""
--------------------------
MULTIPLE LINEAR REGRESSION
--------------------------

--------------------------------------------------
|					   							 |
|   y = b0 + b1 * x1 + b2 * x2 + ... + bn * xn   |
|					   							 |
--------------------------------------------------

• Used for multiple independent variable

• Example Questions Answered:

	Q: Do age and IQ scores effectively predict GPA?

	Q: Do weight, height, and age explain the variance in cholesterol levels?

• Assumptions: https://i.imgur.com/ZmKxD7O.png

"""
#####################################################################################

# FITTING MULTIPLE LINEAR REGRESSION TO THE TRAINING SET
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# PREDICTING THE TEST SET RESULTS
# Now visually compare y_test (real results) against y_pred (predicted results)
y_pred = regressor.predict(X_test)


"""
--------------------------------
#1 BACKWARD ELIMINATION (MANUAL)
--------------------------------
"""

# BUILDING THE OPTIMAL MODEL USING BACKWARD ELIMINATION
"""
CONSIDER: https://i.imgur.com/lhIn742.png

STEP 1: Select a significance level to stay in the model (SL = 0.05 or 5%)
STEP 2: Fit the full model with all possible predictors
STEP 3: Consider the predictor with the highest P-value. If P > SL go to STEP 4, else FIN (FIN = MODEL IS READY)
STEP 4: Remove the predictor
STEP 5: Fit model without this variable

"""
import statsmodels.formula.api as sm
# Append a new column (x0) by creating a new column with 50 rows of 1 and adding the X matrix to it, this way the 0x column comes first followed by the X matrix
# This is needed for the statsmodels library
X = np.append( arr =  np.ones((50,1)).astype(int), values = X, axis = 1)
# STEP 2
# The X_opt matrix will eventually only contain the variables that have high impact on the "profit" (y)
X_opt = X[:, [0, 1, 2, 3, 4, 5]] # We need to write out the X matrix specifying each row index (look at matrix X if need be in spyder)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# STEP 3
"""
If we run this in Spyder (CMD + CTRL + ENTER) we'll get something like this: https://i.imgur.com/EYV4Q4L.png
TAKEAWAY: The lower the P-value the more significant the independent variable will be with respsect to the dependent variable
"""
regressor_OLS.summary()

# We want to remove the independent variable with the highest P-value (in our case it's index 2) and repeat this until P-value < SL
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

"""
!! IMPORTANT !!!

Check the Adjusted R Square value as R Square can get biased when adding new variables.
Should be as close as possible to 1.
Reference: https://www.udemy.com/machinelearning/learn/v4/t/lecture/5794346?start=0

When comparing the coefficients (from the summary) talk about "per unit". 
So for example: 1 unit increase in R&D spent will drive a <<coef value>> (0.79 in our example) unit increase in profit.
Coefficients only talk about the additional effect
Reference: https://www.udemy.com/machinelearning/learn/v4/t/lecture/5795482?start=0

"""

"""
------------------------------------------------
#2 BACKWARD ELIMINATION WITH P-VAUES ONLY (AUTO)
------------------------------------------------
"""
import statsmodels.formula.api as sm

X = np.append( arr =  np.ones((50,1)).astype(int), values = X, axis = 1) # This is needed for the statsmodels library

def backwardElimination(x, SL):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    print(regressor_OLS.summary()) # just to see some results in the console
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)

"""
------------------------------------------------------------------
#3 BACKWARD ELIMINATION WITH P-VAUES AND ADJUSTED R SQUARED (AUTO)
------------------------------------------------------------------
Best fit as it check for adjusted R squared value, see above
comment for manual backward elimination
------------------------------------------------------------------
"""

import statsmodels.formula.api as sm

X = np.append( arr =  np.ones((50,1)).astype(int), values = X, axis = 1) # This is needed for the statsmodels library

def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((len(x),len(x[0]))).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)


"""
--------------------------------------------------------------------
#4 BACKWARD ELIMINATION WITH P-VAUES ONLY (AUTO) + SHOWING VARIABLES
--------------------------------------------------------------------
Example from Q&A, good to have it here just in case I ever need it
--------------------------------------------------------------------
"""
variables = ["Cons","FL","NY","R&D","Admin","Market"]

import statsmodels.formula.api as sm

X = np.append( arr =  np.ones((50,1)).astype(int), values = X, axis = 1) # This is needed for the statsmodels library

def backwardElimination(x, SL):
    numVars = len(x[0])
    for i in range(0,numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > SL:
            for j in range(0,numVars -i):
                if (regressor_OLS.pvalues[j].astype(float)== maxVar):
                    variables.pop(j)
                    x = np.delete(x,j,1)
    regressor_OLS.summary()
    return x               


SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)
regressor_OLS = sm.OLS(endog = y, exog = X_Modeled ).fit()
print(regressor_OLS.summary())
print (variables)
