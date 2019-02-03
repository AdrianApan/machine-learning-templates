#######################################################################################################################
"""
-------------------------------
ARTIFICIAL NEURAL NETWORK (ANN)
-------------------------------

1. STRUCTURE
-------------

• Neuron: https://www.udemy.com/machinelearning/learn/v4/t/lecture/6760380?start=0

• Input (x1, x2, x3, ... xm):
	- https://i.imgur.com/22iUT7s.png
	- Are in fact independent variables
	- Need standardize (or normalize)

• Synapses (w1, w2, w3, .... wm)
	- Weights are crucial and it's how NNs learn

• Neuron:
	- Weighted sum of all the input values + activation function
	- Weighted sum of all the input values: w1x1 + w2x2 + w3x3 + .... + wmxm

• Output (y1, y2, y3 ... yp)
	- Continuous (price), binary (wil exit yes/no), categorical (in which case there will be several output values)


2. ACTIVATION FUNCTION
-----------------------

https://www.udemy.com/machinelearning/learn/v4/t/lecture/6760384?start=0

• There are multiple activation functions, but some of the most predominant ones are:
	- Threshold function: https://i.imgur.com/l2BfNnr.png - super simple x < 0 sends 0, x >= 0 sends 1
	- ! Sigmoid function: https://i.imgur.com/kH9rBfR.png - useful for output layers especially when working with probabilities
	- ! Rectifier function: https://i.imgur.com/uzrD1Pw.png - one of the most popular activation functions for NNs
	- Hyperbolic tangent function (tahn): https://i.imgur.com/LdfVU8B.png - similar to sigmoid but goes below 0 too (-1 to 0)



3. TRAINING
------------

	- Udemy: https://www.udemy.com/machinelearning/learn/v4/t/lecture/6760388?start=1
	- Cost function: C = Σ 1/2(yˆ-y)² where Σ is the sum, y^ is the output value and y is the actual value (our goal is to minimize C)

	3.1. GRADIENT DESCENT
	----------------------
		- Run all the rows then adjust weights
		- https://i.imgur.com/UHrexd5.png
		- Udemy: https://www.udemy.com/machinelearning/learn/v4/t/lecture/6760390?start=0


	3.2. STOCHASTIC GRADIENT DESCENT
	--------------------------------
		- Run one row then adjust weight, run next row and ajust weights again (https://i.imgur.com/xBCLnWL.png)
		- https://i.imgur.com/AVpWrr9.png
		- To be used when we use a different cost function or when the cost function is not convex
		- Udemy: https://www.udemy.com/machinelearning/learn/v4/t/lecture/6760392?start=0

"""
#######################################################################################################################

"""
For OSX fuckery:
	- conda create -n ml python=3.6 keras
	- creates a new env called "ml" which will be using pything 3.6
	- install shit from here: https://medium.com/i-want-to-be-the-very-best/installing-keras-tensorflow-using-anaconda-for-machine-learning-44ab28ff39cb
	- install tensorflow with this:
		pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.12.0-py3-none-any.whl
	- install theano with this:
		pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

	Commands to remember:
		- conda activate ml
		- conda deactivate ml
"""

#######################################################################################################################

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:] #removing a dummy var from the country

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
#from keras.layers import Dropout # used for Dropout Regularization (see below)

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
"""
• input_dim -> we have 11 independent variables
• output_dim (units) -> how many nodes (neurons) we want to add to the hidden layer. Tip from course: use the average of inputs + outputs although it can be done by creating a separate dataset and using k-means more about tihs later
• init ->  starting weight values
• activation -> activation function (relu = rectifier, see above)
"""
classifier.add(Dense(input_dim=11, units = 6, kernel_initializer="uniform", activation="relu"))
#classifier.add(Dropout(p = 0.1)) # p - what percentage of the neurons should be dropped/disabled at each iteration (0.1 would be 10%). Add this to new layers as needed.


# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation="relu"))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = "uniform", activation = "sigmoid"))

# Compiling the ANN
"""
• optimizer -> is the algorithm to find the optimal set of weights in the NN (adam = stochastic gradient descent)
• loss -> loss function of the SGD algorithm
• metrics -> criterion to evaluate the model
"""
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5) # get results in the form of TRUE or FALSE, remove if you want to keep %

# Predicting a single new observation
"""
Predict if the customer with the following informations will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000
"""
new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]))) # we use 0.0 at the 1st categorical value to avoid the console error
new_prediction = (new_prediction > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Part 4 - Evaluating, Improving and Tuning the ANN

# Evaluating the ANN with k-fold cross validation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()

# Improving the ANN
# Dropout Regularization is used to reduce overfitting if needed in Deep Learning
# See lines 119, 132

# Tuning the ANN with Grid Search
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

#####################################################################################################################################
"""
-----
NOTES
-----

• HOW TO SAVE THE MODEL
	- https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model
	- https://machinelearningmastery.com/save-load-keras-deep-learning-models/

• Great Q&A about Epochs: https://www.udemy.com/machinelearning/learn/v4/questions/2220620

• Apply for a new customer: https://www.udemy.com/machinelearning/learn/v4/questions/5692329

• Overfitting: https://www.udemy.com/machinelearning/learn/v4/questions/4953864

• Feature importance: https://stackoverflow.com/questions/45361559/feature-importance-chart-in-neural-network-using-keras-in-python

• Explaining predictions in NNs: https://github.com/marcotcr/lime
"""
#####################################################################################################################################