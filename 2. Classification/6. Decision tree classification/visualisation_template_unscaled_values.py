# VISUALISATION TEMPLATE FOR UNSCALED VALUES
# THIS IS STRICLY FOR VISUALISATION, MODEL WON'T WORK WITHOUT SCALING
# SOURCE: https://www.udemy.com/machinelearning/learn/v4/questions/5607470
 
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values
 
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
 
#make non-scaled labels
x_axis = np.linspace( X_train[:, 0].min()-10 , X_train[:, 0].max()-2 , 6).round()
x_axis = list(map(str,list(map(int, x_axis))))
y_axis = np.linspace( X_train[:, 1].min() , X_train[:, 1].max() , 10).round()
y_axis = list(map(str,list(map(int, y_axis))))
 
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
 
# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
 
# Predicting the Test set results
y_pred = classifier.predict(X_test)
 
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
 
# Visualising the Training set results
from matplotlib.colors import ListedColormap
f, ax = plt.subplots()
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 0.2, stop = X_set[:, 0].max() + 0.2, step = 0.01),
                         np.arange(start = X_set[:, 1].min() - 0.2, stop = X_set[:, 1].max() + 0.2, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('red', 'green'))(i), label = j)
ax.set_xticklabels(x_axis)
ax.set_yticklabels(y_axis)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()