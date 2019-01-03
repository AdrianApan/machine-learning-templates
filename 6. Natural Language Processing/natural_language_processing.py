###################################
"""
---------------------------------
NATURAL LANGUAGE PROCESSING (NLP)
---------------------------------

"""
###################################


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3) # quoting param is configured in order to ignore the quotes from the text

# Cleaning the texts
"""
• Removing stopwords (the, of, on etc.), applying stemming (loved = love, hated = hate etc.), lowercase only. remove punctuation

• NOT and NO will also be considered a stopword which can complicate things ("crust not good" becomes "crust good") - read up on this " n-gramming" or simple example could be:

stopwords = nltk.corpus.stopwords.words('english')
stopwords.remove('not')

• Might be also worth looking into https://pythonprogramming.net/lemmatizing-nltk-tutorial/ instead of stemming

• My example: https://repl.it/repls/AgonizingValidGraphs
"""
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) # remove everything except a to z and A to Z and replace with a space
    review = review.lower() # transform to lowercase
    review = review.split() # string to list (of words)
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
"""

• Udemy link: https://www.udemy.com/machinelearning/learn/v4/t/lecture/6065884?start=0

• We create a column for each word = independent variables and the positive/negative (1/0) outcomes
  = dependent variable (or in other words this is tokenization). This way we can use a classification
  algorithm (done below). We had to clean the corpus (see above) to reduce the number of independent
  variables (sparse matrix)

"""
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray() # sparse matrix
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# Predicting a new review (from Q&A)
"""
def predict(new_review):    
    new_review = re.sub("[^a-zA-Z]", " ", new_review)    
    new_review = new_review.lower().split()
    new_review = [ps.stem(word) for word in new_review if word not in set(stopwords.words("english"))]    
    new_review = " ".join(new_review)    
    new_review = [new_review]    
    new_review = cv.transform(new_review).toarray()    
    if classifier.predict(new_review)[0] == 1:
        return "Positive"    
    else:        
        return "Negative"

"""

"""
For homework:

Apply other classification models.

Evaluate the performance of each of these models. Try to beat the Accuracy obtained in the tutorial.
But remember, Accuracy is not enough, so you should also look at other performance metrics like
Precision (measuring exactness), Recall (measuring completeness) and the F1 Score (compromise between Precision and Recall).

Please find below these metrics formulas (TP = # True Positives, TN = # True Negatives, FP = # False Positives, FN = # False Negatives):

Accuracy = (TP + TN) / (TP + TN + FP + FN)

Precision = TP / (TP + FP)

Recall = TP / (TP + FN)

F1 Score = 2 * Precision * Recall / (Precision + Recall)

***************************************************************************************************************************************

TP = cm[1][1]
TN = cm[0][0]
FP = cm[1][0]
FN = cm[0][1]

accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * precision * recall / (precision + recall)

***************************************************************************************************************************************
"""