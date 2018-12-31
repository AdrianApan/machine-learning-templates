####################################################################################
"""
------------------
DATA PREPROCESSING
------------------
"""
####################################################################################

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#####################################################################################
"""
-----------------
THOMPSON SAMPLING
-----------------

• Intuition: https://www.udemy.com/machinelearning/learn/v4/t/lecture/6456840?start=1

• It's a probabilistic algorithm (UCB is deterministic)

• UCB vs Thompson Sampling: https://www.udemy.com/machinelearning/learn/v4/t/lecture/6468288?start=0

• Can accomodate delayed feedback (UCB can't, UCB needs to feed back the result to start a new round)

"""
#####################################################################################

import random
N = 10000
d = 10
ads_selected = []
numbers_of_rewards_1 = [0] * d
numbers_of_rewards_0 = [0] * d
total_reward = 0
for n in range(0, N):
    ad = 0
    max_random = 0
    for i in range(0, d):
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:
        numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
    else:
        numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1
    total_reward = total_reward + reward

# Visualising the results - Histogram
plt.hist(ads_selected, ec='black')
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()


"""
-------------
CODE FROM Q&A
-------------

n = len(dataset.columns)
N=numpy.zeros((n,2))
theta=numpy.zeros(n)
ads_selected =[]
 
for round in range(len(dataset.index)):
    for i in range(n):        
        theta[i] = numpy.random.beta(N[i][1]+1, N[i][0]+1)
 
    ad = theta.argmax()
    ads_selected.append(ad)
    N[ad][dataset.values[round,ad]] += 1
                        
print("Click through rate :",  int(N[:,1].sum()))
"""