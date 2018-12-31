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
----------------------------
UPPER CONFIDENCE BOUND (UCB)
----------------------------

• Exploration + Exploitation (A)

• A modern application could be ads (finding best ad)

• The multi-armed bandit problem: https://i.imgur.com/T8bHb01.png

• The dataset in this case is just a simulation. In a real life scenario we would have a live stream of data.

• UCB visualized:
    - https://www.dataforums.org/project_list/ucb/
    - https://github.com/Kprzeorski/Udemy_Machine_Learning/blob/master/UCB_Animate.py

"""
#####################################################################################

import math
N = 10000 # total number of rounds
d = 10 # ads (10 version of the ad)
ads_selected = [] # vector that contains the list of all the different version of ads that are selected at each round
numbers_of_selections = [0] * d # creating a vector of size "d" containing only 0s
sums_of_rewards = [0] * d # creating a vector of size "d" containing only 0s
total_reward = 0
for n in range(0, N): # n = rounds - for round 1, round 2, round 3 ... round n
    ad = 0
    max_upper_bound = 0
    # During the first ten rounds the algorithm selects on purpose each one of the ten ads to have a first insight into the users response of each of the ten ads
    # In real world for the first ten users connecting to the webpage, you would show ad 1 to the first user, ad 2 to the second user, ad 3 to the third user,..., ad 10 to the 10th user. Then the algorithm starts.
    for i in range(0, d):
        if (numbers_of_selections[i] > 0):
            # These are coming from the formula (check folder for UCB_Algorithm_Slide.png)
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400 # 10ˆ400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward

# Visualising the results
plt.hist(ads_selected, ec='black')
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()

"""
-------------
CODE FROM Q&A
-------------

c = len(dataset.columns)
ads_selected = []
UCB = numpy.ones(c)/0
sums_of_rewards = numpy.zeros(c)
numbers_of_selections = numpy.zeros(c)
 
for n in range(len(dataset)):
    ad=UCB.argmax()
    numbers_of_selections[ad]  +=1
    average_reward = sums_of_rewards[ad]/numbers_of_selections[ad]
    delta_i = numpy.sqrt(3/2*numpy.log10(n+1)/numbers_of_selections[ad])
    UCB[ad] = average_reward + delta_i
    
    ads_selected.append(ad)
    
    sums_of_rewards[ad] += dataset.values[n,ad] 
"""