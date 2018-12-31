######################################################################################################################################
"""
----------------
Random Selection
----------------


• The random_selection.py algorithm helps us to show a random ad each time (each round) -  to be used to compare results against UCB.

• With the random_selection.py algorithm we got a total reward of aprox 1200 give or take (clicks on ads).

"""
######################################################################################################################################

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing Random Selection
import random
N = 10000
d = 10
ads_selected = []
total_reward = 0
for n in range(0, N):
    ad = random.randrange(d)
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    total_reward = total_reward + reward

# Visualising the results
plt.hist(ads_selected, ec='black')
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()