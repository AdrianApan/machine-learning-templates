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

# Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)

# Formating it as a list of lists (required by the apyori.py function - see root folder)
transactions = []
for i in range(0, 7501): # len(dataset) = 7501
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

# Another way to do it
#transactions = dataset.values.tolist()

# And another one
#transactions = [ [item for item in row if not pd.isnull(item) ] for i,row in dataset.iterrows()]

#####################################################################################
"""
---------------------------
APRIORI RULE LEARNING (ARL)
---------------------------

• People who watched x also watched y

• Udemy explanation: https://www.udemy.com/machinelearning/learn/v4/t/lecture/6455322?start=251

• Apriori has 3 parts to it:

	1) Support
	- How many people watched the movie? Let's say 10 out of 100 saw M1 so Support = 10%

	2) Confidence
	- Let's say out of 100 people 40 watched M2. Out of those 40 how many say M1? (Let's say 7) Confidence = 7/40 (17.5%)

	3) Lift
	- Lift = Confidence/Support = 17.5/10 =  1.75%

• STEPS:
	1. Set a minimum support and confidence
	2. Take all the subsets in transactions having higher support than minimum support
	3. Take all the rules of this subsets having higher confidence than minimum confidence
	4. Sort the rules by decreasing lift


"""
#####################################################################################

# Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# min_support 3x7/7500 --> 3 products bought everyday for a whole week / total number of baskets bought in a week
# min_confidence --> can't set it too high else it get skewed (too obvious rules)
# min_lift --> https://www.udemy.com/machinelearning/learn/v4/t/lecture/5969220?start=2
# min_length --> rules with min 2 products recommendations

# Visualising the results
results = list(rules)

# Better way to visualise the results
"""
results_list = []
for i in range(0, len(results)):
    results_list.append('RULE:\t' + str(results[i][0]) + '\nSUPPORT:\t' + str(results[i][1]) + '\nCONFIDENCE:\t' + str(results[i][2][0].confidence) +'\nLIFT:\t' + str(results[i][2][0].lift))
"""



#####################################################################################
"""
----------------
VERSION FROM Q&A
----------------

REQUIREMENTS: pip install mlxtend

"""
#####################################################################################

"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
# Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
dataset.replace([np.nan], "nan" , inplace=True)
transactions = list(map(lambda x: [i for i in x],dataset.values))
 
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import time
 
support_threshold = 0.003
 
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)
df.drop(columns = 'nan', inplace = True)
# time 
start_time = time.process_time()
# apriori
frequent_itemsets = apriori(df, min_support=support_threshold, use_colnames=True)
# end time to calculation
end_time = time.process_time()
time_apriori = (end_time-start_time)/60
apriori_decimals = "%.2f" % round(time_apriori,2)
print("\n\nCompleted in %s minutes\n" % apriori_decimals)
 
lift = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

"""