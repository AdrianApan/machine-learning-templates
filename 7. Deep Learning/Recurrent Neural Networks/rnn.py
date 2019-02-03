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

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values # we take range 1:2 because we want a numpy array not just a vector
# For simplicity we use just 1 data column but also check out: https://www.udemy.com/deeplearning/learn/v4/questions/3554002

# Feature Scaling
# We use Normalisation in RNNs especially if the activation function is a sigmoid
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set) # we use a new var so that we can keep the original dataset

# Creating a data structure with 60 timesteps and 1 output
"""
• At each time T the RNN is going to look at the 60 stock prices before time T and based on the trends
  that is capturing during these 60 previous time steps it's going to predixct the next step

• 60 is something we can up after testing, 1 would be overfitting etc.

• https://www.udemy.com/deeplearning/learn/v4/t/lecture/8374802?start=0
"""
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train) # we make the vars numpy arrays so they'll be accepted by the RNN

# Reshaping - adding a new dimension to the numpy array
"""
• X_train.shape[0] is the number of rows in the dataset
• X_train.shape[1] is the number of columns in the dataset.
"""
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#######################################################################################################################
"""
------------------------------
RECURRENT NEURAL NETWORK (RNN)
------------------------------

• https://www.superdatascience.com/the-ultimate-guide-to-recurrent-neural-networks-rnn/
• https://www.superdatascience.com/ppt-the-ultimate-guide-to-recurrent-neural-networks-rnn/
• Used for time series analysis
• It's a supervised learning method, it's like a short term memory
• Applications: https://i.imgur.com/xHp8eD6.png (one to many, many to one, many to many)

VANISHING GRADIENT
------------------

• https://www.udemy.com/deeplearning/learn/v4/t/lecture/6820148?start=0
• https://i.imgur.com/iag13pq.png
• When unrevaling the temporal loop the further you go through your network the lower the gradient is and it's harder to train the weights
• Solutions for the problem: 
	Exploding Gradient
		Truncated Backpropagation
 		Penalties
 		Gradient clipoing
 	Vanishing Gradient
 		Weight initialization
		Echo state networks
		LSTM

LONG SHORT-TERM MEMORY (LSTM)
-----------------------------

• https://www.udemy.com/deeplearning/learn/v4/t/lecture/6820150?start=0
• Visual: https://i.imgur.com/xJnVxiE.png
• http://colah.github.io/posts/2015-08-Understanding-LSTMs/
• Practical intuition: https://www.udemy.com/deeplearning/learn/v4/t/lecture/6820154?start=0
• https://arxiv.org/pdf/1506.02078.pdf
• http://karpathy.github.io/2015/05/21/rnn-effectiveness/

LSTM VARIATIONS
---------------

• https://www.udemy.com/deeplearning/learn/v4/t/lecture/6820164?start=0
• Adding peepholes
• Connecting Forget Valve and Memory Valve
• Gated recurring units (GRUs) - quite popular - getting rid of the memory cell

"""
#######################################################################################################################

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()


"""
Dropout regularisation is to prevent overfitting: https://www.udemy.com/deeplearning/learn/v4/questions/2597526
units = number of LSTM cells
return_sequences = True (for the last layer we need to set it to False but False is the default value so we can leave it out)
input_shape = input shape in 3D - see reshape from above (we emit the first value)
"""
# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)


# Making the predictions and visualising the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
# https://www.udemy.com/deeplearning/learn/v4/t/lecture/8374820?start=0
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0) # axis = concat on vertical axis, horizontal is 1
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1) # we haven't used the iloc so we need this formating/reshaping step
inputs = sc.transform(inputs) # scaling
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price) #inversing the scaled values

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
