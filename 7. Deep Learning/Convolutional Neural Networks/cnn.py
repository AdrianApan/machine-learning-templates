#######################################################################################################################
"""
----------------------------------
CONVOLUTIONAL NEURAL NETWORK (CNN)
----------------------------------

• CNN structure: https://i.imgur.com/mFz0WRj.png
• Summary: https://www.udemy.com/machinelearning/learn/v4/t/lecture/6761150?start=0


1. CONVOLUTION
---------------

• Udemy: https://www.udemy.com/machinelearning/learn/v4/t/lecture/6761140?start=0
• Input image ⓧ Feature detector (aka. Kernel or Filter) = Feature map (aka. Convolt feature or Activation map)
• We create multiple feature maps to obtain our first convolution layer


    1.1. ReLU LAYER
    ---------------
    
    • Udemy: https://www.udemy.com/machinelearning/learn/v4/t/lecture/6761142?start=0
    • Helps removing linearity (increase non-linearity)


2. POOLING
-----------

• Udemy: https://www.udemy.com/machinelearning/learn/v4/t/lecture/6761144?start=0
• https://i.imgur.com/VoxsHbN.png
• We apply the max pooling to each feature map
• http://ais.uni-bonn.de/papers/icann2010_maxpool.pdf
• Helps to distinguish the features even when these are skewed or the pattern is slightly different etc., also helps with overfitting
  as it reduces the size and gets rid of the noise while preserving the feature
• Aka. Downsampling


3. FLATTENING
--------------

• Udemy: https://www.udemy.com/machinelearning/learn/v4/t/lecture/6761146?start=0
• https://i.imgur.com/wofSCXY.png
• Basically matrix to vector so we can use it as input layer for a future ANN


4. FULL CONNECTION
-------------------

• Udemy: https://www.udemy.com/machinelearning/learn/v4/t/lecture/6761148?start=0
• Adding ANN after flattening basically
• In CNN Hidden Layers are called Fully Connected Layers
• In CNN Cost Function is called Loss Function (cross entropy / mean squared error)
• Same as in an ANN the weights get adjusted via backpropagation, but in a CNN the Feature Detector also gets adjusted


4. SOFTMAX AND CROSS-ENTROPY
-----------------------------

• Udemy: https://www.udemy.com/machinelearning/learn/v4/t/lecture/6761152?start=0
• Output layer neurons of a CNN add up to 1 (example: dog 95% and cat 5%) because of the Softmax function
• Cross-entropy goes hand in hand with the Softmax function
• Cross-entropy: https://i.imgur.com/zGv5O1t.png
• Cross-entropy will improve the CNN much better than mean squared error:
    - https://i.imgur.com/rBRdxb5.png
    - https://www.udemy.com/deeplearning/learn/v4/t/lecture/6761110?start=713
• Cross-entropy is ok for classification if we want regression then we are better of with mean squared error

"""
#######################################################################################################################

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
"""
• 32, (3, 3) -> 32 feature detectors, 3x3 size (perhaps use 64 when on GPU)
• input_shape -> the shape of the input image, 64x64, 3 = 3d array (colored images so 3 channels RGB) we use "1" if we deal with b&w images (perhaps use 256x256 when on GPU)
• activation = relu -> activation function (rectifier) used to remove negative pixel in order to increase non-linearity in the model
"""
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
# We don't need input_shape anymore
# This new convolution layer helps improving the accuracy
# When adding a new convolution layer (3rd for example) we can double the feature detector 32 becomes 64 then 128 at a new layer etc.
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(activation = 'relu', units=128))
classifier.add(Dense(activation = 'sigmoid', units=1))  # sigmoid because of binary outcome x or y but if we have more we need to use softmax

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) # for more categories (output neurons) we would use categorical_entropy

# Part 2 - Fitting the CNN to the images
# ---> IMAGE AUGMENTATION
# Using keras for preprocessing the images to avoid oferfitting
# NOTE: Great accuracy result on the training set but much lower accuracy on the test set = overfitting
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)
# <--- END OF IMAGE AUGMENTATION
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64), # the target size of the images expected by the CNN (defined above in Step 1)
                                                 batch_size = 32, # batch size of images after which the weights will be adjusted
                                                 class_mode = 'binary') # if more categories we need to use "categorical"

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = (8000/32), # number of images that we have in the training set / batch size
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000) # number of images that we have in the test set / batch size
                         # use_multiprocessing = True,
                         # workers = 4)

# Saving the model
# https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model

# Improving the model
# https://www.udemy.com/deeplearning/learn/v4/questions/2276518
# https://www.udemy.com/deeplearning/learn/v4/t/lecture/6744898?start=0

# Part 3 - Making new predictions
# https://www.udemy.com/deeplearning/learn/v4/t/lecture/6798970?start=0
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

"""
--------
FROM Q&A
--------

How to predict more images with CNN:
1) add more folders for categories you want to train
2) make the output activation 'softmax'
3) output_dimensions = n_categories (dont subtract
4) change the loss from 'binary_crossentropy' to 'categorical_crossentropy'
5) change class_mode from 'binary' to 'categorical'
"""