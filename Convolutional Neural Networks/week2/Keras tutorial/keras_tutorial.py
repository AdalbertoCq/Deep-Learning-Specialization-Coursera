import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow


'''
Example of a quick setup for a Keras model NN. 
CONV2D --> Batch Normalization --> Activation RELU --> Max Pooling --> Flatten & FC
'''
def model(input_shape):
    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = Input(input_shape)

    # Zero-Padding: pads the border of X_input with zeroes
    X = ZeroPadding2D((3, 3))(X_input)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0')(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool')(X)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='HappyModel')

    return model


# GRADED FUNCTION: HappyModel

def HappyModel(input_shape):
    """
    Implementation of the HappyModel.

    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """

    ### START CODE HERE ###
    # Feel free to use the suggested outline in the text above to get started, and run through the whole
    # exercise (including the later portions of this notebook) once. The come back also try out other
    # network architectures as well.
    X_input = Input(input_shape)

    # Adding padding of 3.
    X = ZeroPadding2D((3, 3))(X_input)
    # Convolutional, F=7, S=1.
    X = Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(X)
    # Batch Normalization after convolution. Axis=3, normalize over the #channels.
    X = BatchNormalization(axis=3, name='bn0')(X)
    # ReLU
    X = Activation('relu')(X)

    # Max Pooling F=2.
    X = MaxPooling2D((2, 2), name='max_pool')(X)

    # Flatten output
    # Just one neuron for this layer.
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)

    # Model
    model = Model(inputs=X_input, outputs=X, name='HappyModel')

    ### END CODE HERE ###

    return model

'''
Remember how to code a model in Keras and the four steps leading to the evaluation of your model on the test set. Create->Compile->Fit/Train->Evaluate/Test.
'''

happyModel = HappyModel(input_shape=(64,64,3))
happyModel.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
happyModel.fit(x=X_train, y=Y_train, batch_size=60, epochs=1000)
preds = happyModel.evaluate(x=X_test, y=Y_test, batch_size=150)
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
'''
model.summary(): prints the details of your layers in a table with the sizes of its inputs/outputs
plot_model(): plots your graph in a nice layout. You can even save it as ".png" using SVG() if you'd like to share it on social media ;). 
It is saved in "File" then "Open..." in the upper bar of the notebook
'''
happyModel.summary()
plot_model(happyModel, to_file='HappyModel.png')
SVG(model_to_dot(happyModel).create(prog='dot', format='svg'))