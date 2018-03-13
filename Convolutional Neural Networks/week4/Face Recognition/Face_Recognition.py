from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *

np.set_printoptions(threshold=np.nan)

'''
Based on FaceNet paper.
'''

# Loading the Inception Network already trained.
FRmodel = faceRecoModel(input_shape=(3, 96, 96))
print("Total Params:", FRmodel.count_params())

'''
 The Triplet Loss
 
Training will use triplets of images (A,P,N):

    A is an "Anchor" image--a picture of a person.
    P is a "Positive" image--a picture of the same person as the Anchor image.
    N is a "Negative" image--a picture of a different person than the Anchor image.

These triplets are picked from our training dataset. We will write (A(i),P(i),N(i)) to denote the is-th training example.

You'd like to make sure that an image A(i) of an individual is closer to the Positive P(i) than to the Negative image N(i) by at least a margin αα:

∣∣f(A(i))−f(P(i))∣∣22+α<∣∣f(A(i))−f(N(i))∣∣22

You would thus like to minimize the following "triplet cost":

J=∑i=1m[∣∣f(A(i))−f(P(i))∣∣22−∣∣f(A(i))−f(N(i))∣∣22+α]+
                 ⏟                    ⏟
                (1)                   (2)

Here, we are using the notation "[z]+" to denote max(z,0).

Notes:

    The term (1) is the squared distance between the anchor "A" and the positive "P" for a given triplet; you want this to be small.
    The term (2) is the squared distance between the anchor "A" and the negative "N" for a given triplet, you want this to be relatively large, so it thus makes sense to have a minus sign preceding it.
    αα is called the margin. It is a hyperparameter that you should pick manually. We will use α=0.2α=0.2.

Most implementations also normalize the encoding vectors to have norm equal one (i.e., ∣∣f(img)∣∣2∣∣f(img)∣∣2=1); you won't have to worry about that here.

    Compute the distance between the encodings of "anchor" and "positive": ∣∣f(A(i))−f(P(i))∣∣22∣∣f(A(i))−f(P(i))∣∣22
    Compute the distance between the encodings of "anchor" and "negative": ∣∣f(A(i))−f(N(i))∣∣22∣∣f(A(i))−f(N(i))∣∣22
    Compute the formula per training example: ∣∣f(A(i))−f(P(i))∣−∣∣f(A(i))−f(N(i))∣∣22+α∣∣f(A(i))−f(P(i))∣−∣∣f(A(i))−f(N(i))∣∣22+α
    Compute the full formula by taking the max with zero and summing over the training examples:
    J=∑i=1m[∣∣f(A(i))−f(P(i))∣∣22−∣∣f(A(i))−f(N(i))∣∣22+α]+(3)
'''
def triplet_loss(y_true, y_pred, alpha=0.2):
    """
    Implementation of the triplet loss as defined by formula (3)

    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)

    Returns:
    loss -- real number, value of the loss
    """

    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    # (M samples, dimension per image)
    # Step 1: Compute the (encoding) distance between the anchor and the positive, you will need to sum over axis=-1
    pos_dist = tf.reduce_sum(tf.square(tf.substract(anchor, positive)), axis=-1)
    # Step 2: Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
    neg_dist = tf.reduce_sum(tf.square(tf.substract(anchor, negative)), axis=-1)
    # Step 3: subtract the two previous distances and add alpha.
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(basic_loss)

    return loss

'''
Compute the encoding of the image from image_path
Compute the distance about this encoding and the encoding of the identity image stored in the database
Open the door if the distance is less than 0.7, else do not open.

As presented above, you should use the L2 distance Using Euclidean distance. 
(Note: In this implementation, compare the L2 distance, not the square of the L2 distance, to the threshold 0.7.) 
'''
def verify(image_path, identity, database, model):
    """
    Function that verifies if the person on the "image_path" image is "identity".

    Arguments:
    image_path -- path to an image
    identity -- string, name of the person you'd like to verify the identity. Has to be a resident of the Happy house.
    database -- python dictionary mapping names of allowed people's names (strings) to their encodings (vectors).
    model -- your Inception model instance in Keras

    Returns:
    dist -- distance between the image_path and the image of "identity" in the database.
    door_open -- True, if the door should open. False otherwise.
    """

    # Step 1: Compute the encoding for the image. Use img_to_encoding() see example above.
    encoding = img_to_encoding(image_path, model)

    # Step 2: Compute distance with identity's image
    dist = np.linalg.norm(database[identity] - encoding)

    # Step 3: Open the door if dist < 0.7, else don't open
    if dist < 0.7:
        print("It's " + str(identity) + ", welcome home!")
        door_open = True
    else:
        print("It's not " + str(identity) + ", please go away")
        door_open = False

    return dist, door_open

'''
        Face Recognition

Your face verification system is mostly working well. But since Kian got his ID card stolen, when he came back to the house that evening he couldn't get in!

To reduce such shenanigans, you'd like to change your face verification system to a face recognition system. 
This way, no one has to carry an ID card anymore. An authorized person can just walk up to the house, and the front door will unlock for them!

You'll implement a face recognition system that takes as input an image, and figures out if it is one of the authorized persons (and if so, who). 
Unlike the previous face verification system, we will no longer get a person's name as another input.

    Compute the target encoding of the image from image_path
    Find the encoding from the database that has smallest distance with the target encoding.
        Initialize the min_dist variable to a large enough number (100). It will help you keep track of what is the closest encoding to the input's encoding.
        Loop over the database dictionary's names and encodings. To loop use for (name, db_enc) in database.items().
            Compute L2 distance between the target "encoding" and the current "encoding" from the database.
            If this distance is less than the min_dist, then set min_dist to dist, and identity to name.

'''
def who_is_it(image_path, database, model):
    """
    Implements face recognition for the happy house by finding who is the person on the image_path image.

    Arguments:
    image_path -- path to an image
    database -- database containing image encodings along with the name of the person on the image
    model -- your Inception model instance in Keras

    Returns:
    min_dist -- the minimum distance between image_path encoding and the encodings from the database
    identity -- string, the name prediction for the person on image_path
    """

    ## Step 1: Compute the target "encoding" for the image. Use img_to_encoding() see example above.
    encoding = img_to_encoding(image_path, model)

    ## Step 2: Find the closest encoding ##

    # Initialize "min_dist" to a large value, say 100 (≈1 line)
    min_dist = 100

    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():

        # Compute L2 distance between the target "encoding" and the current "emb" from the database. (≈ 1 line)
        dist = np.linalg.norm(db_enc - encoding)

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
        if dist < min_dist:
            min_dist = dist
            identity = name

    if min_dist > 0.7:
        print("Not in the database.")
    else:
        print("it's " + str(identity) + ", the distance is " + str(min_dist))

    return min_dist, identity



FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
load_weights_from_FaceNet(FRmodel)
verify("images/camera_0.jpg", "younes", database, FRmodel)