import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *
import numpy as np
import tensorflow as tf

'''
How do you ensure the generated image G matches the content of the image C?

As we saw in lecture, the earlier (shallower) layers of a ConvNet tend to detect lower-level features such as edges and simple textures, 
and the later (deeper) layers tend to detect higher-level features such as more complex textures as well as object classes.

We would like the "generated" image G to have similar content as the input image C. 
Suppose you have chosen some layer's activations to represent the content of an image. 
In practice, you'll get the most visually pleasing results if you choose a layer in the middle of the network--neither too shallow nor too deep. 
(After you have finished this exercise, feel free to come back and experiment with using different layers, to see how the results vary.)

So, suppose you have picked one particular hidden layer to use. Now, set the image C as the input to the pretrained VGG network, and run forward propagation. 
Let a(C)a(C) be the hidden layer activations in the layer you had chosen. (In lecture, we had written this as a[l](C)a[l](C), 
but here we'll drop the superscript [l][l] to simplify the notation.) 
This will be a nH×nW×nCnH×nW×nC tensor. Repeat this process with the image G: Set G as the input, and run forward progation. Let
a(G)
a(G)
be the corresponding hidden layer activation. We will define as the content cost function as:

Jcontent(C,G)=14×nH×nW×nC∑all entries(a(C)−a(G))2(1)
Jcontent(C,G)=14×nH×nW×nC∑all entries(a(C)−a(G))2

Here, nH,nWnH,nW and nCnC are the height, width and number of channels of the hidden layer you have chosen, and appear in a normalization term in the cost. 
For clarity, note that a(C)a(C) and a(G)a(G) are the volumes corresponding to a hidden layer's activations. 
In order to compute the cost Jcontent(C,G)Jcontent(C,G), it might also be convenient to unroll these 3D volumes into a 2D matrix, as shown below. 
(Technically this unrolling step isn't needed to compute JcontentJcontent, 
but it will be good practice for when you do need to carry out a similar operation later for computing the style const JstyleJstyle.)
'''
def compute_content_cost(a_C, a_G):
    """
    Computes the content cost

    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G

    Returns:
    J_content -- scalar that you compute using equation 1 above.
    """

    m, n_H, n_W, n_C = a_G.shape

    new_shape = [int(m), int(n_H * n_W), int(n_C)]

    # Reshape a_C and a_G
    a_C_unrolled = tf.reshape(a_C, new_shape)
    a_G_unrolled = tf.reshape(a_G, new_shape)

    # compute the cost with tensorflow
    J_content = (.25 / float(int(n_H * n_W * n_C))) * tf.reduce_sum(np.power(a_G_unrolled - a_C_unrolled, 2))

    return J_content

'''
The style matrix is also called a "Gram matrix." 
In linear algebra, the Gram matrix G of a set of vectors (v1,…,vn)(v1,…,vn) is the matrix of dot products, whose entries are Gij=vTivj=np.dot(vi,vj)Gij=viTvj=np.dot(vi,vj). 
In other words, Gij compares how similar vivi is to vjvj: If they are highly similar, you would expect them to have a large dot product, and thus for Gij to be large.

Note that there is an unfortunate collision in the variable names used here. 
We are following common terminology used in the literature, but GG is used to denote the Style matrix (or Gram matrix) as well as to denote the generated image GG. 
We will try to make sure which GG we are referring to is always clear from the context.

In NST, you can compute the Style matrix by multiplying the "unrolled" filter matrix with their transpose:

The result is a matrix of dimension (nC,nC)(nC,nC) where nCnC is the number of filters. The value Gij measures how similar the activations of filter ii are to the activations of filter jj.

One important part of the gram matrix is that the diagonal elements such as Gii also measures how active filter ii is. 
For example, suppose filter ii is detecting vertical textures in the image. 
Then Gii measures how common vertical textures are in the image as a whole: If Gii is large, this means that the image has a lot of vertical texture.

By capturing the prevalence of different types of features (Gii), as well as how much different features occur together (Gij), the Style matrix GG measures the style of an image. 
'''
def gram_matrix(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_H*n_W)

    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """

    GA = tf.matmul(A, tf.transpose(A))
    
    return GA

'''
Style cost

After generating the Style matrix (Gram matrix), your goal will be to minimize the distance between the Gram matrix of the "style" image S and that of the "generated" image G. 
For now, we are using only a single hidden layer a[l]a[l], and the corresponding style cost for this layer is defined as:

J[l]style(S,G)=14×nC2×(nH×nW)2∑i=1nC∑j=1nC(G(S)ij−G(G)ij)2(2)
Jstyle[l](S,G)=14×nC2×(nH×nW)2∑i=1nC∑j=1nC(Gij(S)−Gij(G))2

where G(S)G(S) and G(G)G(G) are respectively the Gram matrices of the "style" image and the "generated" image, 
computed using the hidden layer activations for a particular hidden layer in the network.
'''
def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G

    Returns:
    J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
    """

    # Retrieve dimensions from a_G
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape the images to have them of shape (n_C, n_H*n_W)
    a_S = tf.reshape(a_S, [n_H * n_W, n_C])
    a_S = tf.transpose(a_S)
    a_G = tf.reshape(a_G, [n_H * n_W, n_C])
    a_G = tf.transpose(a_G)

    # Computing gram_matrices for both images S and G
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    # Computing the loss
    factor = (.5 / (n_H * n_W * n_C)) ** 2
    J_style_layer = factor * tf.reduce_sum(np.power(GS - GG, 2))

    return J_style_layer

'''
Style Weights

So far you have captured the style from only one layer. We'll get better results if we "merge" style costs from several different layers. 
After completing this exercise, feel free to come back and experiment with different weights to see how it changes the generated image GG. But for now, this is a pretty reasonable default:

STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]

You can combine the style costs for different layers as follows:

Jstyle(S,G)=∑lλ[l]J[l]style(S,G)
Jstyle(S,G)=∑lλ[l]Jstyle[l](S,G)

where the values for λ[l]λ[l] are given in STYLE_LAYERS.


    - The style of an image can be represented using the Gram matrix of a hidden layer's activations. 
        However, we get even better results combining this representation from multiple different layers. 
        This is in contrast to the content representation, where usually using just a single hidden layer is sufficient.
    - Minimizing the style cost will cause the image GG to follow the style of the image SS. 


In the inner-loop of the for-loop below, a_G is a tensor and hasn't been evaluated yet. 
It will be evaluated and updated at each iteration when we run the TensorFlow graph in model_nn() .below


The style of an image can be represented using the Gram matrix of a hidden layer's activations. 
However, we get even better results combining this representation from multiple different layers. 
This is in contrast to the content representation, where usually using just a single hidden layer is sufficient.

Minimizing the style cost will cause the image GG to follow the style of the image SS. 
'''
def compute_style_cost(model, STYLE_LAYERS):
    """
    Computes the overall style cost from several chosen layers

    Arguments:
    model -- our tensorflow model
    STYLE_LAYERS -- A python list containing:
                        - the names of the layers we would like to extract style from
                        - a coefficient for each of them

    Returns:
    J_style -- tensor representing a scalar value, style cost defined above by equation (2)
    """

    # initialize the overall style cost
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:
        # Select the output tensor of the currently selected layer
        out = model[layer_name]

        # Set a_S to be the hidden layer activation from the layer we have selected, by running the session on out
        a_S = sess.run(out)

        # Set a_G to be the hidden layer activation from same layer. Here, a_G references model[layer_name]
        # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
        # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
        a_G = out

        # Compute style_cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S, a_G)

        # Add coeff * J_style_layer of this layer to overall style cost
        J_style += coeff * J_style_layer

    return J_style


def total_cost(J_content, J_style, alpha=10, beta=40):
    """
    Computes the total cost function

    Arguments:
    J_content -- content cost coded above
    J_style -- style cost coded above
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost

    Returns:
    J -- total cost as defined by the formula above.
    """

    ### START CODE HERE ### (≈1 line)
    J = alpha * J_content + beta * J_style
    ### END CODE HERE ###

    return J


#
# model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")
#
# content_image = scipy.misc.imread("images/louvre.jpg")
# imshow(content_image)
#
# style_image = scipy.misc.imread("images/monet_800600.jpg")
# imshow(style_image)
#

# Reset the graph
tf.reset_default_graph()

# Start interactive session
sess = tf.InteractiveSession()

content_image = scipy.misc.imread("images/louvre_small.jpg")
content_image = reshape_and_normalize_image(content_image)
style_image = scipy.misc.imread("images/monet.jpg")
style_image = reshape_and_normalize_image(style_image)

# Now, we initialize the "generated" image as a noisy image created from the content_image.
# By initializing the pixels of the generated image to be mostly noise but still slightly correlated with the content image,
# this will help the content of the "generated" image more rapidly match the content of the "content" image.
generated_image = generate_noise_image(content_image)
imshow(generated_image[0])

model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")

'''
Cost Content computing:
'''

# Assign the content image to be the input of the VGG model.
# Select the output tensor of layer conv4_2
# Set a_C to be the hidden layer activation from the layer we have selected. This has been evaluated since sess.run(out) and value assigned to a_C.
# In the next step, a_G is added as part of the grapth, being the output at that layer.
sess.run(model['input'].assign(content_image))
out = model['conv4_2']
a_C = sess.run(out)
# Set a_G to be the hidden layer activation from same layer. Here, a_G references model['conv4_2']
# and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
# when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
a_G = out
# Compute the content cost
J_content = compute_content_cost(a_C, a_G)

'''
Style Content computing:
'''
# Assign the input of the model to be the "style" image
sess.run(model['input'].assign(style_image))
# Compute the style cost
J_style = compute_style_cost(model, STYLE_LAYERS)
J = total_cost(J_content, J_style, alpha = 10, beta = 40)
# define optimizer (1 line)
optimizer = tf.train.AdamOptimizer(2.0)
# define train_step (1 line)
train_step = optimizer.minimize(J)

# Right now, the grap consists a tree which top is the function tran_step -> J_total_cost ->
#           J_content: a_C value assigned already, and a_G attached at the chosen layer.
#           J_style: Inside function compute_style_cost. a_G is attached to each of the layers at the list and attached to the graph at the function compute_style_cost


def model_nn(sess, input_image, num_iterations=200):
    # Initialize global variables (you need to run the session on the initializer)
    ### START CODE HERE ### (1 line)
    sess.run(tf.global_variables_initializer())
    ### END CODE HERE ###

    # Run the noisy input image (initial generated image) through the model. Use assign().
    ### START CODE HERE ### (1 line)
    sess.run(model['input'].assign(input_image))
    ### END CODE HERE ###

    for i in range(num_iterations):

        # Run the session on the train_step to minimize the total cost
        ### START CODE HERE ### (1 line)
        somthing = sess.run(train_step)
        ### END CODE HERE ###

        # Compute the generated image by running the session on the current model['input']
        ### START CODE HERE ### (1 line)
        generated_image = sess.run(model['input'])
        ### END CODE HERE ###

        # Print every 20 iteration.
        if i % 20 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))

            # save current generated image in the "/output" directory
            save_image("output/" + str(i) + ".png", generated_image)

    # save last generated image
    save_image('output/generated_image.jpg', generated_image)

    return generated_image

model_nn(sess, generated_image)