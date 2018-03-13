import argparse
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
sys.path.append('/Users/aclaudioquiros/Documents/PycCharm/Coursera/Convolutional Neural Networks/week3/')
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body


'''
Model details

First things to know:

    The input is a batch of images of shape (m, 608, 608, 3)
    The output is a list of bounding boxes along with the recognized classes. Each bounding box is represented by 6 numbers (pc,bx,by,bh,bw,c)(pc,bx,by,bh,bw,c) as explained above. If you expand cc into an 80-dimensional vector, each bounding box is then represented by 85 numbers.

We will use 5 anchor boxes. So you can think of the YOLO architecture as the following: IMAGE (m, 608, 608, 3) -> DEEP CNN -> ENCODING (m, 19, 19, 5, 85).



Filtering with a threshold on class scores

You are going to apply a first filter by thresholding. You would like to get rid of any box for which the class "score" is less than a chosen threshold.

The model gives you a total of 19x19x5x85 numbers, with each box described by 85 numbers. It'll be convenient to rearrange the (19,19,5,85) (or (19,19,425)) dimensional tensor into the following variables:

    box_confidence: tensor of shape (19×19,5,1)(19×19,5,1) containing pcpc (confidence probability that there's some object) for each of the 5 boxes predicted in each of the 19x19 cells.
    boxes: tensor of shape (19×19,5,4)(19×19,5,4) containing (bx,by,bh,bw)(bx,by,bh,bw) for each of the 5 boxes per cell.
    box_class_probs: tensor of shape (19×19,5,80)(19×19,5,80) containing the detection probabilities (c1,c2,...c80)(c1,c2,...c80) for each of the 80 classes for each of the 5 boxes per cell.

'''

'''
Exercise: Implement iou(). Some hints:

    In this exercise only, we define a box using its two corners (upper left and lower right): (x1, y1, x2, y2) rather than the midpoint and height/width.
    To calculate the area of a rectangle you need to multiply its height (y2 - y1) by its width (x2 - x1)
    You'll also need to find the coordinates (xi1, yi1, xi2, yi2) of the intersection of two boxes. Remember that:
        xi1 = maximum of the x1 coordinates of the two boxes
        yi1 = maximum of the y1 coordinates of the two boxes
        xi2 = minimum of the x2 coordinates of the two boxes
        yi2 = minimum of the y2 coordinates of the two boxes

'''
def iou(box1, box2):
    """Implement the intersection over union (IoU) between box1 and box2

    Arguments:
    box1 -- first box, list object with coordinates (x1, y1, x2, y2)
    box2 -- second box, list object with coordinates (x1, y1, x2, y2)
    """

    # Calculate the (y1, x1, y2, x2) coordinates of the intersection of box1 and box2. Calculate its Area.
    xi1 = np.maximum(box1[0], box2[0])
    yi1 = np.maximum(box1[1], box2[1])
    xi2 = np.minimum(box1[2], box2[2])
    yi2 = np.minimum(box1[3], box2[3])
    # Case in which they don't intersec --> max(,0)
    inter_area = max(xi2-xi1, 0)*max(yi2-yi1, 0)

    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = (box1[2]-box1[0])*(box1[3]-box1[1])
    box2_area = (box2[2]-box2[0])*(box2[3]-box2[1])
    union_area = box1_area + box2_area - inter_area

    # compute the IoU
    iou = flaot(inter_area)/float(union_area)

    return iou


'''
Exercise: Implement yolo_filter_boxes().

    Compute box scores by doing the elementwise product as described in Figure 4. The following code may help you choose the right operator:

    a = np.random.randn(19*19, 5, 1)
    b = np.random.randn(19*19, 5, 80)
    c = a * b # shape of c will be (19*19, 5, 80)

    For each box, find:
        the index of the class with the maximum box score  (Be careful with what axis you choose; consider using axis=-1)
        the corresponding box score  (Be careful with what axis you choose; consider using axis=-1)
    Create a mask by using a threshold. As a reminder: ([0.9, 0.3, 0.4, 0.5, 0.1] < 0.4) returns: [False, True, False, False, True]. The mask should be True for the boxes you want to keep.
    Use TensorFlow to apply the mask to box_class_scores, boxes and box_classes to filter out the boxes we don't want. You should be left with just the subset of boxes you want to keep. 

'''
def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=.6):
    """Filters YOLO boxes by thresholding on object and class confidence.

    Arguments:
    box_confidence -- tensor of shape (19, 19, 5, 1)
    boxes -- tensor of shape (19, 19, 5, 4)
    box_class_probs -- tensor of shape (19, 19, 5, 80)
    threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box

    Returns:
    scores -- tensor of shape (None,), containing the class probability score for selected boxes
    boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
    classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes

    Note: "None" is here because you don't know the exact number of selected boxes, as it depends on the threshold.
    For example, the actual output size of scores would be (10,) if there are 10 boxes.
    """

    # Step 1: Compute box scores
    # Element wise product for the last dimension. Multiplying the probabilities by the classes.
    box_scores = box_confidence * box_class_probs

    # Step 2: Find the box_classes thanks to the max box_scores, keep track of the corresponding score
    # For each of the bounding boxes, index of the maximum value of the probability vector.
    # Dim (19, 19, 5)
    box_classes = K.argmax(box_scores, axis=-1)
    # Maximum probability fro each of the bounding boxes
    # Dim (19, 19, 5) The max value of each bounding box pc*class
    # Dim (19, 19, 5, 1) if keep_dims=True. This keeps the vector for each bounding box with only the max value.
    box_class_scores = K.max(box_scores, axis=-1)

    # Step 3: Create a filtering mask based on "box_class_scores" by using "threshold". The mask should have the
    # same dimension as box_class_scores, and be True for the boxes you want to keep (with probability >= threshold)
    filtering_mask = box_class_scores > threshold

    # Step 4: Apply the mask to scores, boxes and classes
    # Returns scores of the boxes to keep, each with the highest prob for one class.
    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    # Boxes to keep according to the given Prob threshold.
    boxes = tf.boolean_mask(box_scores, filtering_mask)
    # Dominant class for the boxes that we keep.
    classes = tf.boolean_mask(box_classes, filtering_mask)

    return scores, boxes, classes


'''
You are now ready to implement non-max suppression. The key steps are:
    Select the box that has the highest score.
    Compute its overlap with all other boxes, and remove boxes that overlap it more than iou_threshold.
    Go back to step 1 and iterate until there's no more boxes with a lower score than the current selected box.

This will remove all boxes that have a large overlap with the selected boxes. Only the "best" boxes remain.

Exercise: Implement yolo_non_max_suppression() using TensorFlow. TensorFlow has two built-in functions that are used to implement non-max suppression (so you don't actually need to use your iou() implementation):
    tf.image.non_max_suppression()
    K.gather()

'''
def yolo_non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.5):
    """
    Applies Non-max suppression (NMS) to set of boxes

    Arguments:
    scores -- tensor of shape (None,), output of yolo_filter_boxes()
    boxes -- tensor of shape (None, 4), output of yolo_filter_boxes() that have been scaled to the image size (see later)
    classes -- tensor of shape (None,), output of yolo_filter_boxes()
    max_boxes -- integer, maximum number of predicted boxes you'd like
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering

    Returns:
    scores -- tensor of shape (, None), predicted score for each box
    boxes -- tensor of shape (4, None), predicted box coordinates
    classes -- tensor of shape (, None), predicted class for each box

    Note: The "None" dimension of the output tensors has obviously to be less than max_boxes. Note also that this
    function will transpose the shapes of scores, boxes, classes. This is made for convenience.
    """

    max_boxes_tensor = K.variable(max_boxes, dtype='int32')  # tensor to be used in tf.image.non_max_suppression()
    K.get_session().run(tf.variables_initializer([max_boxes_tensor]))  # initialize variable max_boxes_tensor

    # Use tf.image.non_max_suppression() to get the list of indices corresponding to boxes you keep
    # Given the boxes kept and it's scores, it will iterate to leave the one with the highest prob.
    nms_indices = tf.image.non_max_suppression(boxes, scores, max_output_size=max_boxes, iou_threshold=iou_threshold)

    # Use K.gather() to select only nms_indices from scores, boxes and classes
    # Using the index for the boxes left, get the scores, box, and classes for each.
    scores = K.gather(scores, nms_indices)
    boxes = K.gather(boxes, nms_indices)
    classes = K.gather(classes, nms_indices)

    return scores, boxes, classes

'''
It's time to implement a function taking the output of the deep CNN (the 19x19x5x85 dimensional encoding) and filtering through all the boxes using the functions you've just implemented.

Exercise: Implement yolo_eval() which takes the output of the YOLO encoding and filters the boxes using score threshold and NMS. There's just one last implementational detail you have to know. 
There're a few ways of representing boxes, such as via their corners or via their midpoint and height/width. YOLO converts between a few such formats at different times, using the following functions (which we have provided):
    boxes = yolo_boxes_to_corners(box_xy, box_wh)
which converts the yolo box coordinates (x,y,w,h) to box corners' coordinates (x1, y1, x2, y2) to fit the input of yolo_filter_boxes
    boxes = scale_boxes(boxes, image_shape)
    
YOLO's network was trained to run on 608x608 images. If you are testing this data on a different size image--for example, the car detection dataset had 720x1280 images--this step rescales the boxes so that they can be plotted on top of the original 720x1280 image.
'''
def yolo_eval(yolo_outputs, image_shape=(720., 1280.), max_boxes=10, score_threshold=.6, iou_threshold=.5):
    """
    Converts the output of YOLO encoding (a lot of boxes) to your predicted boxes along with their scores, box coordinates and classes.

    Arguments:
    yolo_outputs -- output of the encoding model (for image_shape of (608, 608, 3)), contains 4 tensors:
                    box_confidence: tensor of shape (None, 19, 19, 5, 1)
                    box_xy: tensor of shape (None, 19, 19, 5, 2)
                    box_wh: tensor of shape (None, 19, 19, 5, 2)
                    box_class_probs: tensor of shape (None, 19, 19, 5, 80)
    image_shape -- tensor of shape (2,) containing the input shape, in this notebook we use (608., 608.) (has to be float32 dtype)
    max_boxes -- integer, maximum number of predicted boxes you'd like
    score_threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering

    Returns:
    scores -- tensor of shape (None, ), predicted score for each box
    boxes -- tensor of shape (None, 4), predicted box coordinates
    classes -- tensor of shape (None,), predicted class for each box
    """

    # Retrieve outputs of the YOLO model
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs

    # Convert boxes to be ready for filtering functions
    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    # Use one of the functions you've implemented to perform Score-filtering with a threshold of score_threshold
    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, score_threshold)

    # Scale boxes back to original image shape.
    boxes = scale_boxes(boxes, image_shape)

    # Use one of the functions you've implemented to perform Non-max suppression with a threshold of iou_threshold
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)

    return scores, boxes, classes

'''
 Summary for YOLO:

    Input image (608, 608, 3)
    The input image goes through a CNN, resulting in a (19,19,5,85) dimensional output.
    After flattening the last two dimensions, the output is a volume of shape (19, 19, 425):
        Each cell in a 19x19 grid over the input image gives 425 numbers.
        425 = 5 x 85 because each cell contains predictions for 5 boxes, corresponding to 5 anchor boxes, as seen in lecture.
        85 = 5 + 80 where 5 is because (pc,bx,by,bh,bw)(pc,bx,by,bh,bw) has 5 numbers, and and 80 is the number of classes we'd like to detect
    You then select only few boxes based on:
        Score-thresholding: throw away boxes that have detected a class with a score less than the threshold
        Non-max suppression: Compute the Intersection over Union and avoid selecting overlapping boxes
    This gives you YOLO's final output.
'''
# Creating a session to start your graph.
sess = K.get_session()
# Recall that we are trying to detect 80 classes, and are using 5 anchor boxes.
# We have gathered the information about the 80 classes and 5 boxes in two files "coco_classes.txt" and "yolo_anchors.txt".
# Let's load these quantities into the model by running the next cell.
# The car detection dataset has 720x1280 images, which we've pre-processed into 608x608 images.
class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")
image_shape = (720., 1280.)

# Training a YOLO model takes a very long time and requires a fairly large dataset of labelled bounding boxes for a large range of target classes.
# You are going to load an existing pretrained Keras YOLO model stored in "yolo.h5".
# (These weights come from the official YOLO website, and were converted using a function written by Allan Zelener.
# Technically, these are the parameters from the "YOLOv2" model, but we will more simply refer to it as "YOLO" in this notebook.)
yolo_model = load_model("model_data/yolo.h5")
yolo_model.summary()

# The output of yolo_model is a (m, 19, 19, 5, 85) tensor that needs to pass through non-trivial processing and conversion.
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
# yolo_outputs gave you all the predicted boxes of yolo_model in the correct format. You're now ready to perform filtering and select only the best boxes.
scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)

'''
    yolo_model.input is given to yolo_model. The model is used to compute the output yolo_model.output
    yolo_model.output is processed by yolo_head. It gives you yolo_outputs
    yolo_outputs goes through a filtering function, yolo_eval. It outputs your predictions: scores, boxes, classes
'''


def predict(sess, image_file):
    """
    Runs the graph stored in "sess" to predict boxes for "image_file". Prints and plots the preditions.

    Arguments:
    sess -- your tensorflow/Keras session containing the YOLO graph
    image_file -- name of an image stored in the "images" folder.

    Returns:
    out_scores -- tensor of shape (None, ), scores of the predicted boxes
    out_boxes -- tensor of shape (None, 4), coordinates of the predicted boxes
    out_classes -- tensor of shape (None, ), class index of the predicted boxes

    Note: "None" actually represents the number of predicted boxes, it varies between 0 and max_boxes.
    """

    # Preprocess your image
    image, image_data = preprocess_image("images/" + image_file, model_image_size=(608, 608))

    # Run the session with the correct tensors and choose the correct placeholders in the feed_dict.
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes], feed_dict={yolo_model.input: image_data, K.learning_phase(): 0})

    # Print predictions info
    print('Found {} boxes for {}'.format(len(out_boxes), image_file))
    # Generate colors for drawing bounding boxes.
    colors = generate_colors(class_names)
    # Draw bounding boxes on the image file
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    # Save the predicted bounding box on the image
    image.save(os.path.join("out", image_file), quality=90)
    # Display the results in the notebook
    output_image = scipy.misc.imread(os.path.join("out", image_file))
    imshow(output_image)

    return out_scores, out_boxes, out_classes

out_scores, out_boxes, out_classes = predict(sess, "test.jpg")

'''
    YOLO is a state-of-the-art object detection model that is fast and accurate
    It runs an input image through a CNN which outputs a 19x19x5x85 dimensional volume.
    The encoding can be seen as a grid where each of the 19x19 cells contains information about 5 boxes.
    You filter through all the boxes using non-max suppression. Specifically:
        Score thresholding on the probability of detecting a class to keep only accurate (high probability) boxes
        Intersection over Union (IoU) thresholding to eliminate overlapping boxes
    Because training a YOLO model from randomly initialized weights is non-trivial and requires a large dataset as well as lot of computation, we used previously trained model parameters in this exercise. 
    If you wish, you can also try fine-tuning the YOLO model with your own dataset, though this would be a fairly non-trivial exercise.


Summary of CNN that was trained to be used with YOLO
Input : (608, 608,   3) --> 1,108,992
Output: (19,   19, 405) -->   146,205
Total layers: 70 layers
        Total params: 50,983,561
            Trainable params: 50,962,889
            Non-trainable params: 20,672
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
input_1 (InputLayer)             (None, 608, 608, 3)   0                                            
____________________________________________________________________________________________________
conv2d_1 (Conv2D)                (None, 608, 608, 32)  864         input_1[0][0]                    
____________________________________________________________________________________________________
batch_normalization_1 (BatchNorm (None, 608, 608, 32)  128         conv2d_1[0][0]                   
____________________________________________________________________________________________________
leaky_re_lu_1 (LeakyReLU)        (None, 608, 608, 32)  0           batch_normalization_1[0][0]      
____________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)   (None, 304, 304, 32)  0           leaky_re_lu_1[0][0]              
____________________________________________________________________________________________________
conv2d_2 (Conv2D)                (None, 304, 304, 64)  18432       max_pooling2d_1[0][0]            
____________________________________________________________________________________________________
batch_normalization_2 (BatchNorm (None, 304, 304, 64)  256         conv2d_2[0][0]                   
____________________________________________________________________________________________________
leaky_re_lu_2 (LeakyReLU)        (None, 304, 304, 64)  0           batch_normalization_2[0][0]      
____________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)   (None, 152, 152, 64)  0           leaky_re_lu_2[0][0]              
____________________________________________________________________________________________________
conv2d_3 (Conv2D)                (None, 152, 152, 128) 73728       max_pooling2d_2[0][0]            
____________________________________________________________________________________________________
batch_normalization_3 (BatchNorm (None, 152, 152, 128) 512         conv2d_3[0][0]                   
____________________________________________________________________________________________________
leaky_re_lu_3 (LeakyReLU)        (None, 152, 152, 128) 0           batch_normalization_3[0][0]      
____________________________________________________________________________________________________
conv2d_4 (Conv2D)                (None, 152, 152, 64)  8192        leaky_re_lu_3[0][0]              
____________________________________________________________________________________________________
batch_normalization_4 (BatchNorm (None, 152, 152, 64)  256         conv2d_4[0][0]                   
____________________________________________________________________________________________________
leaky_re_lu_4 (LeakyReLU)        (None, 152, 152, 64)  0           batch_normalization_4[0][0]      
____________________________________________________________________________________________________
conv2d_5 (Conv2D)                (None, 152, 152, 128) 73728       leaky_re_lu_4[0][0]              
____________________________________________________________________________________________________
batch_normalization_5 (BatchNorm (None, 152, 152, 128) 512         conv2d_5[0][0]                   
____________________________________________________________________________________________________
leaky_re_lu_5 (LeakyReLU)        (None, 152, 152, 128) 0           batch_normalization_5[0][0]      
____________________________________________________________________________________________________
max_pooling2d_3 (MaxPooling2D)   (None, 76, 76, 128)   0           leaky_re_lu_5[0][0]              
____________________________________________________________________________________________________
conv2d_6 (Conv2D)                (None, 76, 76, 256)   294912      max_pooling2d_3[0][0]            
____________________________________________________________________________________________________
batch_normalization_6 (BatchNorm (None, 76, 76, 256)   1024        conv2d_6[0][0]                   
____________________________________________________________________________________________________
leaky_re_lu_6 (LeakyReLU)        (None, 76, 76, 256)   0           batch_normalization_6[0][0]      
____________________________________________________________________________________________________
conv2d_7 (Conv2D)                (None, 76, 76, 128)   32768       leaky_re_lu_6[0][0]              
____________________________________________________________________________________________________
batch_normalization_7 (BatchNorm (None, 76, 76, 128)   512         conv2d_7[0][0]                   
____________________________________________________________________________________________________
leaky_re_lu_7 (LeakyReLU)        (None, 76, 76, 128)   0           batch_normalization_7[0][0]      
____________________________________________________________________________________________________
conv2d_8 (Conv2D)                (None, 76, 76, 256)   294912      leaky_re_lu_7[0][0]              
____________________________________________________________________________________________________
batch_normalization_8 (BatchNorm (None, 76, 76, 256)   1024        conv2d_8[0][0]                   
____________________________________________________________________________________________________
leaky_re_lu_8 (LeakyReLU)        (None, 76, 76, 256)   0           batch_normalization_8[0][0]      
____________________________________________________________________________________________________
max_pooling2d_4 (MaxPooling2D)   (None, 38, 38, 256)   0           leaky_re_lu_8[0][0]              
____________________________________________________________________________________________________
conv2d_9 (Conv2D)                (None, 38, 38, 512)   1179648     max_pooling2d_4[0][0]            
____________________________________________________________________________________________________
batch_normalization_9 (BatchNorm (None, 38, 38, 512)   2048        conv2d_9[0][0]                   
____________________________________________________________________________________________________
leaky_re_lu_9 (LeakyReLU)        (None, 38, 38, 512)   0           batch_normalization_9[0][0]      
____________________________________________________________________________________________________
conv2d_10 (Conv2D)               (None, 38, 38, 256)   131072      leaky_re_lu_9[0][0]              
____________________________________________________________________________________________________
batch_normalization_10 (BatchNor (None, 38, 38, 256)   1024        conv2d_10[0][0]                  
____________________________________________________________________________________________________
leaky_re_lu_10 (LeakyReLU)       (None, 38, 38, 256)   0           batch_normalization_10[0][0]     
____________________________________________________________________________________________________
conv2d_11 (Conv2D)               (None, 38, 38, 512)   1179648     leaky_re_lu_10[0][0]             
____________________________________________________________________________________________________
batch_normalization_11 (BatchNor (None, 38, 38, 512)   2048        conv2d_11[0][0]                  
____________________________________________________________________________________________________
leaky_re_lu_11 (LeakyReLU)       (None, 38, 38, 512)   0           batch_normalization_11[0][0]     
____________________________________________________________________________________________________
conv2d_12 (Conv2D)               (None, 38, 38, 256)   131072      leaky_re_lu_11[0][0]             
____________________________________________________________________________________________________
batch_normalization_12 (BatchNor (None, 38, 38, 256)   1024        conv2d_12[0][0]                  
____________________________________________________________________________________________________
leaky_re_lu_12 (LeakyReLU)       (None, 38, 38, 256)   0           batch_normalization_12[0][0]     
____________________________________________________________________________________________________
conv2d_13 (Conv2D)               (None, 38, 38, 512)   1179648     leaky_re_lu_12[0][0]             
____________________________________________________________________________________________________
batch_normalization_13 (BatchNor (None, 38, 38, 512)   2048        conv2d_13[0][0]                  
____________________________________________________________________________________________________
leaky_re_lu_13 (LeakyReLU)       (None, 38, 38, 512)   0           batch_normalization_13[0][0]     
____________________________________________________________________________________________________
max_pooling2d_5 (MaxPooling2D)   (None, 19, 19, 512)   0           leaky_re_lu_13[0][0]             
____________________________________________________________________________________________________
conv2d_14 (Conv2D)               (None, 19, 19, 1024)  4718592     max_pooling2d_5[0][0]            
____________________________________________________________________________________________________
batch_normalization_14 (BatchNor (None, 19, 19, 1024)  4096        conv2d_14[0][0]                  
____________________________________________________________________________________________________
leaky_re_lu_14 (LeakyReLU)       (None, 19, 19, 1024)  0           batch_normalization_14[0][0]     
____________________________________________________________________________________________________
conv2d_15 (Conv2D)               (None, 19, 19, 512)   524288      leaky_re_lu_14[0][0]             
____________________________________________________________________________________________________
batch_normalization_15 (BatchNor (None, 19, 19, 512)   2048        conv2d_15[0][0]                  
____________________________________________________________________________________________________
leaky_re_lu_15 (LeakyReLU)       (None, 19, 19, 512)   0           batch_normalization_15[0][0]     
____________________________________________________________________________________________________
conv2d_16 (Conv2D)               (None, 19, 19, 1024)  4718592     leaky_re_lu_15[0][0]             
____________________________________________________________________________________________________
batch_normalization_16 (BatchNor (None, 19, 19, 1024)  4096        conv2d_16[0][0]                  
____________________________________________________________________________________________________
leaky_re_lu_16 (LeakyReLU)       (None, 19, 19, 1024)  0           batch_normalization_16[0][0]     
____________________________________________________________________________________________________
conv2d_17 (Conv2D)               (None, 19, 19, 512)   524288      leaky_re_lu_16[0][0]             
____________________________________________________________________________________________________
batch_normalization_17 (BatchNor (None, 19, 19, 512)   2048        conv2d_17[0][0]                  
____________________________________________________________________________________________________
leaky_re_lu_17 (LeakyReLU)       (None, 19, 19, 512)   0           batch_normalization_17[0][0]     
____________________________________________________________________________________________________
conv2d_18 (Conv2D)               (None, 19, 19, 1024)  4718592     leaky_re_lu_17[0][0]             
____________________________________________________________________________________________________
batch_normalization_18 (BatchNor (None, 19, 19, 1024)  4096        conv2d_18[0][0]                  
____________________________________________________________________________________________________
leaky_re_lu_18 (LeakyReLU)       (None, 19, 19, 1024)  0           batch_normalization_18[0][0]     
____________________________________________________________________________________________________
conv2d_19 (Conv2D)               (None, 19, 19, 1024)  9437184     leaky_re_lu_18[0][0]             
____________________________________________________________________________________________________
batch_normalization_19 (BatchNor (None, 19, 19, 1024)  4096        conv2d_19[0][0]                  
____________________________________________________________________________________________________
conv2d_21 (Conv2D)               (None, 38, 38, 64)    32768       leaky_re_lu_13[0][0]             
____________________________________________________________________________________________________
leaky_re_lu_19 (LeakyReLU)       (None, 19, 19, 1024)  0           batch_normalization_19[0][0]     
____________________________________________________________________________________________________
batch_normalization_21 (BatchNor (None, 38, 38, 64)    256         conv2d_21[0][0]                  
____________________________________________________________________________________________________
conv2d_20 (Conv2D)               (None, 19, 19, 1024)  9437184     leaky_re_lu_19[0][0]             
____________________________________________________________________________________________________
leaky_re_lu_21 (LeakyReLU)       (None, 38, 38, 64)    0           batch_normalization_21[0][0]     
____________________________________________________________________________________________________
batch_normalization_20 (BatchNor (None, 19, 19, 1024)  4096        conv2d_20[0][0]                  
____________________________________________________________________________________________________
space_to_depth_x2 (Lambda)       (None, 19, 19, 256)   0           leaky_re_lu_21[0][0]             
____________________________________________________________________________________________________
leaky_re_lu_20 (LeakyReLU)       (None, 19, 19, 1024)  0           batch_normalization_20[0][0]     
____________________________________________________________________________________________________
concatenate_1 (Concatenate)      (None, 19, 19, 1280)  0           space_to_depth_x2[0][0]          
                                                                   leaky_re_lu_20[0][0]             
____________________________________________________________________________________________________
conv2d_22 (Conv2D)               (None, 19, 19, 1024)  11796480    concatenate_1[0][0]              
____________________________________________________________________________________________________
batch_normalization_22 (BatchNor (None, 19, 19, 1024)  4096        conv2d_22[0][0]                  
____________________________________________________________________________________________________
leaky_re_lu_22 (LeakyReLU)       (None, 19, 19, 1024)  0           batch_normalization_22[0][0]     
____________________________________________________________________________________________________
conv2d_23 (Conv2D)               (None, 19, 19, 425)   435625      leaky_re_lu_22[0][0]             
====================================================================================================
Total params: 50,983,561
Trainable params: 50,962,889
Non-trainable params: 20,672
'''