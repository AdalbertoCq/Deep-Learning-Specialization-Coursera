import numpy as np
import math


def basic_sigmoid(x):
    """
    Compute sigmoid of x.

    Arguments:
    x -- A scalar

    Return:
    s -- sigmoid(x)
    """

    ### START CODE HERE ### (≈ 1 line of code)
    s = 1/(1+np.exp(-x))
    ### END CODE HERE ###

    return s


def sigmoid_derivative(x):
    """
    Compute the gradient (also called the slope or derivative) of the sigmoid function with respect to its input x.
    You can store the output of the sigmoid function into variables and then use it to calculate the gradient.

    Arguments:
    x -- A scalar or numpy array

    Return:
    ds -- Your computed gradient.
    """

    ### START CODE HERE ### (≈ 2 lines of code)
    s = 1 / (1 + np.exp(-x))
    one = np.ones(s.shape)
    ds = np.multiply(s,(one-s))
    ### END CODE HERE ###

    return ds


def image2vector(image):
    """
    Argument:
    image -- a numpy array of shape (length, height, depth)

    Returns:
    v -- a vector of shape (length*height*depth, 1)
    """

    ### START CODE HERE ### (≈ 1 line of code)
    new_shape = (image.shape[0]*image.shape[1]*image.shape[2],1)
    v = image.reshape(new_shape)
    ### END CODE HERE ###

    return v


def normalizeRows(x):
    """
    Implement a function that normalizes each row of the matrix x (to have unit length).

    Argument:
    x -- A numpy matrix of shape (n, m)

    Returns:
    x -- The normalized (by row) numpy matrix. You are allowed to modify x.
    """

    ### START CODE HERE ### (≈ 2 lines of code)
    # Compute x_norm as the norm 2 of x. Use np.linalg.norm(..., ord = 2, axis = ..., keepdims = True)
    x_norm = np.linalg.norm(x,ord=2,axis=1,keepdims=True)
    print(x_norm)
    # Divide x by its norm.
    x = np.divide(x,x_norm)
    ### END CODE HERE ###

    return x


def softmax(x):
    """Calculates the softmax for each row of the input x.

    Your code should work for a row vector and also for matrices of shape (n, m).

    Argument:
    x -- A numpy matrix of shape (n,m)

    Returns:
    s -- A numpy matrix equal to the softmax of x, of shape (n,m)
    """

    ### START CODE HERE ### (≈ 3 lines of code)
    # Apply exp() element-wise to x. Use np.exp(...).
    x_exp = np.exp(x)

    # Create a vector x_sum that sums each row of x_exp. Use np.sum(..., axis = 1, keepdims = True).
    x_sum = np.sum(x_exp, axis=1, keepdims=True)

    # Compute softmax(x) by dividing x_exp by x_sum. It should automatically use numpy broadcasting.
    s = np.divide(x_exp,x_sum)

    ### END CODE HERE ###

    return s


def L1(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)

    Returns:
    loss -- the value of the L1 loss function defined above
    """

    ### START CODE HERE ### (≈ 1 line of code)
    loss = np.sum(np.abs(y - yhat), axis=0, keepdims=True)
    ### END CODE HERE ###

    return loss


def L2(yhat, y):
        """
        Arguments:
        yhat -- vector of size m (predicted labels)
        y -- vector of size m (true labels)

        Returns:
        loss -- the value of the L2 loss function defined above
        """

        ### START CODE HERE ### (≈ 1 line of code)
        loss = np.sum(np.multiply(y - yhat), y - yhat, axis=0, keepdims=True)
        ### END CODE HERE ###

        return loss


def main():
    x = np.array(range(-100, 101, 1))
    y = basic_sigmoid(x)
    dy = sigmoid_derivative(x)

    x = np.array([
        [9, 2, 5, 0, 0],
        [7, 5, 0, 0, 0]])
    print(softmax(x))


if __name__ == "__main__":
    main()