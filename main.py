'''
This is a simple example of Convolution neural network model based on numpy, including both forward
and backward propagation.
'''

import numpy as np
import h5py
import matplotlib.pyplot as plt
import functools
from functools import partial

# Configuration for packages
plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

"""
Model architecture:
Input (image) --> [ Conv --> Relu/leaky Relu --> Pool ] --> FC --> Softmax
the layers in [] are the CNN block
"""

# Implementation that model
# Step 1. Padding

def zero_padding(X, pad):
    '''
    Pad the input array X with zeros.
    Args:
        X: ndarray of shape (m, n_H, n_W, c)
        pad: int, amount of padding around each image on vertical and horizontal dimensions

    Returns:
        X_pad -- padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
    '''
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=(0,0))
    return X_pad

def test_func_padding():
    np.random.seed(1)
    x = np.random.randn(4, 3, 3, 2)
    # x_pad = zero_padding(x, 2)
    #using partial wrapper
    pad = 2
    zeroPadding = partial(np.pad, pad_width=((0, 0), (pad, pad), (pad, pad), (0, 0)),
                          mode='constant', constant_values=(0,0))
    x_pad = zeroPadding(x)
    print("x.shape =", x.shape)
    print("x_pad.shape =", x_pad.shape)
    print("x[1,1] =", x[1, 1])
    print("x_pad[1,1] =", x_pad[1, 1])

    fig, axarr = plt.subplots(1, 2)
    axarr[0].set_title('x')
    axarr[0].imshow(x[0, :, :, 0])
    axarr[1].set_title('x_pad')
    axarr[1].imshow(x_pad[0, :, :, 0])

def conv_single_step(a_slice_prev, W, b):
    '''
    Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation
    of the previous layer.
    Args:
        a_slice_prev: slice of input data of shape (f, f, n_C_prev)
        W: Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
        b: Bias parameters contained in a window - matrix of shape (1, 1, 1)

    Returns:

    '''
    s = a_slice_prev * W # elementwise multiply
    z = np.sum(s)
    z += b
    return z

def test_conv_single_step():
    np.random.seed(1)
    a_slice_prev = np.random.randn(4, 4, 3)
    W = np.random.randn(4, 4, 3)
    b = np.random.randn(1, 1, 1)

    Z = conv_single_step(a_slice_prev, W, b)
    print("Z =", Z)

def conv_forward(A_prev, W, b, hparameters):
    '''
    Forward propagation of a conv function
    Args:
        A_prev: output of the activation function of the previous layer,
        ndarray [m, n_H, n_W, n_c_prev]
        W: Weights ndarray [f_h, f_w, n_c_prev, n_c], (f_h, f_w) is filter size
        b: bias ndarray [1, 1, 1, n_c]
        hparameters: python dictionary containing 'stride' and 'pad'

    Returns:

    '''
    m, n_H, n_W, n_c_prev = A_prev.shape
    f_H, f_W, n_c_prev, n_c = W.shape
    stride = hparameters['stride']
    pad = hparameters['pad']
    # pad the previous output
    A_prev_pad = zero_padding(A_prev, pad)

    # Calculate the dimensions for the output and create the template of the output
    n_H = (n_H + 2 * pad - f_H) // stride + 1
    n_W = (n_W + 2 * pad - f_W) // stride + 1
    Z = np.zeros((m, n_H, n_W, n_c))

    # Fill Z with conv results
    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_c):
                    # determine the sliding window location in the previous output
                    vert_start = h * stride
                    hori_start = w * stride
                    a_slice_prev = A_prev_pad[i, vert_start:vert_start+f_H, hori_start:hori_start+f_W, :]
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, W[:,:,:,c], b[:,:,:,c])

    # save info in cache for back propagation
    cache = (A_prev, W, b, hparameters)

    return Z, cache

def test_conv_forward():
    np.random.seed(1)
    A_prev = np.random.randn(10, 4, 4, 3)
    W = np.random.randn(2, 2, 3, 8)
    b = np.random.randn(1, 1, 1, 8)
    hparameters = {"pad": 2,
                   "stride": 2}

    Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
    print("Z's mean =", np.mean(Z))
    print("Z[3,2,1] =", Z[3, 2, 1])
    print("cache_conv[0][1][2][3] =", cache_conv[0][1][2][3])

def pool_forward(A_prev, hparameters, mode = 'max'):
    '''
    Implements the forward pass of the pooling layer
    Args:
        A_prev: output of the activation function of the previous layer,
        ndarray [m, n_H, n_W, n_c_prev]
        hparameters: python dictionary containing 'stride' and 'f'
        mode: pooling mode, max or average

    Returns:

    '''
    m, n_H, n_W, n_c_prev = A_prev.shape
    stride = hparameters['stride']
    f = hparameters['f']

    # calculate the dimension of output
    n_H = (n_H - f) // stride + 1
    n_W = (n_W - f) // stride + 1
    n_c = n_c_prev

    # Initialize the output
    A = np.zeros((m, n_H, n_W, n_c))

    # Fill output with pooling results
    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_c):
                    # determine the sliding window location in the previous output
                    vert_start = h * stride
                    hori_start = w * stride
                    a_slice_prev = A_prev[i, vert_start:vert_start+f, hori_start:hori_start+f, :]
                    if mode == 'max':
                        A[i, h, w, c] = np.max(a_slice_prev[:,:,c])
                    elif mode == 'average':
                        A[i, h, w, c] = np.average(a_slice_prev[:,:,c])
                    else:
                        raise 'Pooling mode "{}" is not supported!'.format(mode)

    cache = (A_prev, hparameters)
    return A, cache

def test_pool_forward():
    np.random.seed(1)
    A_prev = np.random.randn(2, 4, 4, 3)
    hparameters = {"stride": 2, "f": 3}

    A, cache = pool_forward(A_prev, hparameters)
    print("mode = max")
    print("A =", A)
    print()
    A, cache = pool_forward(A_prev, hparameters, mode="average")
    print("mode = average")
    print("A =", A)

def conv_backward(dZ, cache):
    '''
    Implement the backward propagation for a convolution function
    Args:
        dZ: gradient of the cost with respect to the output of the conv layer (Z), numpy array of shape (m, n_H, n_W, n_C)
        cache: cache of values needed for the conv_backward(), output of conv_forward()

    Returns:
    dA_prev: gradient of the cost with respect to the input of the conv layer (A_prev),
               numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    dW: gradient of the cost with respect to the weights of the conv layer (W)
          numpy array of shape (f, f, n_C_prev, n_C)
    db: gradient of the cost with respect to the biases of the conv layer (b)
          numpy array of shape (1, 1, 1, n_C)
    '''
    # retrive parameters
    # Retrieve information from "cache"
    (A_prev, W, b, hparameters) = cache

    # Retrieve dimensions from A_prev's shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # Retrieve dimensions from W's shape
    (f_H, f_W, n_C_prev, n_C) = W.shape

    # Retrieve information from "hparameters"
    stride = hparameters['stride']
    pad = hparameters['pad']

    # Retrieve dimensions from dZ's shape
    (m, n_H, n_W, n_C) = dZ.shape

    #initilize outputs
    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
    dW = np.zeros(W.shape)
    db = np.zeros((1, 1, 1, n_C))

    #pad A_prev and dA_prev
    A_prev_pad = zero_padding(A_prev, pad)
    dA_prev_pad = zero_padding(dA_prev, pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i,:,:,:]
        da_prev_pad = dA_prev_pad[i,:,:,:]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h * stride
                    hori_start = w * stride

                    a_slice = a_prev_pad[vert_start:vert_start+f_H, hori_start:hori_start+f_W, :]

                    da_prev_pad[vert_start:vert_start+f_H, hori_start:hori_start+f_W, :] += W[:,:,:,c] * dZ[i, h, w, c]
                    dW[:,:,:,c] += a_slice * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]
        dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]

    return dA_prev, dW, dZ

def test_conv_backward():
    np.random.seed(1)

    A_prev = np.random.randn(10, 4, 4, 3)
    W = np.random.randn(2, 2, 3, 8)
    b = np.random.randn(1, 1, 1, 8)
    hparameters = {"pad": 2, "stride": 2}

    np.random.seed(1)
    Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
    print("Z's mean =", np.mean(Z))
    print("Z[3,2,1] =", Z[3, 2, 1])
    print("cache_conv[0][1][2][3] =", cache_conv[0][1][2][3])

    dA, dW, db = conv_backward(Z, cache_conv)
    print("dA_mean =", np.mean(dA))
    print("dW_mean =", np.mean(dW))
    print("db_mean =", np.mean(db))

## Pooling layer backward
def create_mask_from_window(x):
    '''
    Creates a mask from an input matrix x, to identify the max entry of x.
    Args:
        x: Array of shape (f_H, f_W)

    Returns:

    '''
    return x == np.max(x)


def distribute_value(dz, shape):
    """
    Distributes the input value in the matrix of dimension shape

    Arguments:
    dz -- input scalar
    shape -- the shape (n_H, n_W) of the output matrix for which we want to distribute the value of dz

    Returns:
    a -- Array of size (n_H, n_W) for which we distributed the value of dz
    """
    n_H, n_W = shape
    average = dz * 1.0 / n_H / n_W
    return average * np.ones(shape)


def pool_backward(dA, cache, mode="max"):
    """
    Implements the backward pass of the pooling layer

    Arguments:
    dA -- gradient of cost with respect to the output of the pooling layer, same shape as A
    cache -- cache output from the forward pass of the pooling layer, contains the layer's input and hparameters
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")

    Returns:
    dA_prev -- gradient of cost with respect to the input of the pooling layer, same shape as A_prev
    """
    A_prev, hparameters = cache
    stride = hparameters['stride']
    f = hparameters['f']
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    m, n_H, n_W, n_C = dA.shape

    # initialize output
    dA_prev = np.zeros(A_prev.shape)

    for i in range(m):
        a_prev = A_prev[i,:,:,:]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h * stride
                    hori_start = w * stride

                    if mode == 'max':
                        a_prev_slice = a_prev[vert_start:vert_start+f, hori_start:hori_start+f, c]
                        mask = create_mask_from_window(a_prev_slice)
                        dA_prev[i, vert_start: vert_start+f, hori_start: hori_start+f, c] += mask * dA[i, h, w, c]
                    elif mode == "average":
                        da = dA[i, h, w, c]
                        shape = (f, f)
                        dA_prev[i, vert_start: vert_start + f, hori_start: hori_start + f, c] += distribute_value(da, shape)

    return dA_prev

def test_pool_backward():
    np.random.seed(1)
    A_prev = np.random.randn(5, 5, 3, 2)
    hparameters = {"stride": 1, "f": 2}
    A, cache = pool_forward(A_prev, hparameters)
    dA = np.random.randn(5, 4, 2, 2)

    dA_prev = pool_backward(dA, cache, mode="max")
    print("mode = max")
    print('mean of dA = ', np.mean(dA))
    print('dA_prev[1,1] = ', dA_prev[1, 1])
    print()
    dA_prev = pool_backward(dA, cache, mode="average")
    print("mode = average")
    print('mean of dA = ', np.mean(dA))
    print('dA_prev[1,1] = ', dA_prev[1, 1])

if __name__ == '__main__':
    test_pool_backward()