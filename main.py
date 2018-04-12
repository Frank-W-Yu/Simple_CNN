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

if __name__ == '__main__':
    test_conv_single_step()