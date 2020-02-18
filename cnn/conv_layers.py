"""Implementation of convolution forward and backward pass"""

import numpy as np


def conv_layer_forward(input_layer, weight, bias, pad_size=1, stride=1):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of M data points, each with C channels, height H and
    width W. We convolve each input with C_o different filters, where each filter
    spans all C_i channels and has height H_w and width W_w.

    Args:
        input_alyer: The input layer with shape (batch_size, channels_x, height_x, width_x)
        weight: Filter kernels with shape (num_filters, channels_x, height_w, width_w)
        bias: Biases of shape (num_filters)

    Returns:
        output_layer: The output layer with shape (batch_size, num_filters, height_y, width_y)
    """
    (batch_size, channels_x, height_x, width_x) = input_layer.shape
    (num_filters, channels_w, height_w, width_w) = weight.shape

    height_y = int(1 + (height_x + (2*pad_size) - height_w)/stride)
    width_y = int(1 + (width_x + (2*pad_size) - width_w)/stride)
    output_layer = np.zeros((batch_size, num_filters, height_y, width_y))

    padded_input = np.pad(
        input_layer, ((0, 0), (0, 0), (pad_size, pad_size), (pad_size, pad_size)),  'constant')

    assert channels_w == channels_x, (
        "The number of filter channels be the same as the number of input layer channels")

    for i in range(batch_size):
        for j in range(num_filters):
            for k in range(channels_x):

                input_p = 0
                for p in range(height_y):
                    input_q = 0
                    for q in range(width_y):
                        output_layer[i, j, p, q] += np.sum(
                            padded_input[i, k, input_p: input_p + height_w, input_q: input_q + width_w] * weight[j, k])

                        input_q += stride
                    input_p += stride

            output_layer[i, j, :, :] += bias[j]

    return output_layer


def conv_layer_backward(output_layer_gradient, input_layer, weight, bias, pad_size=1):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Args:
        output_layer_gradient: Gradient of the loss L wrt the next layer y, with shape
            (batch_size, num_filters, height_y, width_y)
        input_layer: Input layer x with shape (batch_size, channels_x, height_x, width_x)
        weight: Filter kernels with shape (num_filters, channels_x, height_w, width_w)
        bias: Biases of shape (num_filters)

    Returns:
        input_layer_gradient: Gradient of the loss L with respect to the input layer x
        weight_gradient: Gradient of the loss L with respect to the filters w
        bias_gradient: Gradient of the loss L with respect to the biases b
    """
    batch_size, channels_y, height_y, width_y = output_layer_gradient.shape
    batch_size, channels_x, height_x, width_x = input_layer.shape
    num_filters, channels_w, height_w, width_w = weight.shape

    input_layer_gradient = np.zeros(
        (batch_size, channels_x, height_x, width_x))
    weight_gradient = np.zeros((num_filters, channels_x, height_w, width_w))
    bias_gradient = np.zeros((num_filters))

    padded_input = np.pad(
        input_layer, ((0, 0), (0, 0), (pad_size, pad_size), (pad_size, pad_size)),  'constant')

    assert num_filters == channels_y, (
        "The number of filters must be the same as the number of output layer channels")
    assert channels_w == channels_x, (
        "The number of filter channels be the same as the number of input layer channels")

    padded_out = np.pad(output_layer_gradient, ((0, 0), (0, 0),
                                                (pad_size, pad_size), (pad_size, pad_size)),  'constant')

    K = int((height_w + 1)/2)

    for i in range(batch_size):
        for j in range(num_filters):
            # POSITION
            for p in range(height_y):
                for q in range(width_y):
                    bias_gradient[j] += output_layer_gradient[i, j, p, q]

                    # CHANNEL
                    for k in range(channels_x):
                        input_layer_gradient[i, k, p,
                                             q] += np.sum(padded_out[i, j, p:p+K+1, q:q+K+1] * np.flip(weight[j, k]))

                        # FILTER
                        for r in range(height_w):
                            for s in range(width_w):
                                weight_gradient[j, k, r, s] += np.sum(
                                    output_layer_gradient[i, j, p, q] * padded_input[i, k, p + r, q + s])

    return input_layer_gradient, weight_gradient, bias_gradient


def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """
    Evaluate a numeric gradient for a function that accepts a numpy
    array and returns a numpy array.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad
