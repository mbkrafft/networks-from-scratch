"""Define the dense neural network model"""

import numpy as np
from scipy.stats import truncnorm


def one_hot(Y, num_classes):
    """Perform one-hot encoding on input Y.

    It is assumed that Y is a 1D numpy array of length m_b (batch_size) with integer values in
    range [0, num_classes-1]. The encoded matrix Y_tilde will be a [num_classes, m_b] shaped matrix
    with values

                   | 1,  if Y[i] = j
    Y_tilde[i,j] = |
                   | 0,  else
    """
    m = len(Y)
    Y_tilde = np.zeros((num_classes, m))
    Y_tilde[Y, np.arange(m)] = 1
    return Y_tilde


def initialization(conf):
    """Initialize the parameters of the network.

    Args:
        layer_dimensions: A list of length L+1 with the number of nodes in each layer, including
                          the input layer, all hidden layers, and the output layer.
    Returns:
        params: A dictionary with initialized parameters for all parameters (weights and biases) in
                the network.
    """
    params = {}

    layers = conf['layer_dimensions']

    for i, layer in enumerate(layers):
        if i > 0:
            if conf['initialization'] == 'zero':
                params[f'b_{i}'] = np.zeros((layer, 1))
                params[f'W_{i}'] = np.zeros((layers[i - 1], layer))
            else:
                # Standard He initialization
                params[f'b_{i}'] = np.zeros((layer, 1))
                params[f'W_{i}'] = np.random.normal(
                    loc=0, scale=np.sqrt(2/layers[i - 1]), size=(layers[i - 1], layer))

    return params


def activation(Z, activation_function):
    """Compute a non-linear activation function.

    Args:
        Z: numpy array of floats with shape [n, m]
    Returns:
        numpy array of floats with shape [n, m]
    """
    # Activations from https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
    A = np.copy(Z)

    if activation_function == 'relu':
        A[A < 0] = 0
        return A
    elif activation_function == 'tanh':
        return (np.exp(A) - np.exp(-A)) / (np.exp(A) + np.exp(-A))
    elif activation_function == 'sigmoid':
        return 1 / (1 + np.exp(-A))
    else:
        print("Error: Unimplemented activation function: {}", activation_function)
        return None


def softmax(Z):
    """Compute and return the softmax of the input.

    To improve numerical stability, we do the following

    1: Subtract Z from max(Z) in the exponentials
    2: Take the logarithm of the whole softmax, and then take the exponential of that in the end

    Args:
        Z: numpy array of floats with shape [n, m]
    Returns:
        numpy array of floats with shape [n, m]
    """
    softmax = Z - np.log(np.sum(np.exp(Z - np.max(Z)), axis=0))

    return np.exp(softmax - np.max(Z))


def forward(conf, X_batch, params, is_training):
    """One forward step.

    Args:
        conf: Configuration dictionary.
        X_batch: float numpy array with shape [n^[0], batch_size]. Input image batch.
        params: python dict with weight and bias parameters for each layer.
        is_training: Boolean to indicate if we are training or not. This function can namely be
                     used for inference only, in which case we do not need to store the features
                     values.

    Returns:
        Y_proposed: float numpy array with shape [n^[L], batch_size]. The output predictions of the
                    network, where n^[L] is the number of prediction classes. For each input i in
                    the batch, Y_proposed[c, i] gives the probability that input i belongs to class
                    c.
        features: Dictionary with
                - the linear combinations Z^[l] = W^[l]a^[l-1] + b^[l] for l in [1, L].
                - the activations A^[l] = activation(Z^[l]) for l in [1, L].
               We cache them in order to use them when computing gradients in the backpropagation.
    """
    Y_proposed = None
    features = {}

    layers = conf['layer_dimensions']
    activation_fn = conf['activation_function']

    for i in range(1, len(layers)):
        if i == 1:
            features[f'A_{i - 1}'] = activation(X_batch, activation_fn)
            Z = np.dot(params[f'W_{i}'].T, X_batch) + params[f'b_{i}']
        else:
            Z = np.dot(params[f'W_{i}'].T,
                       features[f'A_{i - 1}']) + params[f'b_{i}']

        features[f'Z_{i}'] = Z
        features[f'A_{i}'] = activation(Z, activation_fn)

    Y_proposed = softmax(features[f'Z_{len(layers) - 1}'])

    return Y_proposed, features


def cross_entropy_cost(Y_proposed, Y_reference):
    """Compute the cross entropy cost function.

    Args:
        Y_proposed: numpy array of floats with shape [n_y, m].
        Y_reference: numpy array of floats with shape [n_y, m]. Collection of one-hot encoded
                     true input labels

    Returns:
        cost: Scalar float: 1/m * sum_i^m sum_j^n y_reference_ij log y_proposed_ij
        num_correct: Scalar integer
    """
    cost = - np.sum(Y_reference * np.log(Y_proposed)) / Y_reference.shape[1]

    num_correct = np.sum(Y_proposed.argmax(axis=0) ==
                         Y_reference.argmax(axis=0))

    return cost, num_correct


def activation_derivative(Z, activation_function):
    """Compute the gradient of the activation function.

    Args:
        Z: numpy array of floats with shape [n, m]
    Returns:
        numpy array of floats with shape [n, m]
    """
    # Derivations from https://theclevermachine.wordpress.com/2014/09/08/derivation-derivatives-for-common-neural-network-activation-functions/
    A = np.copy(Z)

    if activation_function == 'relu':
        A[A >= 0] = 1
        A[A <= 0] = 0
        return A
    elif activation_function == 'tanh':
        return 1 - (activation(A, 'tanh') ** 2)
    elif activation_function == 'sigmoid':
        g = activation(A, 'sigmoid')
        return g * (1 - g)
    else:
        print("Error: Unimplemented derivative of activation function: {}",
              activation_function)
        return None


def backward(conf, Y_proposed, Y_reference, params, features):
    """Update parameters using backpropagation algorithm.

    Args:
        conf: Configuration dictionary.
        Y_proposed: numpy array of floats with shape [n_y, m].
        features: Dictionary with matrices from the forward propagation. Contains
                - the linear combinations Z^[l] = W^[l]a^[l-1] + b^[l] for l in [1, L].
                - the activations A^[l] = activation(Z^[l]) for l in [1, L].
        params: Dictionary with values of the trainable parameters.
                - the weights W^[l] for l in [1, L].
                - the biases b^[l] for l in [1, L].
    Returns:
        grad_params: Dictionary with matrices that is to be used in the parameter update. Contains
                - the gradient of the weights, grad_W^[l] for l in [1, L].
                - the gradient of the biases grad_b^[l] for l in [1, L].
    """
    grad_params = {}

    layers = conf['layer_dimensions']
    activation_fn = conf['activation_function']
    m = Y_proposed.shape[1]

    for i in range(len(layers) - 1, 0, -1):
        if i == len(layers) - 1:
            J = Y_proposed - Y_reference
        else:
            J = activation_derivative(
                features[f'Z_{i}'], activation_fn) * np.dot(params[f'W_{i + 1}'], prev)

        A = features[f'A_{i - 1}']
        prev = J

        grad_params[f'grad_W_{i}'] = np.dot(A, J.T) / m
        grad_params[f'grad_b_{i}'] = np.sum(J, axis=1, keepdims=True) / m

    return grad_params


def gradient_descent_update(conf, params, grad_params):
    """Update the parameters in params according to the gradient descent update routine.

    Args:
        conf: Configuration dictionary
        params: Parameter dictionary with W and b for all layers
        grad_params: Parameter dictionary with b gradients, and W gradients for all
                     layers.
    Returns:
        params: Updated parameter dictionary.
    """
    updated_params = {}

    lr = conf['learning_rate']

    for key in params.keys():
        updated_params[key] = params[key] - lr * grad_params[f'grad_{key}']

    return updated_params
