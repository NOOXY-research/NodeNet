# nodenet/function/activation.py
# Description:
# "activation.py" provide activation function for neuralnet.
# Copyright 2018 NOOXY. All Rights Reserved.

from nodenet.imports.commons import *

# Sigmoid
def sigmoid(signal, derivative=False):
    # Prevent NaN error
    signal = np.clip(signal, -1000, 1000)
    signal = 1.0/(1.0+np.exp(-signal))

    if derivative:
        # Return derivation of the activation function
        return np.multiply(signal, 1-signal)
    else:
        # Return the activation signal
        return signal

# Softmax 1 dimension
def softmax1D(signal, derivative=False):
    # Exp all elements
    exp_signal = np.exp(signal-np.max(signal, axis=-1, keepdims=True))
    signal = exp_signal/np.sum(exp_signal, axis=-1, keepdims=True)

    if derivative:
        # Return derivation of the activation function
        return np.ones(signal.shape)
    else:
        # Return the activation signal
        return signal

# Softmax 2 dimension
def softmax2D(signal, derivative=False):
    # Exp all elements
    exp_signal = np.exp(signal-np.max(np.max(signal, axis=-1, keepdims=True), axis=-2, keepdims=True))
    signal = exp_signal/np.sum(np.sum(exp_signal, axis=-1, keepdims=True), axis=-2, keepdims=True)
    if derivative:
        # Return derivation of the activation function
        return np.ones(signal.shape)
    else:
        # Return the activation signal
        return signal

# Elliot (A fast approximation of sigmoid)
def elliot(signal, derivative=False):
    # steepness
    s = 1
    abs_signal = (1+np.abs(signal*s))

    if derivative:
        # Return derivation of the activation function
        return 0.5*s/abs_signal**2
    else:
        # Return the activation signal
        return 0.5*(signal*s)/abs_signal+0.5

# Symmetric Elliot (A fast approximation of tanh)
def symmetric_elliot(signal, derivative=False):
    s = 1.0 # steepness

    abs_signal = (1+np.abs(signal*s))
    if derivative:
        # Return derivation of the activation function
        return s/abs_signal**2
    else:
        # Return the activation signal
        return (signal*s)/abs_signal

# ReLU
def ReLU(signal, derivative=False):
    if derivative:
        # Return derivation of the activation function
        return (signal > 0).astype(float)
    else:
        # Return the activation signal
        return np.maximum(0, signal)

# Leaky Rectified Linear Unit
def LReLU(signal, derivative=False, leakage=0.01):
    if derivative:
        # Return derivation of the activation function
        return np.clip(signal>0, leakage, 1.0)
    else:
        # Return the activation signal
        output = np.copy(signal)
        output[output < 0] *= leakage
        return output

# tanh
def tanh(signal, derivative=False):
    # Calculate activation signal
    signal = np.tanh(signal)

    if derivative:
        # Return derivation of the activation function
        return 1-np.power(signal, 2)
    else:
        # Return the activation signal
        return signal

# Linear
def linear(signal, derivative=False):
    if derivative:
        # Return derivation of the activation function
        return np.ones(signal.shape)
    else:
        # Return the activation signal
        return signal

# Softplus
def softplus(signal, derivative=False):
    if derivative:
        # Return derivation of the activation function
        return np.exp(signal)/(1+np.exp(signal))
    else:
        # Return the activation signal
        return np.log(1+np.exp(signal))

# Softsign
def softsign(signal, derivative=False):
    if derivative:
        # Return derivation of the activation function
        return 1.0/(1+np.abs(signal))**2
    else:
        # Return the activation signal
        return signal/(1+np.abs(signal))
