# nodenet/functions/loss.py
# Description:
# "loss.py" provide cost function for neuralnet.
# Copyright 2018 NOOXY. All Rights Reserved.

from nodenet.imports.commons import *

# Mean Square Loss
def mean_square(output_signal, target, derivative=False):
    if derivative:
        return output_signal-target
    else:
        return np.mean(np.sum(np.power(output_signal-target, 2), axis=-1))

# Binary Cross Entropy Loss
def binary_cross_entropy(output_signal, target, derivative=False, epsilon=1e-11):
    # Prevent overflow output_signal should be in [0, 1]
    output_signal = np.clip(output_signal, epsilon, 1-epsilon)
    divisor = np.maximum(output_signal*(1-output_signal), epsilon)

    if derivative:
        return (output_signal-target)/divisor
    else:
        return -np.mean(np.sum(np.multiply(target, np.log(output_signal))+np.multiply((1-target), np.log(1-output_signal)), axis=-1))

# Softmax Cross Entropy Loss
def softmax_cross_entropy(output_signal, target, derivative=False, epsilon=1e-11):
    # Prevent overflow output_signal should be in [0, 1]
    output_signal = np.clip(output_signal, epsilon, 1-epsilon)

    if derivative:
        return output_signal-target
    else:
        return -np.mean(np.sum(np.multiply(target, np.log(output_signal)), axis=-1))
