# nodenet/learningalgorithm/backpropagations.py
# Description:
# "backpropagations.py" provide backpropagation type of training.
# Copyright 2018 NOOXY. All Rights Reserved.

from nodenet.imports.commons import *

# Vanilla BackPropagation
def vanilla(gradient, configuration, cache):
    # Load learning configuration
    learning_rate = configuration['learning_rate']
    return -learning_rate*gradient, cache

# BackPropagation with vanilla momentum
def vanilla_momentum(gradient, configuration, cache):
    # Load learning configuration
    learning_rate = configuration['learning_rate']
    momentum_rate = configuration['momentum_rate']
    momentum = None
    # Recover variables from cache
    if cache is not None:
        momentum = cache[0]
    else:
        momentum = np.zeros(gradient.shape)
    newgradient = -learning_rate*gradient+momentum_rate*momentum
    return newgradient, [newgradient]

# BackPropagation with nesterov momentum
def nesterov_momentum(gradient, configuration, cache):
    # Load learning configuration
    learning_rate = configuration['learning_rate']
    momentum_rate = configuration['momentum_rate']
    momentum = None
    # Recover variables from cache
    if cache is not None:
        momentum = cache[0]
        newgradient = -learning_rate*gradient+momentum_rate*momentum
    else:
        momentum = np.zeros(gradient.shape)
        newgradient = np.zeros(gradient.shape)
    newmomentum = -learning_rate*gradient+momentum_rate*momentum
    return newgradient, [newmomentum]

# AdaGrad BackPropagation
def adagrad(gradient, configuration, cache):
    # Load learning configuration
    learning_rate = configuration['learning_rate']
    epsilon = configuration['epsilon']
    squaresum = None
    # Recover variables from cache
    if cache is not None:
        squaresum = cache[0]
    else:
        squaresum = np.zeros(gradient.shape)
    squaresum += np.power(gradient, 2)
    newgradient = -learning_rate*gradient/(np.sqrt(squaresum)+epsilon)
    return newgradient, [squaresum]

# AdaDelta BackPropagation
def adadelta(gradient, configuration, cache):
    # Load learning configuration
    epsilon = configuration['epsilon']
    decay_rate = configuration['decay_rate']
    squaresumdelta = None
    squaresumgradient = None
    # Recover variables from cache
    if cache is not None:
        squaresumdelta = cache[0]
        squaresumgradient = cache[1]
    else:
        squaresumdelta = np.zeros(gradient.shape)
        squaresumgradient = np.zeros(gradient.shape)
    squaresumgradient = decay_rate*squaresumgradient + (1-decay_rate)*np.power(gradient, 2)
    newgradient = -np.dot((np.sqrt(squaresumdelta)+epsilon)/(np.sqrt(squaresumgradient)+epsilon), gradient)
    squaresumdelta = decay_rate*squaresumdelta + (1-decay_rate)*np.power(newgradient, 2)
    return newgradient, [squaresumdelta, squaresumgradient]

# RMSprop BackPropagation
def rmsprop(gradient, configuration, cache):
    # Load learning configuration
    learning_rate = configuration['learning_rate']
    epsilon = configuration['epsilon']
    decay_rate = configuration['decay_rate']
    squaresum = None
    # Recover variables from cache
    if cache is not None:
        squaresum = cache[0]
    else:
        squaresum = np.zeros(gradient.shape)
    squaresum += decay_rate*squaresum + (1-decay_rate)*np.power(gradient, 2)
    newgradient = -learning_rate*gradient/(np.sqrt(squaresum)+epsilon)
    return newgradient, [squaresum]

# Adam BackPropagation
def adam(gradient, configuration, cache):
    # Load learning configuration
    learning_rate = configuration['learning_rate']
    epsilon = configuration['epsilon']
    beta1 = configuration['beta1']
    beta2 = configuration['beta2']
    gradientm = None
    gradientv = None
    # Recover variables from cache
    if cache is not None:
        gradientm = cache[0]
        gradientv = cache[1]
        t = cache[2]+1
    else:
        gradientm = np.zeros(gradient.shape)
        gradientv = np.zeros(gradient.shape)
        t = 1
    gradientm = beta1*gradientm+(1-beta1)*gradient
    gradientv = beta2*gradientv+(1-beta2)*np.power(gradient, 2)
    gradientm_delta = gradientm/(1-beta1**t)
    gradientv_delta = gradientv/(1-beta2**t)
    newgradient = -learning_rate*gradientm_delta/(np.sqrt(gradientv_delta)+epsilon)
    return newgradient, [gradientm, gradientv, t]
