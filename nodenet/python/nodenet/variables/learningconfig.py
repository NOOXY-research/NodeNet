# nodenet/learningalgorithm/defaultconfiguration.py
# Description:
# "defaultconfiguration.py" provide default parameters for each learnling algorithm.
# Copyright 2018 NOOXY. All Rights Reserved.

vanilla_config = {
    'learning_rate' : 10e-4,
}

vanilla_momentum_config = {
    'learning_rate' : 10e-4,
    'momentumrate' : 0.9,
}

nesterov_momentum_config = {
    'learning_rate' : 10e-4,
    'momentumrate' : 0.9,
}

adagrad_config = {
    'learning_rate' : 10e-4,
    'epsilon' : 10e-8,
}

adadelta_config = {
    'epsilon' : 10e-8,
    'decay_rate' : 0.9,
}

rmsprop_config = {
    'learning_rate' : 10e-4,
    'epsilon' : 10e-8,
    'decay_rate' : 0.9,
}

adam_config = {
    'learning_rate' : 10e-4,
    'epsilon' : 10e-8,
    'beta1' : 0.9,
    'beta2' : 0.999,
}
