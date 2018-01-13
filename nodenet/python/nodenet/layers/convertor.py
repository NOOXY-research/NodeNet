# nodenet/layers/convertor.py
# Description:
# "convertor.py" provide function of converting layers.
# Copyright 2018 NOOXY. All Rights Reserved.

from nodenet.imports.commons import *
from .base import *

class TensortoVector(Layer):
    def __init__(self):
        self.latest_input_shape = None

    def __str__(self):
        string = ''
        string += 'TensortoVector'
        return string

    def clear_cache(self):
        self.latest_input_shape = None

    def forward(self, input_signal, forward_config, *args):
        trace = forward_config['trace']
        if trace:
            self.latest_input_shape = input_signal.shape

        output_signal = input_signal.reshape(input_signal.shape[0], -1)
        return output_signal

    def backward(self, input_sensitivity_map, backward_config, *args):
        shape = self.latest_input_shape
        sensitivity_map = np.reshape(input_sensitivity_map, (input_sensitivity_map.shape[0], shape[1],shape[2],shape[3]))
        return sensitivity_map
