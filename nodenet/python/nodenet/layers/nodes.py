# nodenet/layers/nodes.py
# Description:
# "nodes.py" provide node layers.
# Copyright 2018 NOOXY. All Rights Reserved.

from nodenet.imports.commons import *
from .base import *
import nodenet.functions as func
import numpy as np2

# Vector Nodes input: 2D vector, output: 2D vector
class Nodes1D(Layer):
    def __init__(self, nodes_number, activatior=func.sigmoid):
        self.nodes_number = nodes_number
        self.activator = activatior
        self.latest_input_signal = None
        self.latest_sensitivity_map = None
        self.latest_dropout_keep_mask = None
        self.is_input_layer = False
        self.is_output_layer = False

    def __str__(self):
        string = ''
        string += 'Nodes1D(nodes number: '+str(self.nodes_number)+', activator: '+str(self.activator)+')'
        return string

    def clear_cache(self):
        self.latest_input_signal = None
        self.latest_sensitivity_map = None
        self.latest_dropout_keep_mask = None

    def new_dropout(self, dropout_keep):
        if (dropout_keep < 1) and (not self.is_input_layer) and (not self.is_output_layer):
            self.latest_dropout_keep_mask = np.array(np2.random.binomial(np2.ones(shape=(1, self.nodes_number), dtype=np2.int64), dropout_keep).tolist())

    def forward(self, input_signal, forward_config, *args):
        trace = forward_config['trace']
        dropout_keep = forward_config['dropout_keep']
        if trace:
            self.latest_input_signal = input_signal
        output_signal = self.activator(input_signal)
        # Do dropout
        if (dropout_keep < 1) and (not self.is_input_layer) and (not self.is_output_layer):
            output_signal = (1.0/dropout_keep)*output_signal*self.latest_dropout_keep_mask

        return output_signal

    def update_gradient(self, input_sensitivity_map, dropout_keep=0):
        # Do dropout
        self.latest_sensitivity_map = np.multiply(input_sensitivity_map, self.activator(self.latest_input_signal, derivative=True))
        if (dropout_keep < 1) and (not self.is_input_layer) and (not self.is_output_layer):
            self.latest_sensitivity_map = (1.0/dropout_keep)*self.latest_sensitivity_map*self.latest_dropout_keep_mask

    def get_sensitivity_map(self):
        return self.latest_sensitivity_map

    def backward(self, input_sensitivity_map, backward_config, *args):
        dropout_keep = backward_config['dropout_keep']
        self.update_gradient(input_sensitivity_map, dropout_keep)
        return self.get_sensitivity_map()

class Nodes2D(Layer):
        def __init__(self, nodes_width, nodes_height, nodes_depth, activatior=func.sigmoid):
            self.nodes_width = nodes_width
            self.nodes_height = nodes_height
            self.nodes_depth = nodes_depth
            self.activator = activatior
            self.latest_input_signal = None
            self.latest_sensitivity_map = None
            self.latest_dropout_keep_mask = None
            self.is_input_layer = False
            self.is_output_layer = False

        def __str__(self):
            string = ''
            string += 'Nodes2D(nodes : '+str(self.nodes_width)+'x'+str(self.nodes_height)+', activator: '+str(self.activator)+')'
            return string

        def clear_cache(self):
            self.latest_input_signal = None
            self.latest_sensitivity_map = None
            self.latest_dropout_keep_mask = None

        def new_dropout(self, dropout_keep):
            if (dropout_keep < 1) and (not self.is_input_layer) and (not self.is_output_layer):
                self.latest_dropout_keep_mask = np.array(np2.random.binomial(np2.ones(shape=(1, self.nodes_depth, self.nodes_height, self.nodes_width), dtype=np2.int64), dropout_keep).tolist())

        def forward(self, input_signal, forward_config, *args):
            # print(input_signal.shape)
            trace = forward_config['trace']
            dropout_keep = forward_config['dropout_keep']
            if trace:
                self.latest_input_signal = input_signal
            output_signal = self.activator(input_signal)
            # Do dropout
            if (dropout_keep < 1) and (not self.is_input_layer) and (not self.is_output_layer):
                output_signal = (1.0/dropout_keep)*output_signal*self.latest_dropout_keep_mask

            return output_signal

        def update_gradient(self, input_sensitivity_map, dropout_keep=0):
            # Do dropout
            self.latest_sensitivity_map = np.multiply(input_sensitivity_map, self.activator(self.latest_input_signal, derivative=True))
            if (dropout_keep < 1) and (not self.is_input_layer) and (not self.is_output_layer):
                self.latest_sensitivity_map = (1.0/dropout_keep)*self.latest_sensitivity_map*self.latest_dropout_keep_mask

        def get_sensitivity_map(self):
            return self.latest_sensitivity_map

        def backward(self, input_sensitivity_map, backward_config, *args):
            dropout_keep = backward_config['dropout_keep']
            self.update_gradient(input_sensitivity_map, dropout_keep)
            return self.get_sensitivity_map()
