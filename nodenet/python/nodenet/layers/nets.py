# nodenet/layers/nets.py
# Description:
# "fullyconnectednet.py" provide net layers.
# Copyright 2018 NOOXY. All Rights Reserved.

from nodenet.imports.commons import *
from .base import *

# Fully Connected Net input: 2D vector, output: 2D vector
class FullyConnected1D(Layer):
    def __init__(self, front_nodes_size=0, back_nodes_size=0, has_bias=True, ramdom_max_num=1e-0):
        # Copy property
        self.front_nodes_size = front_nodes_size
        self.back_nodes_size = back_nodes_size
        self.has_bias = has_bias
        self.ramdom_max_num = ramdom_max_num
        # Initialize weights
        self.weights = np.random.uniform(-ramdom_max_num, ramdom_max_num, (front_nodes_size, back_nodes_size))
        self.weights_grad = np.zeros(self.weights.shape)
        self.weights_learning_cache = None
        # Initialize bias
        if self.has_bias == True:
            self.bias = np.random.uniform(-ramdom_max_num, ramdom_max_num, (1, back_nodes_size))
            self.bias_grad = np.zeros(self.bias.shape)
            self.bias_learning_cache = None
        # Initialize cache
        self.latest_input_signal = None
        self.latest_sensitivity_map = None

    def __str__(self):
        string = ''
        string += 'FullyConnectedNet1D('+str(self.front_nodes_size)+'x'+str(self.back_nodes_size)+')'
        return string

    def clear_cache(self):
        self.weights_learning_cache = None
        self.bias_learning_cache = None
        self.latest_input_signal = None
        self.latest_sensitivity_map = None

    def forward(self, input_signal, forward_config, *args):
        # print(input_signal.shape)
        trace = forward_config['trace']
        if trace:
            self.latest_input_signal = input_signal
        weight_output =  np.dot(input_signal, self.weights)
        output_signal = None
        if self.has_bias == True:
            bias_output = np.dot(np.ones((input_signal.shape[0], 1)), self.bias)
            output_signal = weight_output + bias_output
        else:
            output_signal = weight_output
        return output_signal

    def update_gradient(self, input_sensitivity_map):
        self.latest_sensitivity_map = np.dot(input_sensitivity_map, np.transpose(self.weights))
        self.weights_grad = np.dot(np.transpose(self.latest_input_signal), input_sensitivity_map)
        if self.has_bias == True:
            self.bias_grad = np.dot(np.ones((1, input_sensitivity_map.shape[0])), input_sensitivity_map)

    def update_weight_and_bias(self, learning_algorithm, learning_configuration):
        weight_update, self.weight_learning_cache = learning_algorithm(self.weights_grad, learning_configuration, self.weights_learning_cache)
        self.weights += weight_update
        if self.has_bias == True:
            bias_update, self.bias_learning_cache = learning_algorithm(self.bias_grad, learning_configuration, self.bias_learning_cache)
            self.bias += bias_update

    def get_sensitivity_map(self):
        return self.latest_sensitivity_map

    def backward(self, input_sensitivity_map, backward_config, *args):
        learning_algorithm = backward_config['learning_algorithm']
        learning_configuration = backward_config['learning_configuration']
        self.update_gradient(input_sensitivity_map)
        self.update_weight_and_bias(learning_algorithm, learning_configuration)
        return self.get_sensitivity_map()
