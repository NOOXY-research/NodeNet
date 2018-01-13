# nodenet/layers/convolutional.py
# Description:
# "convolutional.py" provide convolutional layers.
# Copyright 2018 NOOXY. All Rights Reserved.

from nodenet.imports.commons import *
from .base import *

def expand_map(input_data, stride):
    if stride==1:
        return input_data
    interval = stride-1
    input_data_width = input_data.shape[-1]
    input_data_height = input_data.shape[-2]
    expanded_data_width = input_data_width+(input_data_width-1)*interval
    expanded_data_height = input_data_height+(input_data_height-1)*interval
    expanded_data_depth = input_data.shape[-3]
    expanded_data_number = input_data.shape[0]
    expanded_data = np.zeros((expanded_data_number, expanded_data_depth, expanded_data_height, expanded_data_width))
    for i in range(input_data_width):
            for j in range(input_data_height):
                i_pos = i*stride
                j_pos = j*stride
                expanded_data[:, :, i_pos, j_pos] = input_data[:, :, i, j]
    return expanded_data

def padding(input_data, padding_size, padding_value):
    if padding_size == 0:
        return input_data
    input_width = input_data.shape[-1]
    input_height = input_data.shape[-2]
    input_depth = input_data.shape[-3]
    input_number = input_data.shape[-4]
    if type(padding_size) is tuple:
        padded_data = padding_value*np.ones((input_number, input_depth, input_height+2*padding_size[1], input_width+2*padding_size[0]))
        padded_data[:, :, padding_size[1] : padding_size[1]+input_height,padding_size[0] : padding_size[0]+input_width] = input_data
    else:
        padded_data = padding_value*np.ones((input_number, input_depth, input_height+2*padding_size, input_width+2*padding_size))
        padded_data[:, :, padding_size : padding_size+input_height,padding_size : padding_size+input_width] = input_data
    return padded_data

def get_patch(input_data, i, j, kernel_width, kernel_height, stride):
    start_i = i*stride
    start_j = j*stride
    return input_data[:, :, start_i : start_i+kernel_height, start_j : start_j+kernel_width]

def calc_conv_output_size(input_size, filter_size, stride):
    return int(np.floor((input_size-filter_size)/stride)+1)

# Convulutional Layer input: 4D tensor, output: 4D tensor
class Conv2D(Layer):
    def __init__(self, input_width, input_height, input_depth, filter_width, filter_height, filter_number, stride=1, padding_size=0, bias=True, ramdom_max_num=1e-4):
        self.input_width = input_width
        self.input_height = input_height
        self.input_depth = input_depth
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.filter_depth = input_depth
        self.filter_number = filter_number
        self.filter = np.random.uniform(-ramdom_max_num, ramdom_max_num, (filter_number, input_depth, filter_height, filter_width))
        self.filter_grad = np.zeros(self.filter.shape)
        self.filter_learning_cache = None
        self.stride = stride
        self.padding_size = padding_size
        self.has_bias = bias
        if bias ==True:
            self.bias = np.zeros((1, filter_number))
            self.bias_grad = np.zeros(self.bias.shape)
            self.bias_learning_cache = None
        # Initialize cache
        self.latest_input_signal = None
        self.latest_sensitivity_map = None

    def __str__(self):
        string = ''
        string += 'ConvLayer(filter number: '+str(self.filter_number)+'('+str(self.filter_width)+'x'+str(self.filter_height)+') , stride: '+str(self.stride)+')'
        return string

    def clear_cache(self):
        self.filter_learning_cache = None
        self.bias_learning_cache = None
        self.latest_input_signal = None
        self.latest_sensitivity_map = None

    def conv_forward(self, input_data, stride, bias=None):
        kernel = self.filter
        output_width = calc_conv_output_size(input_data.shape[-1], kernel.shape[-1], stride)
        output_height = calc_conv_output_size(input_data.shape[-2], kernel.shape[-2], stride)
        output_depth = kernel.shape[0]
        output_number = input_data.shape[0]
        kernel_width = kernel.shape[-1]
        kernel_height = kernel.shape[-2]
        output_data = np.zeros((output_number, output_depth, output_height, output_width))
        for i in range(output_height):
            for j in range(output_width):
                patched_data = get_patch(input_data, i, j, kernel_width, kernel_height, stride)
                filter_out_data = np.tensordot(patched_data, kernel, axes=([-3,-2,-1], [-3,-2,-1]))
                conved_data = filter_out_data
                if bias is not None:
                    conved_data += bias
                output_data[:, :, i, j] = conved_data
        return output_data

    def conv_sensitivty_map(self, expand_sens_map):
        kernel = np.swapaxes(np.rot90(self.filter, 2, axes=(-2,-1)), 0, 1)
        if self.padding_size > 0:
            expand_sens_map = expand_sens_map[:, :, self.padding_size*(self.stride):-self.padding_size*(self.stride), self.padding_size*(self.stride):-self.padding_size*(self.stride)]
        expand_sens_map = padding(expand_sens_map, (self.filter_height-1, self.filter_width-1), 0)
        output_width = self.latest_input_signal.shape[-1]
        output_height = self.latest_input_signal.shape[-2]
        output_depth = self.latest_input_signal.shape[-3]
        output_number = expand_sens_map.shape[0]
        kernel_width = kernel.shape[-1]
        kernel_height = kernel.shape[-2]
        actual_output_width = calc_conv_output_size(expand_sens_map.shape[-1], kernel_width, 1)
        actual_output_height = calc_conv_output_size(expand_sens_map.shape[-2], kernel_height, 1)
        output_data = np.zeros((output_number, output_depth, output_height, output_width))
        for i in range(actual_output_width):
            for j in range(actual_output_height):
                patched_data = get_patch(expand_sens_map, i, j, kernel_width, kernel_height, 1)
                conved_data = np.tensordot(patched_data, kernel, axes=([-3,-2,-1], [-3,-2,-1]))
                output_data[:, :, i, j] = conved_data
        return output_data

    def conv_filter_grad(self, expand_sens_map):
        kernel = np.swapaxes(expand_sens_map, 0, 1)
        padded_input_data = np.swapaxes(padding(self.latest_input_signal, self.padding_size, 0), 0, 1)
        output_width = self.filter.shape[-1]
        output_height = self.filter.shape[-2]
        output_depth = self.filter.shape[-3]
        output_number = self.filter.shape[-4]
        kernel_width = kernel.shape[-1]
        kernel_height = kernel.shape[-2]
        output_data = np.zeros((output_number, output_depth, output_height, output_width))
        for i in range(output_height):
            for j in range(output_width):
                patched_data = get_patch(padded_input_data, i, j, kernel_width, kernel_height, 1)
                conved_data = np.tensordot(patched_data, kernel, axes=([-3,-2,-1], [-3,-2,-1]))
                output_data[:, :, i, j] = conved_data.transpose()
        return output_data

    def forward(self, input_signal, forward_config, *args):
        if input_signal.shape != (input_signal.shape[0], self.input_depth , self.input_height, self.input_width):
            raise ValueError('shape mismatch: input_signal'+str(input_signal.shape)+', should be'+str((input_signal.shape[0], self.input_depth , self.input_height, self.input_width)))
        trace = forward_config['trace']
        if trace:
            self.latest_input_signal = input_signal
        padded_input_data = padding(input_signal, self.padding_size, 0)
        output_signal = self.conv_forward(padded_input_data, self.stride, self.bias)
        return output_signal

    def update_gradient(self, input_sensitivity_map):
        expand_sens_map = expand_map(input_sensitivity_map, stride=self.stride)
        self.latest_sensitivity_map = self.conv_sensitivty_map(expand_sens_map)
        self.filter_grad = self.conv_filter_grad(expand_sens_map)
        if self.has_bias == True:
            self.bias_grad = input_sensitivity_map.sum(axis=0).sum(axis=-1).sum(axis=-1)

    def update_filter_and_bias(self, learning_algorithm, learning_configuration):
        filter_update, self.filter_learning_cache = learning_algorithm(self.filter_grad, learning_configuration, self.filter_learning_cache)
        self.filter += filter_update
        if self.has_bias == True:
            bias_update, self.bias_learning_cache = learning_algorithm(self.bias_grad, learning_configuration, self.bias_learning_cache)
            self.bias += bias_update

    def get_sensitivity_map(self):
        return self.latest_sensitivity_map

    def backward(self, input_sensitivity_map, backward_config, *args):
        learning_algorithm = backward_config['learning_algorithm']
        learning_configuration = backward_config['learning_configuration']
        self.update_gradient(input_sensitivity_map)
        self.update_filter_and_bias(learning_algorithm, learning_configuration)
        return self.get_sensitivity_map()
