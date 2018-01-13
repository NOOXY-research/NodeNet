# nodenet/layers/pooling.py
# Description:
# "pooling.py" provide net layers.
# Copyright 2018 NOOXY. All Rights Reserved.

from nodenet.imports.commons import *
from .base import *

def get_patch(input_data, i, j, kernel_width, kernel_height, stride):
    start_i = i*stride
    start_j = j*stride
    return input_data[:, :, start_i : start_i+kernel_height, start_j : start_j+kernel_width]

def calc_conv_output_size(input_size, filter_size, stride):
    return int(np.floor((input_size-filter_size)/stride)+1)

class MaxPool2D(Layer):
    def __init__(self, input_width, input_height, input_depth, filter_width, filter_height, stride=1):
        self.input_width = input_width
        self.input_height = input_height
        self.input_depth = input_depth
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.filter_depth = input_depth
        self.stride = stride
        # Initialize cache
        self.latest_sensitivity_map_maxidx = None
        self.latest_input_signal_shape = None

    def __str__(self):
        string = ''
        string += 'MaxPooling()'
        return string

    def clear_cache(self):
        self.latest_sensitivity_map_maxidx = None
        self.latest_input_signal_shape = None

    def pooling_forward(self, input_data):
        stride = self.stride
        kernel_width = self.filter_width
        kernel_height = self.filter_height
        output_width = calc_conv_output_size(input_data.shape[-1], kernel_width, stride)
        output_height = calc_conv_output_size(input_data.shape[-2], kernel_height, stride)
        output_depth = input_data.shape[1]
        output_number = input_data.shape[0]
        self.latest_sensitivity_map_maxidx = np.zeros((output_number, output_depth, output_height, output_width), dtype=int)
        output_data = np.zeros((output_number, output_depth, output_height, output_width))
        for i in range(output_height):
            for j in range(output_width):
                patched_data = get_patch(input_data, i, j, kernel_width, kernel_height, stride)
                patched_data_reshape = patched_data.reshape(patched_data.shape[0]*patched_data.shape[1], -1)
                max_idx = np.argmax(patched_data_reshape, axis=-1)
                self.latest_sensitivity_map_maxidx[:, :, i, j] = max_idx.reshape(output_number, output_depth)
                max_values = patched_data_reshape[np.arange(max_idx.size), max_idx]
                output_data[:, :, i, j] = max_values.reshape(output_number, output_depth)
        return output_data

    def pooling_backward(self, sens_map):
        stride = self.stride
        kernel_width = self.filter_width
        kernel_height = self.filter_height
        output_width = self.latest_input_signal_shape[-1]
        output_height = self.latest_input_signal_shape[-2]
        output_depth = self.latest_input_signal_shape[-3]
        output_number = sens_map.shape[0]
        output_data = np.zeros((output_number, output_depth, output_height, output_width))
        for i in range(calc_conv_output_size(output_height, kernel_height, stride)):
            for j in range(calc_conv_output_size(output_width, kernel_width, stride)):
                window = np.zeros((output_number, output_depth, kernel_height, kernel_height))
                max_idx = self.latest_sensitivity_map_maxidx[:, :, i, j].reshape(output_number*output_depth, -1)
                max_idx = max_idx.reshape(max_idx.shape[0])
                window_reshape = window.reshape(window.shape[0]*window.shape[1], -1)
                window_reshape[np.arange(max_idx.size), max_idx] = sens_map[:, :, i, j].reshape(window.shape[0]*window.shape[1])
                output_data[:, :, i*stride : i*stride+kernel_height, j*stride : j*stride+kernel_width] = window
        return output_data

    def forward(self, input_signal, forward_config, *args):
        trace = forward_config['trace']
        if trace:
            self.latest_input_signal_shape = input_signal.shape
        output_signal = self.pooling_forward(input_signal)
        return output_signal

    def update_gradient(self, input_sensitivity_map):
        self.latest_sensitivity_map = self.pooling_backward(input_sensitivity_map)

    def get_sensitivity_map(self):
        return self.latest_sensitivity_map

    def backward(self, input_sensitivity_map, backward_config, *args):
        self.update_gradient(input_sensitivity_map)
        return self.get_sensitivity_map()
