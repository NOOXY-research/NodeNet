# nodenet/neuralnets/container.py
# Description:
# "container.py" provide container to contain nodelayers, netlayers, and others in able to construct neuralnet.
# Copyright 2018 NOOXY. All Rights Reserved.

import nodenet.variables as var

# The simplest type of container
class SimpleContainer(object):
    #
    def __init__(self, layers=None, name='Unamed NeuralNet'):
        self.name = name
        self.layers = layers
        self.latest_output = None
        if layers is not None:
            layers[0].is_input_layer = True
            layers[-1].is_output_layer = True

    def __str__(self):
        string = ''
        string += self.name +' : \n'
        for x in self.layers:
            string += 'layer('+str(self.layers.index(x))+'): '+str(x)+'\n'
        return string

    __repr__ = __str__

    setup = __init__

    def clear_cache(self):
        for layer in self.layers:
            layer.clear_cache()

    def new_dropout(self, dropout_keep):
        for x in range(len(self.layers)-2):
            self.layers[x+1].new_dropout(dropout_keep)

    def forward(self, input_data, forwardconfig=var.forward_config):
        this_output = input_data
        for layer in self.layers:
            this_output = layer.forward(this_output, forwardconfig)
        self.latest_output = this_output
        return this_output

    def backward(self, target_data, loss_function, backwardconfig):
        sensitivity_map = loss_function(self.latest_output, target_data, derivative=True)
        for layer in reversed(self.layers):
            sensitivity_map = layer.backward(sensitivity_map, backwardconfig)
        return sensitivity_map

#
class LinkedContainer(object):
    pass
