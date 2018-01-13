# nodenet/layers/base.py
# Description:
# "base.py" provide layer template.
# Copyright 2018 NOOXY. All Rights Reserved.

from nodenet.imports.commons import *

class Layer(object):
    def __init__(self):
        # For capability
        self.is_input_layer = False
        self.is_output_layer = False

    def __str__(self):
        string = ''
        string += 'Layer Object'
        return string

    __repr__ = __str__

    setup = __init__
    def clear_cache(self):
        pass

    def new_dropout(self, dropout_keep):
        pass

    def forward(self, input_signal, forward_config, *args):
        pass

    def backward(self, input_sensitivity_map, backward_config, *args):
        pass
