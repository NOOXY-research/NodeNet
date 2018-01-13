# nodenet/io/neuralnet.py
# Description:
# "neuralnet.py" provide access between file and neuralnet.
# Copyright 2018 NOOXY. All Rights Reserved.

from .commons import *

def save_neuralnet(neuralnet, filename):
    neuralnet.clear_cache()
    return dopickle(neuralnet, filename+'.nodenet')

def load_neuralnet(filename):
    return unpickle(filename+'.nodenet')
