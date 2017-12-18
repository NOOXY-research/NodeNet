# nodenet/learningalgorithm/data.py
# Description:
# "data.py" provide commons parameters.
# Copyright 2018 NOOXY. All Rights Reserved.

from nodenet.imports.commons import *
import numpy as np2 # For capability of cupy
import struct

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.array((np2.fromstring(f.read(), dtype=np.uint8).reshape(shape).tolist()))
