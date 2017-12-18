# nodenet/utilities/dataprocessing.py
# Description:
# "dataprocessing.py" provide dataprocessing of data.
# Copyright 2018 NOOXY. All Rights Reserved.

from nodenet.imports.commons import *

def convert_tensor_3D_to_4D(data):
    new_data = np.reshape(data, (data.shape[0], 1, data.shape[-2], data.shape[-1]))
    return new_data
