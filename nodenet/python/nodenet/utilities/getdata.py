# nodenet/utilities/getdata.py
# Description:
# "getdata.py" provide generated data for testing.
# Copyright 2018 NOOXY. All Rights Reserved.

from nodenet.imports.commons import *
import random
import nodenet.variables as var
import nodenet.io as nnio
from . import dataprocessing

def extract_mnist_output(data):
    output_data = np.zeros((data.shape[0], 10))
    for x in range(data.shape[0]):
        output_data[x, data[x]] = 1
    return output_data

def get_sin_1x1_datasets(datasize, noise=0):
    input_data = []
    output_data = []
    input_data_valid = []
    output_data_valid = []

    for x in range(0, datasize):
        randint = random.randint(-50000, 50000)
        while randint in input_data:
            randint = random.randint(-50000, 50000)
        input_data.append([(randint/50000)*np.pi*2])

    input_data_np = np.array(input_data)+np.random.uniform(-noise, noise, np.array(input_data).shape)
    output_data_np = np.sin(input_data)+np.random.uniform(-noise, noise, np.array(input_data).shape)

    return [input_data_np, output_data_np]

def get_mnist_datasets_old():
    input_data_train = nnio.read_idx(var.datasets_path+'MNIST/train-images-idx3-ubyte/data')
    output_data_train = nnio.read_idx(var.datasets_path+'MNIST/train-labels-idx1-ubyte/data')
    input_data_valid = nnio.read_idx(var.datasets_path+'MNIST/t10k-images-idx3-ubyte/data')
    output_data_valid = nnio.read_idx(var.datasets_path+'MNIST/t10k-labels-idx1-ubyte/data')
    # Convert it to 4D tensor
    input_data_train = dataprocessing.convert_tensor_3D_to_4D(input_data_train)
    input_data_valid = dataprocessing.convert_tensor_3D_to_4D(input_data_valid)
    output_data_train = extract_mnist_output(output_data_train)
    output_data_valid = extract_mnist_output(output_data_valid)
    return [input_data_train, output_data_train, input_data_valid, output_data_valid]

def get_mnist_datasets():
    raw = nnio.unpickle(var.datasets_path+'MNIST/data')
    # np.array fo cupy capable
    datasets = []
    datasets.append(np.array(raw[0].tolist()))
    datasets.append(np.array(raw[1].tolist()))
    datasets.append(np.array(raw[2].tolist()))
    datasets.append(np.array(raw[3].tolist()))
    return datasets
