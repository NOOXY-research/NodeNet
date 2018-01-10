# nodenet/utilities/commons.py
# Description:
# "commons.py" provide commons utilities that can be use widely.
# Copyright 2018 NOOXY. All Rights Reserved.

from nodenet.imports.commons import *
import numpy as np2
# np2 for cupy compabable

def cut_dataset_by_ratio_ramdom(datasets, cut_ratio = 0.1):
    dimension = len(datasets[0].shape)
    valid_data_size = int(len(datasets[0])*cut_ratio)
    input_data = np2.array(datasets[0].tolist())
    output_data = np2.array(datasets[1].tolist())
    input_data_valid = np2.empty([0]+list(input_data.shape[1:len(input_data.shape)]))
    output_data_valid = np2.empty([0]+list(output_data.shape[1:len(output_data.shape)]))
    for x in range(valid_data_size):
        index = np2.random.randint(len(input_data))
        input_data_valid = np2.concatenate((input_data_valid, np2.array(([input_data[index].tolist()]))), axis=0)
        output_data_valid = np2.concatenate((output_data_valid, np2.array(([output_data[index].tolist()]))), axis=0)
        input_data = np2.delete(input_data, index, axis=0)
        output_data = np2.delete(output_data, index, axis=0)
    input_data = np.array(input_data.tolist())
    output_data = np.array(output_data.tolist())
    input_data_valid = np.array(input_data_valid.tolist())
    output_data_valid = np.array(output_data_valid.tolist())
    return [input_data, output_data, input_data_valid, output_data_valid]

def shuffle_datasets(datasets):
    a = np2.array(datasets[0].tolist())
    b = np2.array(datasets[1].tolist())
    assert len(a) == len(b)
    order = np2.random.permutation(len(a))
    return [np.array(a[order].tolist()), np.array(b[order].tolist())]

def get_mini_batch_ramdom(datasets, mini_batch_size):
    input_data = datasets[0]
    output_data = datasets[1]
    rand_range = len(input_data)-mini_batch_size
    start_index = 0
    if rand_range != 0:
        start_index = int(np.random.randint(len(input_data)-mini_batch_size))
    return [input_data[start_index:start_index+mini_batch_size], output_data[start_index:start_index+mini_batch_size]]

def get_mini_batch_ramdom2(datasets, mini_batch_size):
    dimension = len(datasets[0].shape)
    data_size = mini_batch_size
    input_data = datasets[0]
    output_data = datasets[1]
    index_list = []
    input_data_result = np.empty([0]+list(input_data.shape[1:len(input_data.shape)]))
    output_data_result = np.empty([0]+list(input_data.shape[1:len(input_data.shape)]))
    index = np.random.randint(len(input_data))
    for x in range(data_size):
        while index in index_list:
            index = np.random.randint(len(input_data))
        index_list.append(index)
        input_data_result = np.concatenate((input_data_result, [input_data[index]]))
        output_data_result = np.concatenate((output_data_result, [output_data[index]]))

    return [input_data_result, output_data_result]
