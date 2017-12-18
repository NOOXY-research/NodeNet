# nodenet/utilities/commons.py
# Description:
# "commons.py" provide commons utilities that can be use widely.
# Copyright 2018 NOOXY. All Rights Reserved.

from nodenet.imports.commons import *

def cut_dataset_by_ratio_ramdom(datasets, cut_ratio = 0.1):
    dimension = len(datasets[0].shape)
    valid_data_size = int(len(datasets[0])*cut_ratio)
    input_data = datasets[0]
    output_data = datasets[1]
    input_data_valid = np.empty([0]+list(input_data.shape[1:len(input_data.shape)]))
    output_data_valid = np.empty([0]+list(input_data.shape[1:len(input_data.shape)]))
    for x in range(valid_data_size):
        index = np.random.randint(len(input_data))
        input_data_valid = np.concatenate((input_data_valid, [input_data[index]]))
        output_data_valid = np.concatenate((output_data_valid, [output_data[index]]))
        input_data = np.delete(input_data, index, axis=0)
        output_data = np.delete(output_data, index, axis=0)
    return [input_data, output_data, input_data_valid, output_data_valid]

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
