# nodenet/tester/cnntest.py
# Description:
# "cnn.py" provide convolutional neuralnet testing.
# Copyright 2018 NOOXY. All Rights Reserved.

import nodenet.variables as var
import nodenet.utilities as util
import nodenet.layers as layers
import nodenet.neuralnets as nn
import nodenet.trainingsessions as sess
import nodenet.functions as func
layers = [
    layers.Conv2D(28, 28, 1, 5, 5, 32),
    layers.Nodes2D(24, 24, 32, func.ReLU),
    layers.MaxPool2D(24, 24, 32, 2, 2, 2),
    layers.Conv2D(12, 12, 32, 5, 5, 64),
    layers.Nodes2D(8, 8, 64, func.ReLU),
    layers.MaxPool2D(8, 8, 64, 2, 2, 2),
    layers.TensortoVector(),
    layers.Nodes1D(1024, func.ReLU),
    layers.FullyConnected1D(1024, 10),
    layers.Nodes1D(10, func.softmax1D)
]
neuralnet = nn.SimpleContainer(layers)
datasets = util.get_mnist_datasets()
session = sess.MiniBatchSession(neuralnet, datasets, 0.1, mini_batch_size=1000, verbose_interval=1, loss_function=func.softmax_cross_entropy, forward_config=var.forward_dropout_training_config, backward_config=var.backward_dropout_training_config)
session.startTraining()

console.log('tester', 'test passed. Press any key to escape.')
input()
