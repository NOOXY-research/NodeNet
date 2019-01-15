# nodenet/tester/fctest.py
# Description:
# "fctest.py" provide fullyconnected neuralnet testing.
# Copyright 2018 NOOXY. All Rights Reserved.

import numpy as np
import nodenet.neuralnets as nn
import nodenet.layers as layers
import nodenet.functions as f
import nodenet.trainingsessions as sessions
import nodenet.interface.graph as graph
import nodenet.utilities as util
import nodenet.interface.console as console
import nodenet.io as nnio
import nodenet.variables as var

console.logo()

# Graphing test 1
fig = graph.Figure((2, 1))
datasets = util.get_sin_1x1_datasets(2000, noise=0.1)
datasets = util.cut_dataset_by_ratio_ramdom([datasets[0], datasets[1]])
console.log('tester', 'graphing test 1...', msg_color='Green')
fig.plot_2D(datasets[0].flatten(), datasets[1].flatten(), 0, 'graph of sin(x) and training result')
console.log('tester', 'graphing 1 passed.', msg_color='Red')

# NeuralNet test
console.log('tester', 'fullyconnectednet test...', msg_color='Green')
neuralnet = nn.SimpleContainer()
layers = [
    layers.Nodes1D(1, f.linear),
    layers.FullyConnected1D(1, 16),
    layers.Nodes1D(16, f.tanh),
    layers.FullyConnected1D(16, 16),
    layers.Nodes1D(16, f.tanh),
    layers.FullyConnected1D(16, 1),
    layers.Nodes1D(1, f.linear),
]
neuralnet.setup(layers, name='tester neuralnet')
console.log('tester', str(neuralnet))
console.log('tester', 'fullyconnectednet passed.', msg_color='Red')

# Training test
console.log('tester', 'fullyconnectednet training test...', msg_color='Green')
batch_training = sessions.MiniBatchSession()
forward_config = var.forward_training_config
backward_config = var.backward_training_config
# forward_config = var.forward_dropout_training_config
# backward_config = var.backward_dropout_training_config
batch_training.setup(neuralnet, datasets, target_loss=0.00001, mini_batch_size=500, max_epoch=10000, forward_config=forward_config, backward_config=backward_config, verbose_interval=1000)
loss = batch_training.startTraining()
fig.plot_traing_loss(loss, 1)
console.log('tester', 'fullyconnectednet training test passed.', msg_color='Red')

# Graphing test 2
console.log('tester', 'graphing test 2...', msg_color='Green')
inputx = np.linspace(-10, 10, 100)
outputy = []
for x in inputx:
    outputy.append(neuralnet.forward(np.array([x]))[0])
fig.plot_2D(inputx, outputy, 0, 'training')
console.log('tester', 'graphing test 2 passed.', msg_color='Red')

# IO test
console.log('tester', 'io test...', msg_color='Green')
nnio.save_neuralnet(neuralnet, 'tester')
newneuralnet = nnio.load_neuralnet('tester')
console.log('tester', str(neuralnet))
console.log('tester', 'io test passed.', msg_color='Red')

console.log('tester', 'test passed. Press any key to escape.')
input()
