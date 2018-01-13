# nodenet/trainingsessions/batch.py
# Description:
# "batch.py" provide batch training session.
# Copyright 2018 NOOXY. All Rights Reserved.

from nodenet.imports.commons import *
import nodenet.interface.console as console
import nodenet.functions as func
import nodenet.variables as var
import nodenet.utilities as util
import nodenet.io as nnio
import math

class MiniBatchSession(object):
    def __init__(self, neuralnet=None, datasets=None, target_loss=None ,mini_batch_size=None, forward_config=var.forward_training_config, backward_config=var.backward_training_config, loss_function=func.mean_square, max_epoch=None, verbose=True, verbose_interval=100, backup_intervel=1000, backup_path='./backup/'):
        self.neuralnet = neuralnet
        self.input_data = None
        self.output_data = None
        self.input_data_valid = None
        self.output_data_valid = None
        self.target_loss = target_loss
        self.mini_batch_size = mini_batch_size
        self.forward_config = forward_config
        self.backward_config = backward_config
        self.loss_function = loss_function
        self.max_epoch = max_epoch
        self.iterations_each_epoch = None
        self.verbose = verbose
        self.verbose_interval = verbose_interval
        self.backup_intervel = backup_intervel
        self.backup_path = backup_path
        if backup_intervel > 0:
            nnio.makedirs(backup_path)

        if datasets is not None:
            if len(datasets) <= 2:
                self.input_data = datasets[0]
                self.output_data = datasets[1]
            else:
                self.input_data = datasets[0]
                self.output_data = datasets[1]
                self.input_data_valid = datasets[2]
                self.output_data_valid = datasets[3]
            self.iterations_each_epoch = int(len(datasets[0])/mini_batch_size)

        if self.max_epoch is not None:
            self.max_iteration = self.max_epoch*self.iterations_each_epoch
        else:
            self.max_iteration = None

    def __str__(self):
        string = ''
        string = 'neuralnet : '+str(self.neuralnet)+', target loss : '+str(target_loss)+', mini batch size : '+str(self.mini_batch_size)+', max epoch : '+str(self.max_epoch)+', cost function : '+str(loss_function)
        return string

    __repr__ = __str__

    setup = __init__

    def dumpLog(self, epochs, batch_train_loss, valid_loss):
        console.log('training', 'epochs: '+str(epochs)+', batch_train_loss: '+str(batch_train_loss)+', valid_loss: '+str(valid_loss))

    def startTraining(self):
        iterations_sum = 0
        loss_record_train = [] # each epoch
        loss_record_valid = []
        latest_loss = 99999
        max_iteration = math.inf

        if self.max_iteration is not None:
            max_iteration = self.max_iteration

        console.log('training', 'start training session...')
        console.log('training', str(len(self.input_data))+' training datasets. '+str(len(self.input_data_valid))+' validation datasets.')
        console.log('training', 'input shape: '+str(self.input_data.shape)+', output shape: '+str(self.output_data.shape))
        console.log('training', 'valid input shape: '+str(self.input_data_valid.shape)+', valid output shape: '+str(self.output_data_valid.shape))
        console.log('training', 'mini batch size: '+str(self.mini_batch_size))
        while(iterations_sum <= max_iteration and latest_loss > self.target_loss):
            # Do forward first reduce computation
            this_batch_input, this_batch_output = util.get_mini_batch_ramdom([self.input_data, self.output_data], self.mini_batch_size)

            # Do record of loss
            if iterations_sum%self.iterations_each_epoch == 0:
                # Do backup
                if (iterations_sum/self.iterations_each_epoch)%self.backup_intervel == 0:
                    nnio.save_neuralnet(self.neuralnet, self.backup_path+self.neuralnet.name)
                self.neuralnet.new_dropout(self.forward_config['dropout_keep'])
                train_loss = np.mean(self.loss_function(self.neuralnet.forward(this_batch_input, var.forward_config), this_batch_output))
                valid_loss = np.mean(self.loss_function(self.neuralnet.forward(self.input_data_valid), self.output_data_valid))
                loss_record_train.append(train_loss)
                loss_record_valid.append(valid_loss)
                if (iterations_sum/self.iterations_each_epoch)%self.verbose_interval == 0 and self.verbose:
                    self.dumpLog(iterations_sum/self.iterations_each_epoch, loss_record_train[-1], loss_record_valid[-1])

            self.neuralnet.forward(this_batch_input, self.forward_config)
            self.neuralnet.backward(this_batch_output, self.loss_function, self.backward_config)

            iterations_sum += 1
            # latest_loss = loss_record_train[-1]
            latest_loss = 999

        return loss_record_train, loss_record_valid
