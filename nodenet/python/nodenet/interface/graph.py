# nodenet/interface/graph.py
# Description:
# "graph.py" provide mathematical graphing support .
# Copyright 2018 NOOXY. All Rights Reserved.

from nodenet.imports.commons import *
import matplotlib.pyplot as plt

class Figure(object):
    def __init__(self, subplot=(1, 1, 1), title='untitled'):
        self.fig, self.ax = plt.subplots(subplot[0], subplot[1])

    def plot_traing_loss(self, loss_record, index=0, title='untitled'):
        self.ax[index].set_title(title)
        self.ax[index].plot(loss_record[0])
        if loss_record[1] is not None:
            self.ax[index].plot(loss_record[1])
        else:
            self.ax[index].plot(loss_record[1])
        self.ax[index].set_xlabel('epochs')
        self.ax[index].set_ylabel('loss')
        self.ax[index].legend(['Training', 'Validation'], loc='upper right')
        plt.show(block=False)

    def plot_2D(self, xlist, ylist, index=0, title='untitled'):
        order = np.argsort(xlist)
        xs = np.array(xlist)[order]
        ys = np.array(ylist)[order]
        self.ax[index].set_title(title)
        self.ax[index].plot(xs, ys)
        self.ax[index].set_xlabel('x')
        self.ax[index].set_ylabel('y')
        plt.show(block=False)

# class DynamicFigure(object):
#     def __init__(self, subplot=(1, 1, 1), title='untitled'):
#         self.fig, self.ax = plt.subplots(subplot[0], subplot[1])
#         self.loss_record = []
#         self.loss_record_valid = []
#
#     def plot_traing_loss(self, loss, title='untitled', xlabel='loss', ylabel='epochs'):
#         self.ax.clear()
#         self.ax.set_title(title)
#         self.ax.set_xlabel('epochs')
#         self.ax.set_ylabel('loss')
#         self.loss_record_valid.append(loss[0])
#         if loss[1] is not None:
#             self.loss_record.append(loss[1])
#         self.ax.plot(self.loss_record)
#         self.ax.plot(self.loss_record_valid)
#         self.ax.legend(['Training', 'Validation'], loc='upper right')
#         time.sleep(0.1)
#         plt.show(block=False)
