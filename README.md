# NodeNet
"NodeNet" is a project aims to bulid neural network. It's oriention is to collect plenty of types of neural networks and provide manager to organize and test it, and the neural network libary can work without needs of manager isolatedly. It is a project belongs to NOOXY. There still lots of miles to  complete it. Article about it might be established some day. Visit us www.nooxy.tk.

# The neuralnet framework NodeNet
## 0. Testers
  Testers part provides fully connected and convolutional neural net testing. Besides, it also test NodeNet’s other functions like graphing.

## 1. Variables
  Variables part provides default parameters for each gradient optimizer, backward configuration, forward configuration, default files path and framework version etc.

## 2. Utilities
  Utilities part provides commons utilities that can be used widely, such as shuffling datasets, getting mini batch randomly or generating dataset for testing like sin function or MNIST handwriting datasets. Even data preprocessing abilities will be added in future.

## 3. Training session
  We only provide mini batch training session now. You can adjust lots of parameters like batch size, target loss, training profile and dropout keep rate etc.

## 4. Neuralnet container
  It provides container to contain nodelayers, netlayers, convolutional layers and others. Which constructs neural net.

## 5. Models
  It provides pre design neural net with already determited activation functions, layers.

## 6. Gradient optimizer
  It provides different backpropagation optimization, such as momentum, nesterov momentum, adagrad, adadelta, rmsprop and adam.

## 7. IO
  IO provides access between neuralnet file and neuralnet objects. And it also gives access of datasets.

## 8. Layers
  It provides nodes, convolutional, fully connected nets, pooling layers and convertor layers.

## 9. Interface
  We provide graphing and console interface to graph training loss, graph neural net output and colored console log  and provide console input interface.

## 10. Functions
  Functions part provides vary types of activation functions like RELU, softmax, sigmoid. And loss functions like MSE, cross entropy error.

## 11. Apps api
  This part we haven’t finished it yet.

## 12. Neuralnet manager (only supported in oldverion, should be rebuild on App api )
  Providing friendly console interface to access NodeNet’s neuralnet. Which can build neuralnet basiclly by command.
