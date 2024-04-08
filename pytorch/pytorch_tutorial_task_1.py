#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script provides a task related to the training of a simple neural network
for the purpose of classifying some timeseries. An exemplary solution is also 
provided; however the idea of the script is to use the commands and building
blocks found in the pytorch_tutorial series to assemble a solution. The goal
is for you to learn to set up a neural network, define a loss function, train
the resulting net, and investigate its performance.

For this, you will do the following:
    1. Imports and definitions
    2. Generate data
    3. Setup neural network class       <-- your task
    4. Train neural network             <-- your task
    5. Plots and illustrations
    
The script is meant solely for educational and illustrative purposes. Written by
Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.com.
"""



"""
    1. Imports and definitions
"""


# i) Imports

import torch
import numpy as np
import copy
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# ii) Definitions

n_time = 100
n_data = 100
n_classes = 2
time = torch.linspace(0,1,n_time)


# iii) Global provisions

torch.manual_seed(0)
torch.set_default_tensor_type(torch.DoubleTensor)



"""
    2. Generate data
"""


# The data consists of timeseries that feature a randomly chosen slope and some
# noise added on top. These timeseries can be interpreted as representing a 
# phenomenon of your own choosing (stock prices, deformations, workhours, ...). 
# We assign class labels depending on the slope of the timeseries.

# i) Generate and classify timeseries

timeseries = torch.zeros([n_data, n_time])
labels = torch.zeros([n_data, 1])

for k in range(n_data):
    random_slope = torch.distributions.Uniform(-2,2).sample()
    random_noise = torch.distributions.Normal(0, 0.5).sample([n_time])
    
    labels[k,0] = random_slope >=0
    timeseries[k,:] = random_slope * time + random_noise

# Create a reshaped version of the labels containing two columns with a 1 in 
# the first column indicating a label of 0 and 1 in the second column indicating
# a label of 1. This labels_reshaped matrix can be compared against the class
# probabilities that the ANN you want to train will output.
labels_reshaped = torch.hstack((1.0*(labels == 0 ), 1.0*(labels == 1)))

# Separate into train and test set
# Split the dataset into training and testing sets
train_size = int(round(0.8 * n_data))           # 80% of the data for training
test_size = n_data - train_size                 # 20% of the data for testing
timeseries_train = timeseries[0:train_size, :]
labels_reshaped_train = labels_reshaped[0:train_size, :]
timeseries_test = timeseries[train_size :, :]
labels_reshaped_test = labels_reshaped[train_size :, :]

# The rule for classifying the timeseries is not available in practice and we 
# employ it here only to have some simple synthetic dataset that we fully understand.
# From now on the rule is considered unknown and has to be learned from data. 


# Consider everything that has happened up till this line to be result of some
# black box process found in nature. Only the data is available to you.
# ----------------------------------------------------------------------------



"""
    3. Setup neural network class       <-- your task
"""

# i) Define ANN class

# To do this, you need to define a class that inherits from the neural network
# base class of torch; this is torch.nn.Module. The ANN class should be built
# using info on the dimensionality of the input, the hidden dimension, and the
# output. Try to come up with a simple ANN that maps full timeseries to the
# hidden layers and then into an output of dim [n_data, 2] where the second 
# dimension records the class probabilities of the labels for each datapoint.
# An output[0,:] = [0.9, 0.1] would therefore be interpreted as the ANN predicting
# for the first timeseries a label of zero with certainty 0.9 and a label of one
# with certainty 0.1.


# Now you need to come up with some code yourself. The comments are meant to give
# some guiderails in constructing the ANN class. It is recommended to steal 
# heavily from the previous tutorials.
#
# 1. Define your class name and inheritance
# 2. Initialize the class
# 3. Define the layers
# 4. Define the result of the forward pass
#
# The result will look a bit like the following blueprint that you need to adapt:
# class ANN(ClassToInheritFrom):
#       def __init__(self, your inputs):
#           super().__init__()
#           self.some_attribute = some_input
#
#           self.linear_layer_1 = torch.nn.Linear(dim_layer_input, dim_layer_output)
#           self.nonlinearity = torch.nn.Sigmoid()
#
#       def forward(self, x):
#           x = x.reshape([-1,appropriate_length])
#           hidden_1 = self.nonlinearity(self.linear_layer_1(x))
#           hidden_2 = ...
#           class_probs = self.nonlinearity(hidden_2)

# Write your own stuff starting from here : ----------------------------------


# YOUR OWN CODE


# Then you need to instantiate one neural network with appropriate dimensions
# by calling the class with some input arguments.


# YOUR OWN CODE


# You can now stop writing your own stuff for section 3. ---------------------

# Since training is only going to happen in section 4, it might be a bit hard
# to immediately see if you did everything correctly. You can do some initial
# sanity checks by printing the neural network and calling it on timeseries_train.
# The expected result is:
# print(classification_ann)
# ANN(
#   (fc1): Linear(in_features=SomeNumber, out_features=SomeNumber, bias=True)
#   (fc2): Linear(in_features=SomeNumber, out_features=2, bias=True)
#   (sigmoid): Sigmoid()
# classification_ann(timeseries_train)
#         tensor([[SomeNumber, SomeNumber],
#                     ...         ...
#                 [SomeNumber, SomeNumber]], grad_fn=<SigmoidBackward0>)
#
# It is quite probable that the class construction and applying the ann to the
# timeseries produces an error. Try to follow the error message and fix the bug
# - this is a great exercise.



"""
    4. Train neural network             <-- your task
"""


# To train the neural network, you have to set up an pytorch optimizer and then
# call it in a loop to have it adjust the parameters in the model by following
# the gradient of the loss function. This means, you have to implement a setup
# procedure and a training loop and requires multiple decisions as the specific
# optimizer, learning rate and the loss function measuring dissatisfaction with
# class predictions can be freely determined by you. This section is much easier
# than the previous one as no classes need to be built. But you will have to 
# consult the pytorch documentation and to find about about optimizers and losses.
# Also, you might have to come back to this section later on after some initial
# successful runs of the program to experiment with e.g. the learning rate.

# Now you need to come up with some code yourself, the comments again provide
# some structure that you can concretize to create some working code. 

# The result will look a bit like the following sequence:

# i) Set up optimizer    
# optimizer = torch.optim.SomeOptimizer(ann_parameters, lr = SomeLearningrate) 
# loss_fun = torch.nn.SomeClassificationLoss()
#
# ii) Optimize
# for step in range(NumberOfSteps):
#     # Zero the gradients
#     optimizer.zero_grad()      
# 
#     # compute loss, gradients, then optimization step
#     loss = loss_fun(classification_ann(timeseries_train), labels_reshaped_train)    
#     loss.backward()
#     optimizer.step()
# 
#     # print the loss value 
#     if step % 100 == 0:
#         print(f"Step {step+1}, Loss {loss.item()}")

# Write your own stuff starting from here : ----------------------------------

# i) Set up optimizer


# ii) Optimize


# You can now stop writing your own stuff for section 4. ---------------------



"""
    5. Plots and illustrations
"""


# i) Plot of the loss on train dataset and test dataset

fig, axs = plt.subplots(1,2,figsize = (10,5), dpi = 300)
axs[0].plot(loss_history)
axs[0].set_xlabel('Optimizer steps')
axs[0].set_ylabel('Classification loss')
axs[0].set_title('Loss on training dataset')

# We now want to see how the accuracy actually developed during the training on
# the test dataset. Normally you would not do this as the training loop should
# have 0 exposure to the test dataset but here we just want to illustrate the
# potential impact of overfitting.
axs[1].plot(loss_history_test)
axs[0].set_xlabel('Optimizer steps')
axs[0].set_ylabel('Classification loss')
axs[0].set_title('Loss on test dataset')


# ii) Plot confusion matrix
# We want to show how successful our classifier was. To this end, we plot the 
# confusion matrix which quantifies for both classes how often they were identified
# correctly and how often some type of confusion occured.
class_probs = classification_ann(timeseries_test)
_, predicted_classes = torch.max(class_probs,1)
_, true_classes = torch.max(labels_reshaped_test,1)
cm = confusion_matrix(true_classes.numpy(), predicted_classes.numpy())
# Plotting the confusion matrix
fig, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True,
            xticklabels=['Class 0', 'Class 1'],
            yticklabels=['Class 0', 'Class 1'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()



"""
    Additional tasks and questions
"""


# What type of loss did you choose and which alternatives might exist?

# Play around with the learning rate to gauge its impact on the training

# Which quantity could you modify to reduce/increase the risk of overfitting?

# Which mitigation strategies do exist to reduce overfitting?

# Introduce a new method to the ANN class, it should plot the weights in the ANN.

# If you think about image classification, what is the role of the max impact
# features in that setting? Could you imagine how they look like there?




""" 
    6. Exemplary solution 
"""


# ---------SPOILERS BELOW, TRY WITHOUT EXEMPLARY SOLUTION FIRST----------------



# For section 3: Define the neural network class and invoke an instance. 

# Neural network class
class ANN(torch.nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_output):
        super().__init__()
        # Remember: The  __init__() method defines how a class is initialized.
        # Here we declare that the first step of class initialization consists
        # in initializing the class we are inheriting from (torch.nn.Module),
        # the archetypical class from which all ANN's are just specific instances.
        # This is what the super.__init__() statement means - > Initialize the
        # superclass of this subclass.
        
        # Integrate basic data into ANN
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        
        # Lets build a transformation that maps timeseries into two class 
        # probabilities. Specify two linear layers and a nonlinearity for that.
        # First, two fully connected linear layers via torch.nn.Linear
        self.fc1 = torch.nn.Linear(dim_input, dim_hidden)
        self.fc2 = torch.nn.Linear(dim_hidden, dim_output)
        
        # Second, setup the sigmoid function as non-linearity
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        # define the forward computation on the input data x
        # first, reshape to have batch dimension on the left
        x = x.reshape([-1,self.dim_input])
        # second, compute hidden layer
        hidden = self.sigmoid(self.fc1(x))
        # third, transform from hidden to output and return
        class_probs = self.sigmoid(self.fc2(hidden))
        return class_probs

# Invoke an instance
classification_ann = ANN(n_time, 2, n_classes)

# You can investigate this ann by printing it out and looking at its attributes 
# and methods, e.g. you can show the architecture and the parameters by calling
# classification_ann or classification_ann.state_dict(), respectively.


# For section 4: Set up optimization and train classification_ann

# Define the optimizer
# We choose Adam as optimizer; it is a reliable standard choice for ANN training
# We set the learning rate to 0.001 corresponding to 1/100th of a gradient step
# for each iteration in the training loop and declare the ann parameters as the
# subjects of optimization.
optimizer = torch.optim.Adam(classification_ann.parameters(), lr=0.01) 

# Define the loss 
# We choose binary cross entropy as loss; it is a standard loss for binary 
# classification tasks
loss_fun = torch.nn.BCELoss()
loss_history_train = []
loss_history_test = []


# ii) Optimize

for step in range(1000):
    # Set the gradients to zero before starting to do backpropragation because 
    # pytorch accumulates the gradients on subsequent backward passes.
    optimizer.zero_grad()  
    
    # compute the loss function
    loss = loss_fun(classification_ann(timeseries_train), labels_reshaped_train)
    
    # compute the gradients
    loss.backward()
    # update the weights, record new parameters and loss
    optimizer.step()
    loss_history_train.append(loss.item())
    loss_history_test.append(loss_fun(classification_ann(timeseries_test), 
                                      labels_reshaped_test).item())
    
    # print the loss value and the value of x at specifix steps
    if step % 100 == 0:
        print(f"Step {step+1}, Loss {loss.item()}")





















