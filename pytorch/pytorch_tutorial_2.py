#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is part of a series of pytorch tutorials. During these tutorials,
we will learn about tensors (part_1), use autograd and optimizers on a minimal
example (part_2), use torch for fitting a basic model via LS (part_3), learn
about pytorchs utility functions (part_4) and train a neural network to perform
a classification task (part_5). 

This script is to showcase the basic pytorch optimization procedure. We will
define a very simple quadrativ function that we subsequently want to minimize 
using pytorchs builtin autograd functionality and its optimization module.
For this, do the following:
    1. Imports and definitions
    2. Set up loss function
    3. Optimize
    4. Illustrate results
    
The script is meant solely for educational and illustrative purposes. Written by
Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.com.
"""



"""
    1. Imports and definitions
"""


# i) Imports

import torch
import matplotlib.pyplot as plt



"""
    2. Set up loss function
"""


# i) Initial tensors

# Initialize a variable x. Here, x is a tensor with requires_grad=True which indicates to autograd 
# that it needs to compute gradients with respect to this tensor during the backward pass.
x = torch.tensor([1.0], requires_grad = True)



"""
    3. Optimize 
"""


# i) Define Adam optimizer

# Adam is a popular optimization algorithm used in deep learning models. It is 
# similar to gradient descent but better suited to batch training. Adam stands
# for adaptive moments and keeps track also of previous gradients to follow
# promising directions event when encountering small local minima. But even Adam
# will not guarantee success and you might get stuck during optimization.
# Setting up Adam requires defining the optimization variables and learning rate
optimizer = torch.optim.Adam([x], lr=0.1)      
x_history = []
loss_history = []


# ii) Optimize

for step in range(100):
    # Set the gradients to zero before starting to do backpropragation because 
    # pytorch accumulates the gradients on subsequent backward passes.
    optimizer.zero_grad()  
    
    # compute the loss function
    y = x**2
    loss = torch.abs(y)
    
    # compute the gradients
    loss.backward()
    # update the weights, record new parameters and loss
    optimizer.step()
    x_history.append(x.item())
    loss_history.append(loss.item())
    
    # print the loss value and the value of x at specifix steps
    if step % 10 == 0:
        print(f"Step {step+1}, Loss {loss.item()}, x {x.item()}")



"""
    4. Illustrate results
"""

# i) Plot history of x adjustments

plt.figure(1, dpi = 300)
plt.plot(x_history, label = 'x')
plt.plot(loss_history, label = 'loss')
plt.legend()
plt.title('Sequence of x values and loss')


# Please discuss briefly: How is any of this useful? Maybe you can already find
# some problems that benefit from formulating them in torch.
# It is worth experimenting a bit with the parameters that are passed during
# optimizer-setup. What happens when you change the learning rates and the number
# of steps for which the iterative optimization is run? Why and what are the
# practival implications?
