#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is part of a series of pytorch tutorials. During these tutorials,
we will learn about tensors (part_1), use autograd and optimizers on a minimal
example (part_2), use torch for fitting a basic model via LS (part_3), learn
about pytorchs utility functions (part_4) and train a neural network to perform
a classification task (part_5). 

This script is to showcase how pytorch can be used to fit a model to some data
by defining and optimizing an appropriate loss function. What we are effectively
doing is fitting a line via least squares. However, we just define the model
function, declare squared deviations to be undesirable and let pytorch handle 
all of the numerics.
For this, do the following:
    1. Imports and definitions
    2. Simulate some data
    3. Define the model and loss
    4. Optimize parameters
    5. Plots and illustrations
    
The script is meant solely for educational and illustrative purposes. Written by
Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.com.
"""



"""
    1. Imports and definitions
"""


# i) Imports

import torch
import copy
import matplotlib.pyplot as plt


# ii) Definitions

n_x = 100
x = torch.linspace(0,1,n_x)



"""
    2. Simulate some data
"""


# i) Define distributional parameters

true_offset = 1
true_slope = 1
true_sigma = 0.1
alpha_true = torch.tensor([true_offset, true_slope])

mu =  alpha_true[0] + alpha_true[1] * x
sigma = true_sigma * torch.ones([n_x])

# Note in the above that torch follows very similar rules for broadcasting as numpy.
# For example, scalar*vector means that the scalar is multiplied wth each element 
# of the vector. Overall, pytorchs broadcasting is powerful and can make 
# professional code a bit hard to read. Convention is that the leftmost dimensions
# are reserved for batch data (independent samples) and the rightmost dimensions
# for tensors belonging to the same event. A stack of 100 images of size 20x20
# would typically be reshaped into a tensor of shape [100, 400]. These rules
# are important to know, because pytorch functions (e.g. neural nets) are 
# implicitly assuming this ordering.


# ii) Simulate from normal distribution

y_data = torch.normal(mean = mu, std = sigma)



"""
    3. Define the model and loss
"""


# i) Initial tensors

# Initialize a parameter vector alpha. It represents the two unknown parameters
# of the model we will try to fit: the offset and slope of a line. alpha is a 
# tensor with requires_grad=True since we need to compute gradients of the loss
# w.r.t. the parameters. We initialize alpha = [offset, slope] as all zeros and
# let the optimizer figure out the best values later on.

alpha_torch =  torch.tensor([0.0, 0.0], requires_grad = True)


# ii) Define the model function

def model(x, alpha):
    y_predicted = alpha[0] + x*alpha[1]
    return y_predicted


# iii) Define the loss function

def loss_function(y, y_predicted):
    loss = torch.sum(torch.square(y - y_predicted))
    return loss



"""
    4. Optimize parameters
"""


# i) Define Adam optimizer

# We again take Adam optimizer since it is an overall solid choice and our problem
# is really easy from a numerical standpoint. It is even convex so we can expect
# rapid convergence.
optimizer = torch.optim.Adam([alpha_torch], lr=0.1)      
alpha_history = []
loss_history = []


# ii) Optimize

for step in range(100):
    # Set the gradients to zero before starting to do backpropragation because 
    # pytorch accumulates the gradients on subsequent backward passes.
    optimizer.zero_grad()  
    
    # compute the loss function
    y_predicted = model(x, alpha_torch)
    loss = loss_function(y_data, y_predicted)
    
    # compute the gradients
    loss.backward()
    # update the weights, record new parameters and loss
    optimizer.step()
    alpha_history.append(copy.deepcopy(alpha_torch.detach()))
    loss_history.append(loss.item())
    
    # print the loss value and the value of x at specifix steps
    if step % 10 == 0:
        print(f"Step {step+1}, Loss = {loss.item()}, alpha = {alpha_torch}")


# iii) Solve analytically using classical LS

# We do this just for purposes of validation and illustration. The analytical
# result coincides with the numerical one computed by torch. Normally however,
# we employ torch to solve problems for which we cannot find an analytical 
# solution: nonlinear models, complicated neural nets, layered models of 
# probability distributions, ...

Design_mat = torch.vstack((torch.ones([n_x]), x)).T
alpha_analytical = torch.linalg.pinv(Design_mat)@y_data



"""
    5. Plots and illustrations
"""

# i) Plot history of los and alpha adjustments

alpha_sequence = torch.vstack(([alpha for alpha in alpha_history])).T

fig, axs = plt.subplots(nrows = 1, ncols = 2, dpi = 300)
axs[0].plot(loss_history, label = 'loss')
axs[0].legend()
axs[0].set_title('Sequence of loss values')
axs[0].set_xlabel('Iteration nr')
axs[0].set_ylabel('Loss')

axs[1].plot(alpha_sequence[0,:], label = 'alpha_0')
axs[1].plot(alpha_sequence[1,:], label = 'alpha_1')
axs[1].legend()
axs[1].set_title('Sequence of alpha values')
axs[1].set_xlabel('Iteration nr')


# ii) Plot data and line fit

plt.figure(num = 2, figsize = (5,5), dpi = 300)
plt.scatter(x.numpy(), y_data.detach(), label = 'data')
plt.plot(x,y_predicted.detach(), color = 'r', label = 'model predictions')
plt.title('Data and model predictions')
plt.legend()

# Print comparson analytical result and result from torch
print('alpha true = {} , alpha analytical = {}, alpha torch = {}'.format(alpha_true, alpha_analytical, alpha_torch))

# Please discuss briefly: How could you extend the problem in such a way that
# least sqaures would not help you anymore? Why would you need to make such an
# extension in practice?
