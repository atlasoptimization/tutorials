#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is part of a series of pyro tutorials. During these tutorials, we
will tackle a sequence of problems of increasing complexity that will take us
from understanding simple pyro concepts to fitting models via svi and finally
fitting a model involving hidden variables and neural networks.
The tutorials consist in the following:
    - key bayesian concepts                 (tutorial_0) 
    - first contact with pyro               (tutorial_1)
    - forward model and diagnosis           (tutorial_2)
    - data generation and analysis          (tutorial_3)
    - fitting simple model                  (tutorial_4)
    - fitting multivariate model            (tutorial_5)
    - fitting model with hidden variables   (tutorial_6)
    - fitting model with neural networks    (tutorial_7)

This script will initiate first contact with the probabilistic programming language
pyro. We will showcase the differences between numpy, pytorch, and pyro, get an
overview of different modules of pyro and sample from some distribution.
For this, do the following:
    1. Imports and definitions
    2. Comparison, numpy, pytorch, pyro
    3. Tensors as central building blocks
    4. Distributions and sampling
    5. Overview pyro functionality
    
The script is meant solely for educational and illustrative purposes. Written by
Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.com.
"""

"""
    1. Imports and definitions
"""


# i) Imports

import numpy as np
import torch
import pyro

import matplotlib.pyplot as plt


# ii) Definitions




"""
    2. Comparison, numpy, pytorch, pyro
"""


# Lets start off by comparing numpy, pytorch, and pyro. While pyro is the main
# focus of this tutorial, it is good to understand its relations to the other
# two, more fundamental libraries. numpy is a library for basic numeric computation
# providing the array classes and associated methods. numpy arrays are collection
# of numbers without much context. pytorch, in comparison, is a library for deep
# deep learning and is built around the tensor class. pytorch tensors are a 
# collection of numbers together with a history of where they came from thereby
# enabling gradient computation. Finally, pyro is a library for enabling deep
# learning in a probabilistic context by augmenting pytorch with probabilistic
# concepts like probability distributions and sampling.
# In short:
#   numpy = numbers
#   pytorch = numbers + history
#   pyro = pytorch + probability


# i) numpy arrays and methods

# Define an array x, showcase the arrays sum() method and showcase the numerical
# library of numpy containing functions that can be called on arrays. 
x = np.array([[0,1],[2,3]])
x_sum = x.sum()
y = np.square(x_sum)

# print arrays to console
print('\n numpy array generation -----------------------------------------------')
print( ' x = \n {}\n x_sum = \n {}\n y = \n {}'.format(x, x_sum, y))


# Apart from the standard analytical functions, there exists a submodule called
# numpy.random that allows to sample from probability distributions. However, 
# the normal distribution itself is not a function in numpy. You can write it
# and use it to compute the probability density values but overall this makes
# it clear that the ecosystem in numpy is not geared towards complicated 
# statistical problems and their analysis.

# Use numpy to compute and sample from standard normal.
sample_z = np.random.normal(loc = 0, scale = 1, size = [1000,1])
def standard_normal_pdf(index_variable):
    return (1/np.sqrt(2*np.pi))*np.exp(-index_variable**2/2)
index = np.linspace(-3,3,100)
distribution_z = standard_normal_pdf(index)

# Plot the distribution and the random data
fig, axs = plt.subplots(2, 1, figsize=(8, 10))
axs[0].plot(np.linspace(-3,3,100), distribution_z)
axs[0].set_title('Probability density')
axs[1].hist(sample_z, bins=20)
axs[1].set_title('Histogram of sample_z')
plt.tight_layout()
plt.show()


# ii) pytorch tensors and methods

# Define a tensor x, showcase the tensors sum() methods and showcase the library
# of differentiable functions built into torch as well as some of the methods
# available to compute gradients. Notice that in order to keep track of the
# history of a tensor, only torch functions may be applied to tensors. The
# requires_grad = True argument advises pytorch to keep track of the gradients.

# Create tensors and dependent values
t_x = torch.tensor([[0.0,1.0],[2.0,3.0]], requires_grad = True)
t_x_sum = t_x.sum()
t_y = torch.square(t_x_sum)

# Compute gradients
gradient_function_1 = t_y.grad_fn
gradient_function_2 = t_y.grad_fn.next_functions
t_y.backward()
dy_dx = t_x.grad

# print tensors and grads to console
print('\n torch tensor vals and grads ------------------------------------------')
print( ' t_x = \n {}\n t_y = \n {} \n gradient_functions = \n {} \n {} \n dy_dx = \n {}'
      .format(t_x, t_y, gradient_function_1, gradient_function_2[0][0], dy_dx))


# Compared to numpy, pytorch's support for probability distributions is more 
# extensive. In pytorch, probability distributions exist as differentiable 
# functions and it is also possible to sample from them. However, there is no
# native support for tying samples to probability distributions and then 
# differentiate the distribution w.r.t. the sample. Although this can be done
# manually, making this type of task more convenient is what pyro was made for.

# Use torch to compute and sample from the standard normal.
t_standard_normal = torch.distributions.Normal(loc = 0, scale = 1)
t_index = torch.linspace(-3,3,100)
t_distribution_z = torch.exp(t_standard_normal.log_prob(t_index))
t_sample_z = t_standard_normal.sample(sample_shape = [10,1])

# We could plot the the samples and the distribution here but there is nothing
# new to see here compared to the numpy samples and distribution. However, since
# pytorch allows for the computation of gradients, we can compute the
# gradients of the probability w.r.t. the mean mu. numpy does not allow this.


# Compute log probability of sample vector given mean parameter t_mu
t_mu = torch.tensor(0.0, requires_grad =True)
t_mu_normal = torch.distributions.Normal(loc = t_mu, scale = 1)
t_mu_log_prob = t_mu_normal.log_prob(t_sample_z)
t_mu_log_prob_sum = t_mu_log_prob.sum()

# Backward pass to compute gradients
t_mu_log_prob_sum.backward()
t_mu_grad = t_mu.grad

# Using the above relationships we can now call an optimizer to adapt the mean
# to maximize the probability of t_sample_z. This corresponds to maximum
# likelihood estimation. The computation is not very convenient in pytorch, 
# pyro was designed to handle this in an easier fashion. Here is how you would
# do it in pytorch:

optimizer = torch.optim.Adam([t_mu], lr=0.1)      
for step in range(100):
    # Set the gradients to zero
    optimizer.zero_grad()  
    
    # compute the loss function
    loss = -1*torch.distributions.Normal(loc = t_mu, scale = 1).log_prob(t_sample_z).sum()
    
    # compute the gradients, update t_mu
    loss.backward()
    optimizer.step()
    
    if step % 10 == 0:
        print(f"Step {step+1}, Loss {loss.item()}, t_mu {t_mu.item()}")
print(" Optimization results t_mu = {} \n Analytical ML estimate t_mu = {}".
      format(t_mu.item(), torch.mean(t_sample_z)))



# iii) Special pyro constructs





"""
    3. Tensors as central building blocks
"""


"""
    4. Distributions and sampling
"""


"""
    5. Overview pyro functionality
"""