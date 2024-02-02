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
    3. Distributions and sampling
    4. Tensors as central building blocks
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

import inspect

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

# Since pyro = pytorch + probability, pyro modelbuilding is centered around 
# tensors. What we did above in pytorch with declaring some data as samples
# of a probabilistic model and then performing gradient descent to estimate
# some parameters - under the hood pyro does basically the same thing. It 
# provides convenient representations of sampling, observation, inference.
# The above problem could be solved in pyro using
#   1. pyro.distributions(...) for declaring a specific distribution
#   2. pyro.sample(... obs = True) for declaring a sample observed and 
#       preparing for inference 
#   3. pyro.plate(...) context manager for declaring independence of samples
#   4. pyro.infer.SVI(...) for setting up the optimization problem
# Finally, a training loop would be executed by calling the SVI's step() 
# functionality. The code for all of this is more complicated than in 
# pytorch due to the use of special constructs like pyro.sample() and pyro.plate()
# and we will only see the full picture in terms of model building and inference
# in tutorial_4. The reader may at this point ask why to use pyro at all but
# they may rest assured that all these extra provisions of clearly defining
#   - probability distributions
#   - sample_statements
#   - observation statements
#   - conditional independence structures
#   - probabilistically motivated loss 
# become essential and convenient in case the models get more complicated.
# When multiple probability distributions, latent random variables, and neural 
# networks are coupled with the complicated control flow of a nontrivial python
# program, writing a pytorch inference scheme to keep track of that is very hard
# while the declarations pyro forces you to do enable automatic bookkeeping.
# In short, pyro has additional declarative overhead but the coding effort 
# scales very well to complicated probabilistic models.



"""
    3. Distributions and sampling
"""


# i) Distributions in pyro

# pyro offers a wide variety of distributions that range from widely used standard
# distributions (normal distribution, uniform distribution) to very specialized 
# distributions (Von Mises 3D distribution, Half Cauchy distribution). Many
# of the multivariate distributions featuring in machine learning tasks like
# the multivariate Gaussian or the Wishart distribution are implemented as well.
# The distributions can all be found in the submodule pyro.distributions, lets
# list the available distributions.

# Iterate through pyro.distributions and check if element is distribution
distribution_classes = []
for name, obj in inspect.getmembers(pyro.distributions):
    if inspect.isclass(obj) and issubclass(obj, pyro.distributions.Distribution):
        distribution_classes.append(name)
# Print the list of distribution classes
print('This is a list of distributions supported by pyro : \n {}'.format(distribution_classes))

# Apart from these distributions, you can also construct new distributions by 
# transforming or mixing preimplemented distributions. Note some of the entries
# in the list that hint at that, e.g. TransformedDistribution, FoldedDistribution,
# or MixtureSameFamily.

# All distributions have the same basic methods available to them; they can be
# sampled from and the log probability can be computed. Apart from these common
# attributed of the pyro.distributions.Distribution class, every distribution
# has more specific properties; see e.g. the Normal distribution

# All normal distributions have some constraints on the parameters and their
# realizations lie in the reals.
normal_distributions = pyro.distributions.Normal
constraints = normal_distributions.arg_constraints
support = normal_distributions.support
print('Normal distributions have constraints \n {} \n and support \n {}'
      .format(constraints, support))

# For specific instantiations of the normal distribution, we can compute
# the log prob, the cumulative distribution, the entropy, and more. Lets do
# this for the standard normal distribution (sn)
sn_dist = pyro.distributions.Normal(loc = 0, scale = 1)
log_prob_values = sn_dist.log_prob(t_index)
cdf_values = sn_dist.cdf(t_index)
plt.plot(cdf_values)
plt.title('Cumulative distribution function of standard normal')
print(' Standard normal has entropy of {:.3f}, mean of {}, variance of {}'
      .format(sn_dist.entropy(), sn_dist.mean, sn_dist.variance) )


# ii) Shapes and rehspaing of distributions 

# The distributions all have two different type of shapes, the event_shape and
# the batch_shape. The event_shape quantifies the shape of a single realization
# of this distribution while the batch_shape quantifies how many independent 
# realizations this distribution is supposed to produce.  

# The above standard normal has trivial shapes of batch_shape = [] and 
# event_shape = [] reflecting the fact that it produces simple scalars.
sn_shape = sn_dist.shape()
sn_event_shape = sn_dist.event_shape
sn_batch_shape = sn_dist.batch_shape
print('Standard normal as defined above has trivial shape  {} = batch_shape {} + event_shape {} '
      .format(shape, event_shape, batch_shape))

# There are commands to reshape distributions, see the tutorial on tensor shapes
# and pytorch  broadcasting rules. Note that the left dimensions are reserved
# for batching and the right ones for the events. A multivariate Gaussian
# distribution whose output is a vector of 2 dimensions and which produces
# 10 independent realizations would have event_shape = 2 and batch_shape = 10 and
# sampling from it once would produce a [10,2] tensor.

mu_vec = torch.zeros([1,2])     # leftmost dim reserved for batching, rightmost dim for 2d vectors
cov_mat = torch.eye(2)          # is a 2 x 2 matrix, interpreted as cov mat for each batch

# Creating the multivariate normal distribution wiht the right dimensions hinges 
# on the broadcasting rules. If mu_vec and cov_mat are compared, the rightmost
# dimensions are compatible (2 for mu_vec and 2 x 2 for cov_mat). The leftmost
# dimension is considered the batch dimension; right now we only have one batch
# = one independent realization. 

single_mvn_dist = pyro.distributions.MultivariateNormal(loc = mu_vec, covariance_matrix = cov_mat)
single_mvn_shape = single_mvn_dist.shape()
single_mvn_batch_shape = single_mvn_dist.batch_shape
single_mvn_event_shape = single_mvn_dist.event_shape
single_mvn_event_dim = single_mvn_dist.event_dim
print('A multivariate normal distribution generating a single sample of a 2D vector'\
      'has \n shape = {} \n batch_shape = {}\n event_shape = {} \n and the index'\
      ' of the location of the event is event_dim = {}'.format(single_mvn_shape, 
        single_mvn_batch_shape, single_mvn_event_shape, single_mvn_event_dim))

# We expand the batch dimension to 10 via the command distribution.expand([10]) 
# which declares the distribution to generate 10 independent copies.
mvn_dist = mvn_dist.expand(batch_shape = [10])
mvn_shape = mvn_dist.shape()
mvn_batch_shape = mvn_dist.batch_shape
mvn_event_shape = mvn_dist.event_shape
mvn_event_dim = mvn_dist.event_dim
print('A multivariate normal distribution generating 10 samples of a 2D vector'\
      'has \n shape = {} \n batch_shape = {}\n event_shape = {} \n and the index'\
      ' of the location of the event is event_dim = {}'.format(mvn_shape, 
        mvn_batch_shape, mvn_event_shape, mvn_event_dim))


# Note that there are several ways which can be used to commnicate the batch_shape
# and event_shape to pyro. This includes choosing appropriate consistent shapes
# for the parameters, the expand() command, the to_event() command, and the 
# pyro plate context manager. In pyro, all dimensions of a distribution either
# need to be declared dependent or independent.


# iii) Sampling in pyro

# You can sample from distributions and declare samples to be observed and equal
# to some data. The first action (sampling) is a simple action that can be done
# in almost any other library as well; it produces some numbers/tensors.
# The second action (declaring a sample equal to some data) is a convenient pyro
# mechanism for computing the log_prob of a distribution and prepare pyro for
# gradient computation and thereby inference.







"""
    4. Tensors as central building blocks
"""

"""
    5. Overview pyro functionality
"""














