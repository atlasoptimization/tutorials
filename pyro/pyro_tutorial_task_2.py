#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script provides tasks related to the estimation of mean and variance of a
distribution that was only observed indirectly. A simple univariate gaussian
distribution with unknown mean and variance is sampled thereby yielding values
z_1, ..., z_m. These values themselves are considered hidden variables that are
measured noisily thereby resulting in the observed data used to perform inference.
An exemplary solution is also provided; however the idea of the script is to use
the commands and building blocks found in the pyro_tutorial series to assemble 
a solution. The goal is for you to learn to set up a stocahstic model, define an
inference scheme, and fit the model to some data.

For this, you will do the following:
    1. Imports and definitions
    2. Generate data
    3. Setup stochastic model       <-- your task
    4. Fit model to data
    5. Plots and illustrations
    
For the purpose of this task, think of a machine producing elements of length z
that are then subsequently measured with some unbiased measurement device. The 
production process is random with unknown mu_prod, sigma_prod and each one of 
the elements has a unique unobserved length z_1 , ... z_m. Measuring an element
of length z_1 yields observations z_11, ..., z_1n that depend on the value of z_1
via z_1j ~ N(z_1, 1) j=1,...n; i.e. the observation is the length + white noise.
From these observations, infer the expected value and the spread of the production
process.
    
The script is meant solely for educational and illustrative purposes. Written by
Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.com.
"""



"""
    1. Imports and definitions
"""


# i) Imports

import torch
import pyro
import copy

import matplotlib.pyplot as plt


# ii) Definitions

torch.manual_seed(0)
n_elements = 5
n_meas = 20



"""
    2. Generate data
"""


# The data consists of a bunch of numbers but they are not all i.i.d. Instead,
# latent variables z_1, ..., z_n are sampled from a distribution N(mu_prod, sigma_prod)
# The latent variables are true unobserved element length and them being random
# represents the model assumption that there is some randomness involved in the 
# element production process. The element lengths are then measured by a device
# that produces a noisy observation = element length + white noise.
#
# Overall, this introduces randomness on multiple levels - the production process
# and the measurement process.


# i) Set up ground truth parameters

# Elements are 1 m long and production errors have standard deviation of 5 cm
mu_prod_true = 1
sigma_prod_true = 0.05
sigma_meas = 0.01


# ii) Define production distribution and sample

production_distribution = torch.distributions.Normal(loc = mu_prod_true,
                                                     scale = sigma_prod_true)
element_lengths_true = production_distribution.sample([n_elements])


# iii) Define production distribution and sample

measurement_distribution = torch.distributions.Normal(loc = element_lengths_true, 
                                                      scale = sigma_meas)
dataset_measurements = measurement_distribution.sample([n_meas]).T


# Consider everything that has happened up till this line to be result of some
# black box process found in nature. Only the data is available to you.
# ----------------------------------------------------------------------------



"""
    3. Set up stochastic model      <--- your task
"""


# You now want to build your stochastic model in pyro. This requires setting up
# a function model() in which parameters, sampling, and independence statements
# are combined to reflect the model you want to fit.
# 
# The model you want to fit is a simple univariate normal distribution with un-
# known mean and variance. From this distribution, n_data samples are taken and
# for these samples we want to plug in the numbers in "dataset".
#
# The model() function should take as input potentially a bunch of numbers and 
# condition on them, alternatively just simulate some new numbers by running
# all the commands and returning the samples drawn from a probability distribution.
# This means, your model should look something like this:
#
# def model(observations = None):
#     # Declare parameters
#     mu = pyro.param( arguments_for_pyro_param)
#     sigma = pyro.param( arguments_for_pyro_param )
#
#     # Declare distribution and sample n_data independent samples, condition on observations
#     model_dist = pyro.distributions.your_distribution( arguments_for_distribution )
#     with pyro.plate( arguments_for_pyro_plate ):    
#         model_sample = pyro.sample( arguments_for_pyro_sample, obs = observations)
    
#     return model_sample
#
# In the model, you make use of the following commands:
# pyro.param() : Define a parameter that will be optimized during training
# pyro.distributions.Normal() : Define a normal distribution
# pyro.plate() : Declare that independend sampling happens
# pyro.sample() : Declare some numbers as coming from a distribution
#
# Write your own stuff starting from here : ----------------------------------


# YOUR OWN CODE


# Then you need to instantiate one neural network with appropriate dimensions
# by calling the class with some input arguments.


# YOUR OWN CODE


# You can now stop writing your own stuff for section 3. ---------------------



"""
    4. Fit model to data
"""


# i) Set up Optimization options

n_iter = 1000
adam = pyro.optim.Adam({"lr": 0.01})
elbo = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(model, guide, adam, elbo)


# ii) Inference and print results

loss_sequence = []
mu_est_sequence = []
sigma_est_sequence = []

# Iterate and train
for step in range(n_iter):
    loss = svi.step(dataset_measurements)
    
    # Attach results
    loss_sequence.append(loss)
    mu_est_sequence.append(copy.copy(pyro.get_param_store()['mu_prod'].detach().numpy()))
    sigma_est_sequence.append(copy.copy(pyro.get_param_store()['sigma_prod'].detach().numpy()))
    
    # Print out loss
    if step % 100 == 0:
        print('epoch: {} ; loss : {}'.format(step, loss))
    else:
        pass




"""
    5. Plots and illustrations
"""


# i) Plot training progress

# evolution of loss
iter_steps = torch.linspace(0, n_iter-1, n_iter)
plt.figure(1, dpi = 300)
plt.plot(iter_steps, loss_sequence)
plt.title('Loss over iteration nr.')
plt.xlabel('Iterations')

# print out inferred parameters
print('The true mean and std of the production process are {:.3f}, {:.3f} . \n'
      'The inferred ones by pyro are {:.3f}, {:.3f}'
      .format(mu_prod_true, sigma_prod_true, pyro.get_param_store()['mu_prod'],
              pyro.get_param_store()['sigma_prod']))

# evolution of parameter estimates
fig, axs = plt.subplots(1,2, figsize = (10,5), dpi = 300)
axs[0].plot(iter_steps, mu_est_sequence, label = 'estimated mu')
axs[0].plot(iter_steps, mu_prod_true*torch.ones([n_iter]), label = 'true mu')
axs[0].set_title('Estimated mu')
axs[0].set_xlabel('Iterations')
axs[0].legend()
    
axs[1].plot(iter_steps, sigma_est_sequence, label = 'estimated sigma')
axs[1].plot(iter_steps, sigma_prod_true*torch.ones([n_iter]), label = 'true sigma')
axs[1].set_title('Estimated sigma')
axs[1].set_xlabel('Iterations')
axs[1].legend()


# ii) Plot posterior distributions

n_disc = 100
x_disc = torch.linspace(0.9, 1.1, n_disc)
n_illu = 5 if 5 <= n_elements else n_elements

# Inferences from pyro
post_densities = torch.zeros([n_illu, n_disc])
mu_post = pyro.get_param_store()['mu_prod_post'].detach()
sigma_post = pyro.get_param_store()['sigma_prod_post'].detach()

# Inferences from solving by hand
empirical_densities = torch.zeros([n_illu, n_disc])
empirical_means = torch.mean(dataset_measurements, dim = 1)
empirical_stds = torch.std(dataset_measurements, dim = 1)/torch.sqrt(torch.tensor(n_meas))

# Build densities
for k in range(n_illu):
    temp_density_pyro = torch.exp(pyro.distributions.Normal(mu_post[k], sigma_post[k]).log_prob(x_disc))
    temp_density_empirical = torch.exp(pyro.distributions.Normal(empirical_means[k], empirical_stds[k]).log_prob(x_disc))
    post_densities[k,:] = temp_density_pyro
    empirical_densities[k,:] = temp_density_empirical
    
fig, axs = plt.subplots(2,1, figsize = (5,10), dpi =300)
axs[0].plot(x_disc.repeat([n_illu,1]).T, post_densities.T)
axs[0].set_xlabel('length in m')
axs[0].set_title('posterior densities by pyro')

axs[1].plot(x_disc.repeat([n_illu,1]).T, empirical_densities.T)
axs[1].set_xlabel('length in m')
axs[1].set_title('posterior densities by hand')





"""
    Additional tasks and questions
"""

# Interpret the output of the model inspection with pyro.poutine.trace.
# You can use: print(pyro.poutine.trace(model).get_trace(None).format_shapes())
#         and: print(pyro.poutine.trace(model).get_trace(None).nodes)





""" 
    6. Exemplary solution 
"""


# ---------SPOILERS BELOW, TRY WITHOUT EXEMPLARY SOLUTION FIRST----------------



# # For section 3: Define the model and guide functions 


# i) Define the model

def model(observations = None):
    # Declare parameters
    mu_prod = pyro.param("mu_prod", init_tensor = torch.tensor(0.0))
    sigma_prod = pyro.param("sigma_prod", init_tensor = torch.tensor(1.0),
                            constraint = pyro.distributions.constraints.positive)
    
    # Define production distribution & sample independently
    extension_tensor_1 = torch.ones([n_elements,1])
    prod_dist = pyro.distributions.Normal(loc = mu_prod * extension_tensor_1,
                                          scale = sigma_prod)
    
    with pyro.plate("element_plate", size = n_elements, dim = -2):
        element_lengths = pyro.sample("element_lengths", prod_dist)
        
        # define measurement distribution and sample for each element multiple times
        extension_tensor_2 = torch.ones([1,n_meas])
        meas_dist = pyro.distributions.Normal(loc = element_lengths * extension_tensor_2,
                                              scale = sigma_meas)
        with pyro.plate("measurement_plate", size = n_meas, dim = -1):
            measurements = pyro.sample("measurements", meas_dist, obs = observations)
            
    return measurements


# ii) Define the guide

# Variational distribution
def guide(observations = None):
    # Declare parameters (n_elements univariate normal posterior distributions)
    mu_prod_post = pyro.param("mu_prod_post", init_tensor = torch.zeros([n_elements,1]))
    sigma_prod_post = pyro.param("sigma_prod_post", init_tensor = torch.ones([n_elements,1]))
    
    # Proposal distribution for posterior
    prod_post_dist = pyro.distributions.Normal(loc = mu_prod_post,
                                               scale = sigma_prod_post)
    
    # sampling the latent variable element_lengths
    with pyro.plate('element_plate', size = n_elements, dim = -2):
        element_lengths = pyro.sample("element_lengths", prod_post_dist)

    return element_lengths


