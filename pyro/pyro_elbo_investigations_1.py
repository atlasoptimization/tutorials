#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is part of a series of investigations to develop an intuition on how
to interpret the evidence lower bound and disentangling its impact on evidence
maximization and KLD minimization. We will try to find out, how model fitting and
adjustment of the variational distribution towards the posterior interact with
each other. The investigation will extend to cover multiple phenomena that deal
with the interpretation, computation, and disentanglement of the loss.
The investigations consist in the following:
    - Tie elbo to maximum likelhood for simple mean estimation
    - Tie elbo to maximum likelihood for simple std estimation
    - Tie elbo to posterior density approximation for simple mean estimation
    - Understand elbo when params in model and guide
    - Understand elbo in multivariate context
    - Understand elbo in complex model involving ANN's'
You will find these investigations in individual scripts, each focusing on the
illumination of one specific phenomenon.

In this script, we showcase how the elbo loss can be interpreted as a maximum 
likelihood estimator in the simple context of estimating the mean parameter of
a normal distribution. We compare the value of the elbo loss to the penalty
function representing maximum likelihood and explain the differences.
For this, do the following:
    1. Definitions and imports
    2. Simulate some data
    3. Define the model
    4. Define the guide
    5. Perform inference
    6. Investigate and compare
    7. Plots and illustrations

The script is meant solely for educational and illustrative purposes. Written by
Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.com.
"""


"""
    1. Definitions and imports
"""


# i) Imports

import numpy as np
import torch
import pyro

import matplotlib.pyplot as plt


# ii) Definitions

np.random.seed(0)
torch.manual_seed(0)

n_data = 1
interval = torch.linspace(-2,5,500)



"""
    2. Simulate some data
"""

# i) Define data distribution

mu_true = torch.tensor(1.0)
sigma_true = torch.tensor(1.0)

data_dist = pyro.distributions.Normal(mu_true, sigma_true)


# ii) Sample from distribution

data = data_dist.sample([n_data])



"""
    3. Define the model
"""

# i) Define the model
#
# Easiest model possible. A single parameter mu that acts as the mean of a normal
# distribution. No latent variables and no further unknown variables.

def model(observations = None):
    # Set up parameter mu
    mu = pyro.param('mu', init_tensor = torch.tensor(0.0))
    
    # Set up observation distribution, take independent samples from it
    obs_dist = pyro.distributions.Normal(mu, sigma_true)
    with pyro.plate('batch_plate', size = n_data, dim =-1):
        obs = pyro.sample('obs', obs_dist, obs = observations)

    return obs


"""
    4. Define the guide
"""


# i) Define the guide
#
# There are not unobserved latent variables so the guide is trivial. No latents
# implies no variational distribution is needed and no need for anaapproximation 
# procedure for aligning variational distribution and posterior density

def guide(observations = None):
    pass


"""
    5. Perform inference
"""


# i) Set up optimization

# Optimization options
adam = pyro.optim.Adam({"lr": 0.01})
elbo = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(model, guide, adam, elbo)


# ii) Run Optimization

loss_sequence = []
n_steps = 1000
for step in range(n_steps):
    loss = svi.step(data)
    loss_sequence.append(loss)
    
    # Print out loss
    if step % 100 == 0:
        print('epoch: {} ; loss : {}'.format(step, loss))
        


"""
    6. Investigate and compare
"""



"""
    7. Plots and illustrations
"""





