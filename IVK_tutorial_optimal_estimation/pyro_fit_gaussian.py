#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script aims to illustrate basic pyro functionality by optimally estimating
the mean parameter of a Gaussian distribution based on measurements. This 
represents one of the simplest possibilities of doing inference and provides
a bit of insight into naming conventions, shapes, and typical commands.
For this, do the following:
    1. Imports and definitions
    2. Simulate some data
    3. Build stochastic model 
    4. Statistical inference
    5. Plots and ilustrations
"""



"""
    1. Imports and definitions ------------------------------------------------
"""


# i) Imports

import numpy as np
import torch
import pyro
# from pyro.infer import SVI



# ii) Definitions

n_observations = 10

pyro.clear_param_store()



"""
    2. Simulate some data -----------------------------------------------------
"""


# i) Draw from a Gaussian 

mu_true = 0         # True mean parameter - later assumed unknown
sigma_true = 1      # True variance parameter - later assumed known

observations = np.random.normal(mu_true,sigma_true, n_observations) # Draw numbers from distribution
observations = torch.tensor(observations)       # Convert numbers to tensors



"""
    3. Build stochastic model -------------------------------------------------
"""


# i) Function taking as inputs observations and linking them to distributions

def model_gaussian_deviations(observations = None):
    theta = pyro.param("theta", torch.ones([1]))    # theta = estimator for the mean
    gaussian_distribution = pyro.distributions.Normal(theta, sigma_true)    # Use theta to define distribution
    
    with pyro.plate("observations", n_observations):
        obs = pyro.sample("obs",gaussian_distribution, obs = observations)
        return obs

    
    

"""
    4. Statistical inference --------------------------------------------------
"""



# i) Create the guide for stochastic variational inference (SVI)

model_guide = pyro.infer.autoguide.AutoNormal(model_gaussian_deviations)


# ii) Run the optimization for SVI

adam = pyro.optim.Adam({"lr": 0.02})
elbo = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(model_gaussian_deviations, model_guide, adam, elbo)

losses = []
for step in range(1000):  
    loss = svi.step(observations)
    losses.append(loss)
    if step % 100 == 0:
        print("Elbo loss: {}".format(loss))



"""
    5. Plots and ilustrations ------------------------------------------------
"""


# i) Print out result and compare to mean

for name, value in pyro.get_param_store().items():
    print(" The pyro solution for the parameter {} is {} ".format(name, pyro.param(name).data.cpu().numpy()))

print('The arithmetic average (maximum likelihood estimator) of the mean is {}'.format(torch.mean(observations)))







































