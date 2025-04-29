#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is part of a series of notebooks that showcase the pyro probabilistic
programming language - basically pytorch + probability. In this series we explore
a sequence of increasingly complicated models meant to represent the behavior of
some measurement device. 
The crash course consist in the following:
    - minimal pyro example                      (cc_0_minimal_inference)
    - inference for control flow model          (cc_1_hello_dataset)
    - Model with no trainable params            (cc_2_model_0)
    - Model with deterministic parameters       (cc_2_model_1)
    - Model with latent variables               (cc_2_model_2)
    - Model with hierarchical randomness        (cc_2_model_3)
    - Model with discrete random variables      (cc_2_model_4)
    - Model with neural net                     (cc_2_model_5)
    


This script will build a stochastic model featuring a univariate Gaussian
distribution and fit the mean and variance so that the resultant distribution
is able to explain well the dataset.

For this, do the following:
    1. Imports and definitions
    2. Generate data
    3. Define stochastic model
    4. Perform inference
    5. Plots and illustrations
    
The script is meant solely for educational and illustrative purposes. Written by
Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.com.
"""



"""
    1. Imports and definitions
"""


# i) Imports

import pyro
import torch

import matplotlib.pyplot as plt


# ii) Definitions

n_data = 100
torch.manual_seed(0)
pyro.set_rng_seed(0)



"""
    2. Generate data
"""


# i) Set up data distribution (=standard normal))

mu_true = torch.zeros([1])
sigma_true = torch.ones([1])

extension_tensor = torch.ones([100])
data_dist = pyro.distributions.Normal(loc = mu_true * extension_tensor, scale = sigma_true)


# ii) Sample from dist to generate data

data = data_dist.sample()



"""
    3. Define stochastic model
"""


# i) Define model as normal with mean and var parameters

def model(observations = None):
    mu = pyro.param(name = 'mu', init_tensor = torch.tensor([5.0])) 
    sigma = pyro.param( name = 'sigma', init_tensor = torch.tensor([5.0]))
    
    obs_dist = pyro.distributions.Normal(loc = mu * extension_tensor, scale = sigma)
    with pyro.plate(name = 'data_plate', size = n_data, dim = -1):
        model_sample = pyro.sample('observation', obs_dist, obs = observations)

    return model_sample



"""
    4. Perform inference
"""


# i) Set up guide

def guide(observations = None):
    pass


# ii) Set up inference


adam = pyro.optim.Adam({"lr": 0.1})
elbo = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(model, guide, adam, elbo)


# iii) Perform svi

for step in range(100):
    loss = svi.step(data)



"""
    5. Plots and illustrations
"""


# i) Print results

print('True mu = {}, True sigma = {} \n Inferred mu = {:.3f}, Inferred sigma = {:.3f}'
      .format(mu_true, sigma_true, 
              pyro.get_param_store()['mu'].item(),
              pyro.get_param_store()['sigma'].item()))


# ii) Plot data

fig1 = plt.figure(num = 1, dpi = 300)
plt.hist(data.detach().numpy())
plt.title('Histogram of data')


# iii) Plot distributions

t = torch.linspace(-3,3,100)
inferred_dist = pyro.distributions.Normal( loc = pyro.get_param_store()['mu'], 
                                          scale = pyro.get_param_store()['sigma'])

fig2 = plt.figure(num = 2, dpi = 300)
plt.plot(t, torch.exp(data_dist.log_prob(t)), color = 'k', label = 'true', linestyle = '-')
plt.plot(t, torch.exp(inferred_dist.log_prob(t)).detach(), color = 'k', label = 'inferred', linestyle = '--')
plt.title('True and inferred distributions')




















