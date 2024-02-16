#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is part of a series of pyro demos. These demos are designed to 
provide some insight into pyro's capabilities. They are not meant to be fully 
understood; rather they showcase potential use cases and how pyro can be em-
ployed in the context of classical statistics and modern machine learning.
will tackle a sequence of problems of increasing complexity that will take us
from understanding simple pyro concepts to fitting models via svi and finally
fitting a model involving hidden variables and neural networks.
The tutorials consist in the following:
    - estimate mean and variance                (demo_1)
    - inference for control flow model          (demo_2)
    - estimate mean and covariance              (demo_3)
    - fit a parametric model                    (demo_4)
    - distribution dependent on ANN             (demo_5)
    - gaussian process regression ++            (demo_6)
    - variational autoencoder                   (demo_7)
    - distribution dependent on DCO             (demo 8)

This script will build a stochastic model that involves a multivariate Gaussian
distribution and a parametric representation of mean and covariance. Both the
mean and covariance are estimated from data and the results are plotted.

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
import copy
import matplotlib.pyplot as plt



# ii) Definitions

n_data = 1000
torch.manual_seed(0)
pyro.set_rng_seed(0)



"""
    2. Generate data
"""
 

# i) Define distribution

mu_true = torch.tensor([[1,1]])
sigma_true = torch.tensor([[1, 0.7],[0.7,1]])

extension_tensor = torch.ones([n_data,1])
data_dist = pyro.distributions.MultivariateNormal(loc = mu_true * extension_tensor,
                                                  covariance_matrix = sigma_true)


# ii) Sample from dist

data = data_dist.sample()




"""
    3. Define stochastic model
"""


# i) Define model

def model(observations = None):
    # Parameters and distribution
    mu = pyro.param('mu', torch.zeros([1,2]))
    sigma = pyro.param('sigma', torch.eye(2), constraint = pyro.distributions.constraints.positive_definite)
    obs_dist = pyro.distributions.MultivariateNormal(loc = mu, covariance_matrix = sigma)
    
    # Sampling
    with pyro.plate('batch_plate', size = n_data, dim = -1):
        sample = pyro.sample('sample', obs_dist, obs = observations)
    return sample

# Illustrate model and sample
pyro.render_model(model, model_args=(), render_distributions=True, render_params=True)
untrained_sample = copy.copy(model())


# ii) Define guide

def guide(observations = None):
    pass



"""
    4. Perform inference
"""


# i) Set up inference


adam = pyro.optim.NAdam({"lr": 0.1})
elbo = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(model, guide, adam, elbo)


# ii) Perform svi

for step in range(100):
    loss = svi.step(data)
    if step % 10 == 0:
        print('epoch: {} ; loss : {}'.format(step, loss))
    else:
        pass


# iii) Sample trained model

trained_sample = copy.copy(model())



"""
    5. Plots and illustrations
"""


# i) Print results

for name, value in pyro.get_param_store().items():
    print(name, pyro.param(name).data.cpu().numpy())
    

# ii) Plot distributions

# Creating the figure and subplots for 2D histograms
fig, axs = plt.subplots(3, 1, figsize=(8, 12), dpi=300)

# Plotting the 2D scatterplots
axs[0].scatter(data[:,0].detach().numpy(), data[:,1].detach().numpy())
axs[0].set_title('Histogram of data')
axs[0].set_xlabel('Dimension 1')
axs[0].set_ylabel('Dimension 2')

axs[1].scatter(untrained_sample[:,0].detach().numpy(), untrained_sample[:,1].detach().numpy())
axs[1].set_title('Sampling from untrained model')
axs[1].set_xlabel('Dimension 1')
axs[1].set_ylabel('Dimension 2')

axs[2].scatter(trained_sample[:,0].detach().numpy(), trained_sample[:,1].detach().numpy())
axs[2].set_title('Sampling from trained model')
axs[2].set_xlabel('Dimension 1')
axs[2].set_ylabel('Dimension 2')

plt.tight_layout()
plt.show()