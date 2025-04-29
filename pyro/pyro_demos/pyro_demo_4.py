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
distribution and a parametric model for mean and covariance. There will be 
multiple realizations of a timeseries that features a randomly chosen trend and
a deterministic, parametric covariance model. 

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
import numpy as np
import copy
import matplotlib.pyplot as plt



# ii) Definitions

n_time = 100
n_data = 100
time = torch.linspace(0,1,n_time)

torch.manual_seed(0)
pyro.set_rng_seed(0)

torch.set_default_tensor_type(torch.DoubleTensor)



"""
    2. Generate data
"""
 

# i) Prepare data generation by simulating random coefficients alpha that are
# ingredients to the computation of the mean functions.

offset = torch.linspace(1,1,n_time)
slope = torch.linspace(0,1,n_time)
funs_mu = torch.vstack((offset,slope))

alpha_dist = pyro.distributions.MultivariateNormal(loc = torch.zeros([2]), 
                                  covariance_matrix = torch.tensor([[1, 0.7],[0.7,1]]))
alpha = alpha_dist.sample([n_data])


# ii) Define mean and covariance of distribution. The mean mu is a alpha_1 * offset +
# alpha_2 * slope where the alphas are chosen as random. The covariance sigma
# is k(s,t) = min(s,t), the covariance of a wiener process / random walk. Think
# of a random line sampled with an increasingly uncertain measurement instrument.

mu_true = alpha@funs_mu
sigma_true = torch.zeros([n_time,n_time])
for k in range(n_time):
    for l in range(n_time):
        sigma_true[k,l] = torch.min(torch.tensor([time[k], time[l]]))
sigma_true = sigma_true + 1e-3*torch.eye(n_time)


# iii) Define distribution and sample

data_dist = pyro.distributions.MultivariateNormal(loc = mu_true, covariance_matrix = sigma_true)
data = data_dist.sample()
z_true = alpha



"""
    3. Define stochastic model
"""


# i) Functions for parametric covariance

offset_sigma = torch.ones([n_time,n_time])
slope_sigma = torch.outer(slope,slope)
sqrt_sigma = torch.sqrt(slope_sigma)
min_sigma = sigma_true
funs_sigma = torch.cat((offset_sigma.unsqueeze(2), 
                        slope_sigma.unsqueeze(2), 
                        sqrt_sigma.unsqueeze(2),
                        min_sigma.unsqueeze(2)), dim = -1)


# ii) Define model with hidden random variable z

def model(observations = None):
    # Parameters
    cov_coeffs = pyro.param('cov_coeffs', torch.zeros([4]))
    mu_z = pyro.param('mu_z', torch.ones([1,2]))
    sigma_z = pyro.param('sigma_z', torch.eye(2), constraint = pyro.distributions.constraints.positive_definite)
    
    # Hidden variable z chosen at random 
    z_dist = pyro.distributions.MultivariateNormal(mu_z, sigma_z)
    with pyro.plate('batch_plate', size = n_data, dim = -1):
        z = pyro.sample('latent_z', z_dist)
    
        # Means (different for each sample) and covariance matrices (same for each sample)
        mu = z @ funs_mu
        sigma = funs_sigma @ cov_coeffs + 1e-3 * torch.eye(n_time)
        
        # individual realization
        sample_dist = pyro.distributions.MultivariateNormal(mu, sigma)
        samples = pyro.sample('samples', sample_dist, obs = observations)
    
    return samples, sigma


# Illustrate model and sample
pyro.render_model(model, model_args=(), render_distributions=True, render_params=True)
untrained_sample, untrained_sigma = copy.copy(model())


# iii) Define guide

# def guide(observations = None):
#     pass

# guide = pyro.infer.autoguide.AutoNormal(model)

def guide(observations = None):    
    # Parameters
    mu_z_q = pyro.param('mu_z_q', torch.zeros([1, 2]))
    sigma_z_q_base = pyro.param('sigma_z_q_base', torch.eye(2),
                                constraint=pyro.distributions.constraints.positive_definite)
    
    # Transformation mapping observations to distributional params
    linear_map = pyro.param('linear_map', torch.zeros([observations.shape[1], 4]))
    features = observations @ linear_map
    
    # Adjusting mu_z_q and sigma_z_q using the features derived from observations
    mu_z_q_obs = mu_z_q + features[:, :2]
    sigma_z_q = 1e-2 * torch.eye(2) + sigma_z_q_base  #+ torch.exp(torch.diag_embed(features[:, 2:])) 
    
    # Approximate distribution for z conditioned on observations
    z_dist_q = pyro.distributions.MultivariateNormal(mu_z_q_obs, sigma_z_q)
    
    with pyro.plate('batch_plate', size=observations.shape[0], dim=-1):
        # Sample z from its approximate posterior conditioned on observations
        z_q = pyro.sample('latent_z', z_dist_q)

    return z_q, mu_z_q_obs, sigma_z_q



"""
    4. Perform inference
"""


# i) Set up inference


adam = pyro.optim.NAdam({"lr": 0.3})
elbo = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(model, guide, adam, elbo)


# ii) Perform svi

for step in range(10000):
    loss = svi.step(data)
    if step % 100 == 0:
        print('epoch: {} ; loss : {}'.format(step, loss))
    else:
        pass


# iii) Sample trained model

trained_sample, trained_sigma = copy.copy(model())



"""
    5. Plots and illustrations
"""


# i) Print results

param_dict = dict()
for name, value in pyro.get_param_store().items():
    param_dict[name] = pyro.param(name).data.cpu().numpy()
print(' cov_coeffs_true = {} \n mu_z_true = {} \n sigma_z_true ={}'
      .format(torch.tensor([0,0,0,1]), [0,0], 
              np.array2string(torch.tensor([[1, 0.7],[0.7,1]]).numpy())))
print(' cov_coeffs_est = {} \n mu_z_est = {} \n sigma_z_est ={}'
      .format(np.array2string(param_dict['cov_coeffs'], precision = 3),
              np.array2string(param_dict['mu_z'], precision = 3), 
              np.array2string(param_dict['sigma_z'], precision = 3))) 


# ii) Plot distributions

# Creating the figure and subplots for sample visualization
fig1, axs = plt.subplots(3, 1, figsize=(8, 12), dpi=300)

# Plotting the realizations
axs[0].plot(time, data.detach().numpy().T)
axs[0].set_title('Data')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Value')

axs[1].plot(time, untrained_sample.detach().numpy().T)
axs[1].set_title('Untrained model')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Value')

axs[2].plot(time, trained_sample.detach().numpy().T)
axs[2].set_title('Trained model')
axs[2].set_xlabel('Time')
axs[2].set_ylabel('Value')

plt.tight_layout()
plt.show()

# Creating the figure and subplots for covariance matrices
fig2, axs = plt.subplots(1, 3, figsize=(12, 8), dpi=300)

# Plotting the cov mats
axs[0].imshow(sigma_true)
axs[0].set_title('True Cov')

axs[1].imshow(untrained_sigma.detach())
axs[1].set_title('Untrained Cov')

axs[2].imshow(trained_sigma.detach())
axs[2].set_title('Trained Cov')

plt.tight_layout()
plt.show()

# Illustrate true z and posterior distributions

n_showcase = 5
n_sampling = 20
z_q = torch.zeros([n_sampling, n_showcase, 2])
for k in range(n_sampling):
    z_q[k,:,:], mu_q , sigma_q = guide(data[0:n_showcase,:])

fig3 = plt.figure(3, dpi = 300)
plt.scatter(z_q.detach()[:,:,0], z_q.detach()[:,:,1], color = [0.5,0.5,0.5], label = 'Posterior sample')
plt.scatter(z_true[:n_showcase,0], z_true[:n_showcase,1], color = 'k', label = 'True latent')
plt.scatter(mu_q.detach()[:,0], mu_q.detach()[:,1], color = 'r', label = 'Posterior mean')
plt.title('Latent vars and posteriors')
plt.legend()



