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

This script will perform gaussian process regression to learn the structure of
timeseries exhibiting complex, nonstationary behavior. The stochastic model of
the gaussian process to be trained features a nonparametric representation of
the covariance function by a semidefinite coefficient tensor that is also endowed
with a wishart prior. We will showcase the complex covariance matrices that 
result from this approach and compare performance of the model for interpolation
of data to more simple approaches.

For this, do the following:
    1. Imports and definitions
    2. Generate data
    3. Define stochastic model
    4. Perform inference
    5. Interpolation
    6. Plots and illustrations
    
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


# iii) Global provisions

torch.manual_seed(0)
pyro.set_rng_seed(0)

torch.set_default_tensor_type(torch.DoubleTensor)



"""
    2. Generate data
"""
 
# i) Distributional parameters for brownian bridge

mu_true = torch.zeros([1,n_time])
sigma_true = torch.zeros([n_time,n_time])
def cov_fun_true(s,t):
    cov_val = torch.min(s,t) - s*t
    return cov_val
for k in range(n_time):
    for l in range(n_time):
        sigma_true[k,l] = cov_fun_true(time[k],time[l])
sigma_true = sigma_true + 1e-3*torch.eye(n_time)

# ii) Define and sample from distribution

extension_tensor = torch.ones([n_data,1])
dist_true = pyro.distributions.MultivariateNormal(loc = mu_true * extension_tensor,
                                                  covariance_matrix = sigma_true)
data = dist_true.sample()



"""
    3. Define stochastic model
"""


# i) Auxiliary constructions

# Construct exponential covariance as an ingredient to the prior
sigma_prior = torch.zeros([n_time,n_time])
def cov_fun_prior(s,t):
    cov_val = 1*torch.exp(-torch.abs(s - t)/ 0.1)
    return cov_val
for k in range(n_time):
    for l in range(n_time):
        sigma_prior[k,l] = cov_fun_prior(time[k], time[l])
        
# Decomposition of prior       
n_exp = 50   # size of coefficient tensor to be inferred
u, l, v =  torch.linalg.svd(sigma_prior)
u_cut = u[:,0:n_exp]

# Construct prior for coefficient tensor gamma. Prior over gamma preferres 
# covariance matrices looking like the sigma_prior.
prior_gamma = torch.diag_embed(l[0:n_exp])


# ii) Define model

# Model consists of sampling Wishart distributed coefficient tensor, using it to
# construct covariance matrix, then sample from a multivariate gaussian with
# that covariance. Scheme is effectively regularization on the covariance matrix.
def model(observations = None, gamma = None):
    # Sample semidefinite coefficient tensor or define via arg
    dist_gamma = pyro.distributions.Wishart(n_exp, prior_gamma)
    gamma_model = pyro.sample('gamma', dist_gamma)
    gamma_model = gamma_model if gamma == None else gamma
    
    # construct sample_distribution
    mu_model =  torch.zeros([1,n_time])
    sigma_model = u_cut@gamma_model@u_cut.T + 1e-3 * torch.eye(n_time)
    dist_sample = pyro.distributions.MultivariateNormal(loc = extension_tensor*mu_model,
                                                        covariance_matrix = sigma_model)
    
    # Sample batch independently
    with pyro.plate('batch_plate', size = n_data, dim = -1):
        sample = pyro.sample('sample', dist_sample, obs = observations)
    
    return sample


# iii) Define guide
# We will let the posterior of gamma given observations be approximated by a
# Wishart distribution. The parameters of that posterior Wishart are unknown
# and to be inferred.
def guide(observations = None, gamma = None):
    # Define parameters to be inferred
    gamma_posterior = pyro.param('gamma_posterior', prior_gamma, 
                                 constraint = pyro.distributions.constraints.positive_definite)
    gamma_sample_dist = pyro.distributions.Wishart(n_exp, gamma_posterior)
    gamma_sample = pyro.sample('gamma', gamma_sample_dist)
    
    return gamma_sample


# Sampling from the approximate posterior then allows e.g. estimating the
# conditional expectation of gamma. Or we directly extract the parameter 
# 'gamma_posterior'.

sample_untrained = copy.copy(model())
posterior_untrained = copy.copy(guide())
gamma_untrained = copy.copy(pyro.get_param_store()['gamma_posterior'])
sigma_untrained = u_cut @ gamma_untrained @ u_cut.T



"""
    4. Perform inference
"""


# i) Set up inference


adam = pyro.optim.NAdam({"lr": 0.003})
elbo = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(model, guide, adam, elbo)


# ii) Perform svi

loss_sequence = []
for step in range(2000):
    loss = svi.step(data)
    loss_sequence.append(loss)
    if step % 100 == 0:
        print('epoch: {} ; loss : {}'.format(step, loss))
    else:
        pass


# iii) Sample trained model

gamma_trained = copy.copy(pyro.get_param_store()['gamma_posterior'])
sigma_trained = u_cut @ gamma_trained @ u_cut.T
sample_trained = copy.copy(model(observations = None, gamma = gamma_trained))
posterior_trained = copy.copy(guide())



"""
    5. Interpolation
"""


# i) Subsample data to generate observation

sample_index = np.linspace(20,80,4).astype(int)
sample_time = time[sample_index]
data_estimation = data[0,:][sample_index]
data_estimation_true = data[0,:]


# ii) Estimate using sigma_true, sigma_untrained, sigma_trained

# using true covariance
sigma_true_obs = sigma_true[np.ix_(sample_index, sample_index)]
sigma_true_est = sigma_true[:,sample_index]
est_sigma_true = sigma_true_est @ torch.linalg.pinv(sigma_true_obs) @ data_estimation

sigma_untrained_obs = sigma_untrained[np.ix_(sample_index, sample_index)]
sigma_untrained_est = sigma_untrained[:,sample_index]
est_sigma_untrained = sigma_untrained_est @ torch.linalg.pinv(sigma_untrained_obs) @ data_estimation

sigma_trained_obs = sigma_trained[np.ix_(sample_index, sample_index)]
sigma_trained_est = sigma_trained[:,sample_index]
est_sigma_trained = sigma_trained_est @ torch.linalg.pinv(sigma_trained_obs) @ data_estimation



"""
    6. Plots and illustrations
"""


# i) Plot training progress

fig = plt.figure(1, figsize = (10,5))
plt.plot(loss_sequence)
plt.title('Loss during computation')


# ii) Showcase data and model

fig, ax = plt.subplots(3,1, figsize = (10,10))
ax[0].plot(time, data.T)
ax[0].set_title('Original data')

ax[1].plot(time, sample_untrained.detach().T)
ax[1].set_title('Samples of model pre-training')

ax[2].plot(time, sample_trained.detach().T)
ax[2].set_title('Samples of model post-training')
plt.tight_layout()
plt.show()


# iii) Plot parameters

fig, ax = plt.subplots(2,3, figsize = (10,10))

ax[0,0].imshow(sigma_true)
ax[0,0].set_title('True covariance matrix')
ax[0,1].imshow(sigma_untrained.detach())
ax[0,1].set_title('Covariance matrix pre-training')
ax[0,2].imshow(sigma_trained.detach())
ax[0,2].set_title('Covariance matrix post-training')

ax[1,0].axis('off')
ax[1,1].imshow(gamma_untrained.detach())
ax[1,1].set_title('gamma pre-training')
ax[1,2].imshow(gamma_trained.detach())
ax[1,2].set_title('gamma post-training')
plt.tight_layout()
plt.show()


# iv) Showcase estimation

fig, ax = plt.subplots(3,1, figsize = (10,10))
ax[0].plot(time, data_estimation_true, label = 'True data')
ax[0].scatter(sample_time, data_estimation, label = 'observed data', color = 'k')
ax[0].plot(time, est_sigma_true, label = 'Estimation using true covariance')
ax[0].set_title('Original data and estimation using true covariance')

ax[1].scatter(sample_time, data_estimation, label = 'observed data', color = 'k')
ax[1].plot(time, est_sigma_untrained.detach(), label = 'Estimation using untrained covariance')
ax[1].set_title('Estimation using untrained covariance')

ax[2].scatter(sample_time, data_estimation, label = 'observed data', color = 'k')
ax[2].plot(time, est_sigma_trained.detach(), label = 'Estimation using trained covariance')
ax[2].set_title('Estimation using trained covariance')

plt.legend()
plt.tight_layout()
plt.show()


