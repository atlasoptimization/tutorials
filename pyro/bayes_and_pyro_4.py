#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is part of a series of tutorials that showcase bayesian concepts
and use pyro to perform inference and more complicated computations. During these
tutorials, we will tackle a sequence of problems of increasing complexity that 
will take us from fitting of distributional parameters to the computation of 
nontrivial posterior distributions. Use best in conjunction with companion slides.
The tutorials consist in the following:
    - fit a mean                            (tutorial_1) 
    - compute simple posterior              (tutorial_2)
    - fit mean and compute posterior        (tutorial_3)
    - fit params, multivariate posterior    (tutorial_4)


This script is to showcase a problem in which temperature measurements are 
performed on nultiple objects at multiple locations. The temperature of the
objects is random exhibiting spatial correlation and the measurements are
are affected by randomness, too. The goal is to compute distributional parameters 
and the posterior of the true temperature at some location given observations at 
other locations. We will do this by computing the true posterior and compare it 
to the results of pyros inference machinery.
For this, do the following:
    1. Imports and definitions
    2. Generate synthetic data
    3. True posterior
    4. Pyro model and inference
    5. Plots and illustrations
    
The script is meant solely for educational and illustrative purposes. Written by
Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.com.
"""



"""
    1. Imports and definitions
"""


# i) Imports

import torch
import pyro
import matplotlib.pyplot as plt


# ii) Definitions

n_data = 100
n_locs = 3
n_disc = 100


torch.manual_seed(0)
pyro.set_rng_seed(0)



"""
    2. Generate synthetic data
"""


# i) Define params

# mean of [10,8,12] degree celsius (deg c) at water, mountain, forest locations
# measurements are unbiased -> mu_epsion = 0 deg c
mu_z = 1.0 * torch.tensor([[10, 8, 12]])               
mu_epsilon = torch.zeros([1,3])

# autocorrelated true temperatures and i.i.d. noise
sigma_z = 1.0 * torch.tensor([[2,1,1],[1,4,2],[1,2,3]])
sigma_epsilon = 1 * torch.eye(n_locs)


# ii) Simulate data

# sample underlying true temperature in deg c, then sample subsequent measurement
dist_z = pyro.distributions.MultivariateNormal(mu_z, sigma_z).expand([n_data])
z_true = dist_z.sample()         
dist_data = pyro.distributions.MultivariateNormal(loc = z_true + mu_epsilon ,
                        covariance_matrix = sigma_epsilon)
data = dist_data.sample()


# iii) Observation for conditioning

# Measurement in water location is given, temperature at mountain and forest
# location is to be estimated
data_observation = torch.tensor([11])
index_observation = [0]
index_to_estimate = [1,2]



"""
    3. Analytic MLE & Posterior
"""


# i)  Compute conditional mean and covariance

# means for water location (w), mountain location (m), forest location (f)
mu_w = mu_z[0,0]
mu_mf = mu_z[0,1:3]

# covariances for locations (w,m,f) and measurements at location w (Tw)
sigma_T_T = sigma_z + sigma_epsilon
sigma_Tw_Tw = sigma_T_T[0,0]
sigma_mf_w = sigma_z[1:3,0]
sigma_mf_mf = sigma_z[1:3, 1:3]

# use equations for conditional mean and conditional covariance of Gaussian
conditional_mu = mu_mf + sigma_mf_w * (1/(sigma_Tw_Tw)) * (data_observation - mu_w) 



# ii) Construct posterior from mean and covariance




"""
    4. Pyro model and inference
"""


# i) Pyro model

def model(observations = None):    
    # Define unknown bias
    bias_mu = pyro.param('bias_mu', torch.zeros([1,1]))
    
    # Sample from latent and observation distribution
    dist_z = pyro.distributions.Normal(mu_z, sigma_z).expand([n_epochs,1])
    with pyro.plate('z_plate', size = n_epochs, dim = -2):
        
        # Sample from latent variable to determine observation distribution
        z = pyro.sample('latent_z', dist_z)
        dist_obs = pyro.distributions.Normal(z + bias_mu, sigma_T)
        dist_obs_reshaped = dist_obs.expand([n_epochs,n_data])
        
        with pyro.plate('measure_plate', size = n_data, dim = -1):
            # Sample from observation distribution
            sample = pyro.sample('sample', dist_obs_reshaped, obs = observations)
    return sample, z


# ii) Pyro guide

def guide(observations = None):
    # Define posterior distribution
    mu_post = pyro.param('mu_post', init_tensor = torch.zeros([n_epochs,1]))
    sigma_post = pyro.param('sigma_post', init_tensor = torch.ones([n_epochs,1]))
    
    # Sample from posterior distribution
    dist_post = pyro.distributions.Normal(mu_post, sigma_post)
    with pyro.plate('z_plate', size = n_epochs, dim = -2):
        z_post = pyro.sample('latent_z', dist_post)

    return z_post


# # Alternative formulation for the model and guide. Difference: Use of pyro.plate
# # as declarator and constructor. This allows easier dimensions for z, obs dists

# # i) Pyro model

# def model(observations = None):    
#     # Define unknown bias
#     bias_mu = pyro.param('bias_mu', torch.zeros([1]))
    
#     # Sample from latent variable to determine observation distribution
#     dist_z = pyro.distributions.Normal(mu_z, sigma_z)

    
#     # Sample from observation distribution
#     with pyro.plate('z_plate', size = n_epochs, dim = -2):
#         z = pyro.sample('latent_z', dist_z)
#         dist_obs = pyro.distributions.Normal(z + bias_mu, sigma_T)
        
#         with pyro.plate('measure_plate', size = n_data, dim = -1):
#                 sample = pyro.sample('sample', dist_obs, obs = observations)
                
#     return sample, z


# # ii) Pyro guide

# def guide(observations = None):
#     # Define posterior distribution
#     mu_post = pyro.param('mu_post', init_tensor = torch.zeros([n_epochs,1]))
#     sigma_post = pyro.param('sigma_post', init_tensor = torch.ones([n_epochs,1]),
#                             constraint = pyro.distributions.constraints.positive)
    
#     # Sample from posterior distribution
#     with pyro.plate('z_plate', size = n_epochs, dim = -2):
#         dist_post = pyro.distributions.Normal(mu_post, sigma_post)
#         z_post = pyro.sample('latent_z', dist_post)

#     return z_post


# iii) Pyro diagnostics

# Render model and guide
pyro.render_model(model, model_args=(), render_distributions=True, render_params=True)  
pyro.render_model(guide, model_args=(), render_distributions=True, render_params=True)  

# Showcase the execution traces
model_trace_data = pyro.poutine.trace(model).get_trace(data)
model_trace_data.compute_log_prob()
print(model_trace_data.format_shapes())

guide_trace_data = pyro.poutine.trace(guide).get_trace(data)
guide_trace_data.compute_log_prob()
print(guide_trace_data.format_shapes())



# iii) Pyro inference

adam = pyro.optim.NAdam({"lr": 0.05})
elbo = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(model, guide, adam, elbo)

loss_sequence = []
mu_bias_sequence = []
mu_post_sequence = []
sigma_post_sequence = []
for step in range(5000):
    loss = svi.step(data)
    if step % 500 == 0:
        print('epoch: {} ; loss : {}'.format(step, loss))
    else:
        pass
    loss_sequence.append(loss)
    mu_bias_sequence.append(pyro.get_param_store()['bias_mu'].detach().numpy().flatten())
    mu_post_sequence.append(pyro.get_param_store()['mu_post'].detach().numpy().flatten())
    sigma_post_sequence.append(pyro.get_param_store()['sigma_post'].detach().numpy().flatten())



"""
    5. Plots and illustrations
"""


# i) likelihood and prior for first epoch

ticksample = torch.round(torch.linspace(0,n_disc-1,5)).long()
fig, ax = plt.subplots(1,3, figsize = (15,5))
ax[0].imshow(likelihood_mat[0,:,:])
ax[0].set_title('Likelihood first epoch')
ax[0].set_xlabel('bias mu')
ax[0].set_ylabel('latent z')
# Set the tick labels for x and y axes
ax[0].set_xticks(ticksample)
ax[0].set_yticks(ticksample)
ax[0].set_xticklabels(torch.round(mu_disc[ticksample], decimals = 1).numpy())
ax[0].set_yticklabels(torch.round(z_disc[ticksample], decimals = 1).numpy())

ax[1].imshow(prior_mat[0,:,:])
ax[1].set_title('Prior first epoch')
ax[1].set_xlabel('bias mu')
ax[1].set_ylabel('latent z')
# Set the tick labels for x and y axes
ax[1].set_xticks(ticksample)
ax[1].set_yticks(ticksample)
ax[1].set_xticklabels(torch.round(mu_disc[ticksample], decimals = 1).numpy())
ax[1].set_yticklabels(torch.round(z_disc[ticksample], decimals = 1).numpy())

ax[2].imshow(nonnormalized_posterior[0,:,:])
ax[2].set_title('Nonnormalized Posterior first epoch')
ax[2].set_xlabel('bias mu')
ax[2].set_ylabel('latent z')
# Set the tick labels for x and y axes
ax[2].set_xticks(ticksample)
ax[2].set_yticks(ticksample)
ax[2].set_xticklabels(torch.round(mu_disc[ticksample], decimals = 1).numpy())
ax[2].set_yticklabels(torch.round(z_disc[ticksample], decimals = 1).numpy())

plt.tight_layout()
plt.show()


# ii) Estimation of the bias

fig, ax = plt.subplots(4,1, figsize = (10,15))
ax[0].plot(mu_disc, likelihood_given_mu[0,0,:])
ax[0].set_title('Likelihood given mu first epoch')
ax[0].set_xlabel('bias mu')
ax[0].set_ylabel('likelihood')

ax[1].plot(mu_disc, likelihood_given_mu[1,0,:])
ax[1].set_title('Likelihood given mu second epoch')
ax[1].set_xlabel('bias mu')
ax[1].set_ylabel('likelihood')

ax[2].plot(mu_disc, likelihood_given_mu[2,0,:])
ax[2].set_title('Likelihood given mu third epoch')
ax[2].set_xlabel('bias mu')
ax[2].set_ylabel('likelihood')

ax[3].plot(mu_disc, product_likelihoods_given_mu[0,:])
ax[3].set_title('Likelihood given mu all epochs')
ax[3].set_xlabel('bias mu')
ax[3].set_ylabel('likelihood')

plt.tight_layout()
plt.show()


# iii) Training process

fig, ax = plt.subplots(1,4, figsize = (15,5))
ax[0].plot(loss_sequence)
ax[0].set_title('Loss during computation')

ax[1].plot(mu_bias_sequence)
ax[1].set_title('Estimated bias during computation')

ax[2].plot(mu_post_sequence)
ax[2].set_title('Estimated posterior mu during computation')

ax[3].plot(sigma_post_sequence)
ax[3].set_title('Estimated posterior sigma during computation')

plt.tight_layout()
plt.show()


# iv) Print results

mu_bias_pyro = pyro.get_param_store()['bias_mu'].detach().numpy().flatten()
mu_post_pyro = pyro.get_param_store()['mu_post'].detach().numpy().flatten()
sigma_post_pyro = pyro.get_param_store()['sigma_post'].detach().numpy().flatten()
print(' True bias = {} \n bias_numerical = {} \n bias_pyro = {}'
      .format(mu_bias, mu_bias_numerical, mu_bias_pyro))
# print(' True z = {} \n true posterior mu = {} \n pyro posterior mu = {}'\
#       ' \n true posterior sigma = {} \n pyro posterior sigma = {}'
#       .format(z_true, mu_post, mu_post_pyro, torch.sqrt(var_post), sigma_post_pyro))


# v) Data & posterior densities of z

def pyro_posterior(z):
    coeff_1 = (1/(torch.tensor(sigma_post_pyro)*torch.sqrt(2*torch.tensor(torch.pi))))
    coeff_2 = -(1/(2*torch.tensor(sigma_post_pyro)**2))
    fun_vals = coeff_1 * torch.exp( coeff_2 * (z - mu_post_pyro)**2)
    return fun_vals

pyro_posterior_z = torch.zeros([n_epochs, n_disc])
for k in range(n_disc):
    prob_vals = pyro_posterior(z_disc[k])
    pyro_posterior_z[:,k] = prob_vals.flatten()

fig, ax = plt.subplots(3,1, figsize = (10,10))
ax[0].plot(z_disc, posterior_z[0,:], label = 'true posterior')
ax[0].plot(z_disc, pyro_posterior_z[0,:], label = 'pyro posterior')
ax[0].hist(data[0,:].numpy(), density = True, label = 'data')
ax[0].stem(z_true[0].numpy(),[1], label = 'true z')
ax[0].set_title('Data & posterior over z first epoch')
ax[0].set_xlabel('z')
ax[0].set_ylabel('probability')
ax[0].legend()

ax[1].plot(z_disc, posterior_z[1,:], label = 'true posterior')
ax[1].plot(z_disc, pyro_posterior_z[1,:], label = 'pyro posterior')
ax[1].hist(data[1,:].numpy(), density = True, label = 'data')
ax[1].stem(z_true[1].numpy(),[1], label = 'true z')
ax[1].set_title('Data & posterior over z second epoch')
ax[1].set_xlabel('z')
ax[1].set_ylabel('probability')

ax[2].plot(z_disc, posterior_z[2,:], label = 'true posterior')
ax[2].plot(z_disc, pyro_posterior_z[2,:], label = 'pyro posterior')
ax[2].hist(data[2,:].numpy(), density = True, label = 'data')
ax[2].stem(z_true[2].numpy(),[1], label = 'true z')
ax[2].set_title('Data & posterior over z third epoch')
ax[2].set_xlabel('z')
ax[2].set_ylabel('probability')

plt.tight_layout()
plt.show()







