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
performed on an object with a randomly distributed temperature. The measurements
are affected by an unknown bias. The goal is to compute an MLE of the bias and 
to compute the posterior of the true temperature given observations. We will do 
this by providing the analytical solution and also by emplying pyros inference 
machinery.
For this, do the following:
    1. Imports and definitions
    2. Generate synthetic data
    3. Analytic MLE & posterior
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

n_data = 20
n_epochs = 3
n_disc = 100

z_disc = torch.linspace(7, 13, n_disc)
mu_disc = torch.linspace(-3, 3, n_disc)

torch.manual_seed(0)
pyro.set_rng_seed(0)



"""
    2. Generate synthetic data
"""


# i) Define params

mu_z = 10           # 10 degree celsius (deg c) is mean of temperature control
sigma_z = 1         # 1 deg c is standard deviation of temperature control
sigma_T = 1         # 1 deg c is standard deviation of measurements
mu_bias = 2         # 2 deg c is bas of measurement instrument


# ii) Simulate data

# sample underlying true temperature in deg c, then sample subsequent measurement
dist_z = pyro.distributions.Normal(mu_z, sigma_z)
z_true = dist_z.sample([n_epochs,1])         
dist_data = pyro.distributions.Normal(loc = z_true + mu_bias , scale = sigma_T).expand([n_epochs,n_data])
data = dist_data.sample()



"""
    3. Analytic MLE & Posterior
"""


# i)  Compute likelihood and prior as functions of z

def likelihood(z, mu):
    coeff_1 = (1/(sigma_T*torch.sqrt(2*torch.tensor(torch.pi))))
    coeff_2 = -(1/(2*sigma_T**2))
    fun_vals_epochs = coeff_1 * torch.exp( coeff_2 * (data - z - mu)**2)
    fun_vals = torch.prod(fun_vals_epochs, dim = 1)
    return fun_vals

def prior(z):
    coeff_1 = (1/(sigma_z*torch.sqrt(2*torch.tensor(torch.pi))))
    coeff_2 = -(1/(2*sigma_z**2))
    fun_val = coeff_1 * torch.exp( coeff_2 * (z - mu_z)**2)
    fun_vals = fun_val.repeat([n_epochs])
    return fun_vals

likelihood_mat = torch.zeros([n_epochs, n_disc, n_disc])
prior_mat = torch.zeros([n_epochs, n_disc, n_disc])


for k in range(n_disc):
    for l in range(n_disc):
        likelihood_mat[:,k,l] = likelihood(z_disc[k], mu_disc[l])
        prior_mat[:,k,l] = prior(z_disc[k])


# ii) Compute posterior numerically

nonnormalized_posterior = likelihood_mat*prior_mat
likelihood_given_mu = (torch.sum(nonnormalized_posterior,1).reshape([n_epochs,1,n_disc]))
product_likelihoods_given_mu = torch.prod(likelihood_given_mu, dim = 0)
numerical_posterior = nonnormalized_posterior/(likelihood_given_mu.repeat([1,n_disc,1]) * 6/ n_disc)

max_index_mu = torch.argmax(product_likelihoods_given_mu)
mu_bias_numerical = mu_disc[max_index_mu]

posterior_z = numerical_posterior[:, :, max_index_mu]
max_index_z = torch.argmax(posterior_z, 1)
z_map_graphical = z_disc[max_index_z]



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







