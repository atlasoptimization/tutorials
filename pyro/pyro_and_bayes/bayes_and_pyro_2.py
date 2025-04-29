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
performed on an object with a randomly distributed temperature. The goal is to
compute the posterior of the true temperature given observations. We will do 
this by providing the analytical solution and also by emplying pyros inference 
machinery.
For this, do the following:
    1. Imports and definitions
    2. Generate synthetic data
    3. Analytic posterior
    4. Pyro model and inference
    6. Plots and illustrations
    
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
n_disc = 1000
z_disc = torch.linspace(7, 13, n_disc)

torch.manual_seed(0)
pyro.set_rng_seed(0)



"""
    2. Generate synthetic data
"""


# i) Define params

mu_z = 10           # 10 degree celsius (deg c) is mean of temperature control
sigma_z = 1         # 1 deg c is standard deviation of temperature control
sigma_T = 1         # 1 deg c is standard deviation of measurements


# ii) Simulate data

z_true = 10         # 10 degree celsius (deg c) is underlying true temperature
dist_data = pyro.distributions.Normal(loc = z_true , scale = sigma_T).expand([n_data])
data = dist_data.sample()



"""
    3. Analytic Posterior
"""


# i)  Compute likelihood and prior as functions of z

def likelihood(z):
    coeff_1 = (1/(sigma_T*torch.sqrt(2*torch.tensor(torch.pi))))
    coeff_2 = -(1/(2*sigma_T**2))
    fun_vals = coeff_1 * torch.exp( coeff_2 * (data - z)**2)
    fun_val = torch.prod(fun_vals)
    return fun_val

def prior(z):
    coeff_1 = (1/(sigma_z*torch.sqrt(2*torch.tensor(torch.pi))))
    coeff_2 = -(1/(2*sigma_z**2))
    fun_val = coeff_1 * torch.exp( coeff_2 * (z - mu_z)**2)
    return fun_val

likelihood_vec = torch.zeros([n_disc])
prior_vec = torch.zeros([n_disc])


for k in range(n_disc):
    likelihood_vec[k] = likelihood(z_disc[k])
    prior_vec[k] = prior(z_disc[k])


# ii) Compute posterior numerically

nonnormalized_posterior = likelihood_vec*prior_vec
numerical_posterior = nonnormalized_posterior/(torch.sum(nonnormalized_posterior) * 6/ n_disc)

max_index = torch.argmax(numerical_posterior)
z_map_graphical = z_disc[max_index]


# iii) Compute closed form solution of posterior

data_mean = torch.mean(data)
var_post = torch.tensor(1/((n_data/sigma_T**2)+(1/sigma_z**2)))
mu_post = var_post * ((mu_z/sigma_z**2) + n_data * (data_mean/sigma_T**2))

def posterior(z):
    coeff_1 = (1/(torch.sqrt(var_post)*torch.sqrt(2*torch.tensor(torch.pi))))
    coeff_2 = -(1/(2*var_post))
    fun_val = coeff_1 * torch.exp( coeff_2 * (z - mu_post)**2)
    return fun_val

analytical_posterior = torch.zeros([n_disc])
for k in range(n_disc):
    analytical_posterior[k] = posterior(z_disc[k])



"""
    4. Pyro model and inference
"""


# i) Pyro model

def model(observations = None):
    # Sample from latent variable to determine observation distribution
    dist_z = pyro.distributions.Normal(mu_z, sigma_z)
    z = pyro.sample('latent_z', dist_z)
    dist_obs = pyro.distributions.Normal(z, sigma_T)
    
    # Sample from observation distribution
    with pyro.plate('batch_plate', size = n_data, dim = -1):
        sample = pyro.sample('sample', dist_obs, obs = observations)
    return sample


# ii) Pyro guide

# def guide(observations = None):
#     pass

def guide(observations = None):
    # Define posterior distribution
    mu_post = pyro.param('mu_post', init_tensor = torch.zeros([1]))
    sigma_post = pyro.param('sigma_post', init_tensor = torch.ones([1]))
    
    dist_post = pyro.distributions.Normal(mu_post, sigma_post)
    z_post = pyro.sample('latent_z', dist_post)

    return z_post


# iii) Pyro inference

adam = pyro.optim.NAdam({"lr": 0.1})
elbo = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(model, guide, adam, elbo)

loss_sequence = []
mu_post_sequence = []
sigma_post_sequence = []
for step in range(1000):
    loss = svi.step(data)
    if step % 100 == 0:
        print('epoch: {} ; loss : {}'.format(step, loss))
    else:
        pass
    loss_sequence.append(loss)
    mu_post_sequence.append(pyro.get_param_store()['mu_post'].item())
    sigma_post_sequence.append(pyro.get_param_store()['sigma_post'].item())



"""
    5. Plots and illustrations
"""

# # i) Plot negative log likelihood 

# plt.figure(1, dpi = 300)
# plt.plot(mu_disc, nll)
# plt.title('Shifted negative log likelihood')
# plt.xlabel('bias mu')
# plt.ylabel('implausibility of bias mu')


# ii) Training process

fig, ax = plt.subplots(1,3, figsize = (10,5))
ax[0].plot(loss_sequence)
ax[0].set_title('Loss during computation')

ax[1].plot(mu_post_sequence)
ax[1].set_title('Estimated posterior mu during computation')

ax[2].plot(sigma_post_sequence)
ax[2].set_title('Estimated posterior sigma during computation')


# iii) Print results

mu_post_pyro = pyro.get_param_store()['mu_post'].item()
sigma_post_pyro = pyro.get_param_store()['sigma_post'].item()
print(' True z = {} \n true posterior mu = {} \n pyro posterior mu = {}'\
      ' \n true posterior sigma = {} \n pyro posterior sigma = {}'
      .format(z_true, mu_post, mu_post_pyro, torch.sqrt(var_post), sigma_post_pyro))

