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
likelihood estimator in the simple context of estimating the scale parameter of
a normal distribution. We compare the value of the elbo loss to the penalty
function representing maximum likelihood estimation and explain the differences.
For this, do the following:
    1. Definitions and imports
    2. Simulate some data
    3. Define the model
    4. Define the guide
    5. Investigate losses
    6. Perform inference and compare
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

import copy
import matplotlib.pyplot as plt


# ii) Definitions

np.random.seed(0)
torch.manual_seed(0)

n_data = 10
sigma_interval = torch.linspace(0.5,5,500)



"""
    2. Simulate some data
"""

# i) Define data distribution

mu_true = torch.tensor(1.0)
sigma_true = torch.tensor(2.0)

data_dist = pyro.distributions.Normal(mu_true, sigma_true)


# ii) Sample from distribution

data = data_dist.sample([n_data])



"""
    3. Define the model
"""

# i) Define the model
#
# A single parameter sigma that acts as the scale of a normal  distribution. This
# case is slightly different from the simple case of an unknown mean due to the
# different way that the scale factorizes in the nll / elbo due to it being in
# the normalization constant and the exponent of the gaussian distribution.

def model(observations = None):
    # Set up parameter sigma We ignore the positivity constraint, as otherwise 
    # direct modification of the parameter is not possible anymore.
    sigma = pyro.param('sigma', init_tensor = torch.tensor(1.0))
    
    # Set up observation distribution, take independent samples from it
    obs_dist = pyro.distributions.Normal(mu_true, sigma)
    with pyro.plate('batch_plate', size = n_data, dim =-1):
        obs = pyro.sample('obs', obs_dist, obs = observations)

    return obs

# Run once to initialize params
untrained_simulation = model()



"""
    4. Define the guide
"""


# i) Define the guide

def guide(observations = None):
    pass



"""
    5. Investigate losses
"""


# i) Investigate elbo

elbo = pyro.infer.Trace_ELBO()
def elbo_fn(data):
    elbo_val = elbo.differentiable_loss(model, guide, data)
    return elbo_val

# Estimate elbo loss for different values of sigma
elbo_vals_sigma = []
for k in range(500):
    pyro.param('sigma').data = copy.deepcopy(sigma_interval[k])
    elbo_vals_sigma.append(elbo.differentiable_loss(model, guide, data).detach())


# ii) Investigate nll

# Defne nll and log evidence and compute nll loss for different values of sigma
def negloglikelihood(sigma, data):
    nll = -1*torch.distributions.Normal(loc = mu_true, scale = sigma).log_prob(data).sum()
    return nll

# Record nll values for different mu
nll_vals_sigma = []
for k in range(500):
    nll_vals_sigma.append(negloglikelihood(sigma_interval[k], data).detach())


# iii) Compare them

# If we print their values, we find them to be identical:
sigma_guess = pyro.get_param_store()['sigma'].detach()
elbo_val = elbo_fn(data)
nll_val = negloglikelihood(sigma_guess, data)
print('elbo_loss = {}, nll = {}'.format(elbo_val, nll_val))



"""
    6. Perform inference and compare
"""

# i) Set up optimization

# Optimization options
adam = pyro.optim.Adam({"lr": 0.01})
elbo = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(model, guide, adam, elbo)


# ii) Run Optimization in pyro

elbo_loss_sequence = []
n_steps = 1000
for step in range(n_steps):
    elbo_loss = svi.step(data)
    elbo_loss_sequence.append(elbo_loss)
    
    # Print out loss
    if step % 100 == 0:
        print('epoch: {} ; loss : {}'.format(step, elbo_loss))


# iii) Optimization using torch and likelihood function

sigma_nll = torch.tensor(1.0, requires_grad = True)
nll_loss_sequence = []
optimizer = torch.optim.Adam([sigma_nll], lr=0.01)      
for step in range(n_steps):
    # Set the gradients to zero
    optimizer.zero_grad()  
    
    # compute the loss function
    nll_loss = negloglikelihood(sigma_nll, data)
    nll_loss_sequence.append(nll_loss.detach())
    # compute the gradients, update t_mu
    nll_loss.backward()
    optimizer.step()
    
    if step % 100 == 0:
        print(f"Step {step+1}, Neg log likelihood {nll_loss.item()}, sigma_nll {sigma_nll.item()}")
    
# Final optimization results
print(" Optimization results sigma_nll = {} \n Analytical ML estimate sigma_nll = {}".
  format(sigma_nll.item(), torch.std(data)))



"""
    7. Plots and illustrations
"""


# i) Convergence of the optimization

fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# Plot ELBO loss on the first subplot
axs[0].plot(elbo_loss_sequence, label='ELBO Loss')
axs[0].set_title('ELBO Loss Over Time')
axs[0].set_xlabel('Iteration')
axs[0].set_ylabel('ELBO Loss')
axs[0].legend()
axs[0].grid(True)

# Plot NLL loss on the second subplot
axs[1].plot(nll_loss_sequence, label='NLL Loss')
axs[1].set_title('NLL Loss Over Time')
axs[1].set_xlabel('Iteration')
axs[1].set_ylabel('NLL Loss')
axs[1].legend()
axs[1].grid(True)

# Adjust layout to prevent overlap
plt.tight_layout()

# Show plot
plt.show()


# ii) Showcase elbo and nll for different values of the parameter sigma

fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# Plot ELBO loss for different vals of mu
axs[0].plot(sigma_interval, elbo_vals_sigma, label='ELBO Loss')
axs[0].set_title('ELBO Loss for different values of sigma')
axs[0].set_xlabel('sigma')
axs[0].set_ylabel('ELBO Loss')
axs[0].legend()
axs[0].grid(True)

# Plot NLL loss for different vals of sigma
axs[1].plot(sigma_interval, nll_vals_sigma, label='NLL Loss')
axs[1].set_title('NLL Loss for different values of sigma')
axs[1].set_xlabel('sigma')
axs[1].set_ylabel('NLL Loss')
axs[1].legend()
axs[1].grid(True)

# Adjust layout to prevent overlap
plt.tight_layout()

# Show plot
plt.show()







