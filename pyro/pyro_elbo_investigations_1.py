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

n_data = 1
mu_interval = torch.linspace(-2,5,500)



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

# Run once to initialize params
untrained_simulation = model()


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
    5. Investigate losses
"""

# i) Investigate elbo
#
# ELBO = Evidence lower bond
# L(theta, phi, x) = E_q_phi[log p_theta(x,z)] - E_q_phi[log q_phi(z|x)]
#                  = log p_theta(x) - D_KL(q_phi(z|x) || p_theta(z|x))
# The last line is great for understanding maximization of the elbo to aim for
# maximization of the evidence p_theta(x) and the minimization of the KL divergence
# between true posterior and variational distribution.
# The first line gives some hints at how the elbo might be estimated -  sampling
# from the guide and forming the log prob sums of model and guide provide individual
# estimators of the elbo that might be averaged together over many samples to
# provide a more reliable estimate.

# Lets evaluate the elbo and tie it to likelihood, evidence, KL divergence.

elbo = pyro.infer.Trace_ELBO()
def elbo_fn(data):
    elbo_val = elbo.differentiable_loss(model, guide, data)
    return elbo_val

# Estimate elbo loss for different values of mu
elbo_vals_mu = []
for k in range(500):
    pyro.param('mu').data = copy.deepcopy(mu_interval[k])
    elbo_vals_mu.append(elbo.differentiable_loss(model, guide, data).detach())

# The negative log likelihood is the same as the negative log evidence in this 
# case as there are no latents and therefore
# p_theta(x) = int p_theta(x,z) dz          (evidence)
#            = int p_theta(x|z) p(z)dz      (factorization)
#            = p_theta(x|z) int p(z)dz      (since no dependence)
#            = p_theta(x|z)                 (likelihod)
# where theta = params, x = data, and z = latents are empty .


# ii) Investigate nll

# Defne nll and log evidence and compute nll loss for different values of mu
def negloglikelihood(mu, data):
    nll = -1*torch.distributions.Normal(loc = mu, scale = 1).log_prob(data).sum()
    return nll

# Record nll values for different mu
nll_vals_mu = []
for k in range(500):
    nll_vals_mu.append(negloglikelihood(mu_interval[k], data).detach())

def logevidence(mu,data):
    logevi = torch.distributions.Normal(loc = mu, scale = 1).log_prob(data).sum()
    return logevi


# iii) Compare them

# If we print their values, we find them to be identical:
mu_guess = pyro.get_param_store()['mu'].detach()
elbo_val = elbo_fn(data)
nll_val = negloglikelihood(mu_guess, data)
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
#
# The same optimization should be executable by maximizing the likelihood by
# passing the negloglikelihood function to the optimizer.

mu_nll = torch.tensor(0.0, requires_grad = True)
nll_loss_sequence = []
optimizer = torch.optim.Adam([mu_nll], lr=0.01)      
for step in range(n_steps):
    # Set the gradients to zero
    optimizer.zero_grad()  
    
    # compute the loss function
    nll_loss = negloglikelihood(mu_nll, data)
    nll_loss_sequence.append(nll_loss.detach())
    # compute the gradients, update t_mu
    nll_loss.backward()
    optimizer.step()
    
    if step % 100 == 0:
        print(f"Step {step+1}, Neg log likelihood {nll_loss.item()}, mu_nll {mu_nll.item()}")
    
# Final optimization results
print(" Optimization results mu_nll = {} \n Analytical ML estimate mu_nll = {}".
  format(mu_nll.item(), torch.mean(data)))



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


# ii) Showcase elbo and nll for different values of the parameter mu

fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# Plot ELBO loss for different vals of mu
axs[0].plot(mu_interval, elbo_vals_mu, label='ELBO Loss')
axs[0].set_title('ELBO Loss for different values of mu')
axs[0].set_xlabel('mu')
axs[0].set_ylabel('ELBO Loss')
axs[0].legend()
axs[0].grid(True)

# Plot NLL loss for different vals of mu
axs[1].plot(mu_interval, nll_vals_mu, label='NLL Loss')
axs[1].set_title('NLL Loss for different values of mu')
axs[1].set_xlabel('mu')
axs[1].set_ylabel('NLL Loss')
axs[1].legend()
axs[1].grid(True)

# Adjust layout to prevent overlap
plt.tight_layout()

# Show plot
plt.show()







