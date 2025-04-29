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

In this script, we showcase how the elbo loss can be interpreted as producing an
approximation to a posterior in the simple context of estimating the mean parameter
of a normal distribution. We compare the value of the elbo loss to the penalty
function representing KLD between variational distribution and posterior distribution
and explain the differences.
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


# ii) Definitions

n_meas = 10

n_mu_disc = 100
n_sigma_disc = 100

mu_interval = torch.linspace(0,5,n_mu_disc)
sigma_interval = torch.linspace(0.5, 5, n_sigma_disc)



"""
    2. Simulate some data
"""


# i) Set up sample distributions

mu_prod_true = torch.tensor(1.0)
sigma_prod_true = torch.tensor(0.3)
sigma_meas_true = torch.tensor(0.01)



# ii) Sample from distributions

prod_distribution = pyro.distributions.Normal(mu_prod_true, sigma_prod_true)
prod_lengths = prod_distribution.sample()

data_distribution = pyro.distributions.Normal(prod_lengths, sigma_meas_true)
data = data_distribution.sample([n_meas])

# Interpret the data as some products being produces by an uncertain production
# process with a varying length. That length stays constant for each product
# but may vary between products. However, when measuring the product lengths
# with a measurement device, all measurements are subject to some randomness
# leading to length measurements that are unequal even for the same product.


"""
    3. Define the model
"""

# i) Define the model
#
# The model will not feature any unknown parameters; instead it will consist
# in a two-stage sampling process with the first sample being an unobserved latent
# serving as the mean of the second stage of samples. The latter ones are observed.
# In all cases the distributions are normal and the scales are assumed known
# leading us to have the simplest nontrivial model still featuring latents and
# therefore a nontrivial guide and variational approximation procedure.

def model(observations = None):
    # Set up distribution for mu, the second stage mean
    mu_prior_dist = pyro.distributions.Normal(loc = torch.tensor(0.0), scale = torch.tensor(1.0))
    
    # Sample scalar mu and set up observation dist
    mu = pyro.sample('mu', mu_prior_dist)
    obs_dist = pyro.distributions.Normal(mu, sigma_meas_true).expand([n_meas])
    
    # Take independent samples from observation distribution
    with pyro.plate('batch_plate', size = n_meas, dim =-1):
        obs = pyro.sample('obs', obs_dist, obs = observations)

    return obs




"""
    4. Define the guide
"""


# i) Define the guide

def guide(observations = None):
    # Set up parameters for posterior density
    mean_post = pyro.param('mu_post', init_tensor = torch.tensor(0.0))
    sigma_post = pyro.param('sigma_post', init_tensor = torch.tensor(1.0), 
                            constraint = pyro.distributions.constraints.positive)
    
    # Sample from density to produce sample of mu_post
    mu_post_dist = pyro.distributions.Normal(mean_post, sigma_post)
    mu_post = pyro.sample('mu', mu_post_dist)
        
    return mu_post

# Run once to initialize params
untrained_posterior = guide()



"""
    5. Investigate losses
"""


# i) Investigate elbo
#
# ELBO = Evidence lower bond
# L(theta, phi, x) = E_q_phi[log p_theta(x,z)] - E_q_phi[log q_phi(z|x)]
#                  = log p_theta(x) - D_KL(q_phi(z|x) || p_theta(z|x))
# In our specific case, we have trivial theta but nontrivial x,z, phi. The elbo
# reduces basically to the Kullback Leibler divergence. The DKL term becomes 
# D_KL(q_phi(z|x) || p(z|x)) with the actual true posterior not depending on any
# parameters. The term log p_theta(x) = log int p_theta(x|z)p_theta(z)dz simplifies
# to log p(x) which is difficult to compute due to the integration against against
# the latent z. However, it does not depend on any adjustable parameter and is
# therefore simply a constant that has no impact on the optimization. Consequently,
# The only parameters to be adjusted are the parameters phi; they are to be chosen
# to minimize the divergence between q_phi(z|x) and the true posterior p(z|x).

# Lets evaluate the elbo and tie it to likelihood, evidence, KL divergence.

elbo = pyro.infer.Trace_ELBO()
def elbo_fn(data):
    elbo_val = elbo.differentiable_loss(model, guide, data)
    return elbo_val

# Estimate elbo loss for different values of mu
elbo_vals_grid = torch.zeros([n_mu_disc, n_sigma_disc])
for k in range(n_mu_disc):
    for l in range(n_sigma_disc):
        pyro.param('mu_post').data = copy.deepcopy(mu_interval[k])
        pyro.param('sigma_post').data = copy.deepcopy(sigma_interval[k])
        elbo_vals_grid[k,l] = elbo.differentiable_loss(model, guide, data).detach()

# The negative log likelihood is hard to compute but constant. We will ignore it
# for now. The Kullback Leibler divergence is a nonnegative measure for the difference
# between two probability distributions. It can be written as 
# DKL   = DKL(q_phi(z|x) || p_theta(z|x)) 
#       = DKL(q_phi(z|x) || p(z|x))
#       = E_q_phi[ log (q_phi(z|x)/p(z|x))]
#       = E_q_phi[log q_phi(z|x) - log p(z|x)]
# The difference of the logs of the probabilities is known to be a powerful indicator
# of statistical distance between distributions (see log likelihood test and Neyman
# Pearson Lemma). Clearly, it is 0 if q_phi and p are equal, i.e. if the variational
# distribution approximates the posterior perfectly. It is also nonnegative, i.e.
# every other situation leads to a larger value of DKL therefore establishing it
# as a useful loss in this case.


# ii) Investigate DKL

# Define DKL loss and compute it for different values of mu, sigma
# Can be estimated e.g. by sampling z from the guide multiple times, then computing
# for each sample log q_phi(z|x) - log p(z|x). Averaging the samples leads to an
# unbiased Monte Carlo estimate of the DKL.
def dkl(mu, sigma, data):
    
    
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







