#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is part of a series of investigations to develop an intuition on how
to interpret the posterior density of a latent variable and how to construct this
density with the help of a guide.  The investigation will extend to cover multiple
phenomena that deal with the interpretation, computation, and representation of
the posterior.
The investigations consist in the following:
    - Simplified posterior by ignoring prior
    - Simple posterior for mean estimation problem
    - The impact of the variational distribution
    - ELBO mediating between likelihood and posterior
    - A multivariate posterior
    - Model, guide, and posterior featuring ANN's
You will find these investigations in individual scripts, each focusing on the
illumination of one specific phenomenon.

In this script, we showcase how the posterior density can be interpreted in a 
very simple example featuring an unknown mean that is sampled noisily. When we
tune out the effect of the prior density, the posterior is just the density of
the estimator of the mean - an object that is very familiar in the context of
estimation theory. For this, do the following:
    1. Definitions and imports
    2. Simulate some data
    3. Define the model
    4. Define the guide
    5. Perform inference
    6. Investigate and compare
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


import matplotlib.pyplot as plt


# ii) Definitions

np.random.seed(0)
torch.manual_seed(0)

n_data = 1
interval = torch.linspace(-2,5,500)



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


# i) Model definition
#
# The model consists in a prior density and a likelihood. Sampling mu from a 
# prob dist communicates to pyro that mu is a random variable and establishes
# the prior for mu. Then passing it to another distribution and observing the 
# subsequent samples establishes the likelihood function.

def model(observations = None):
    # Sample one mu from the prior distribution
    mu_prior_dist = pyro.distributions.Normal(0,1)
    # Mask out the log probs of the prior
    with pyro.poutine.mask(mask=torch.tensor(False)):
        mu = pyro.sample('mu', mu_prior_dist)
    
    # Noisily sample the mu
    obs_dist = pyro.distributions.Normal(mu, sigma_true)
    with pyro.plate('batch_plate', size = n_data, dim =-1):
        obs = pyro.sample('obs', obs_dist, obs = observations)
    
    return obs

# The above model masks out the prior distribution. The effect is that the prior
# does not contribute anymore to the logprobs passed to the optimization. In effect
# this is similar to putting an improper uniform prior over the random variable
# mu. Nonetheless, mu stays an unobserved (latent) random variable, whose posterior
# is approximated by the guide function below. 



"""
    4. Define the guide
"""


# i) Guide definition
#
# The guide function determines the variational distribution used to approximate
# the posterior density. We employ herre a Normal distribution with unknown mean
# and standard deviation. Optimizing these parameters w.r.t. the elbo implies
# choosing mean and standard deviation of the distribution mu_post_dist such that
# the resultant distribution most colesly matches the true posterior. 

def guide(observations = None):
    # Set up parameters for posterior density
    mean_post = pyro.param('mu_post', init_tensor = torch.tensor(0.0))
    sigma_post = pyro.param('sigma_post', init_tensor = torch.tensor(1.0), 
                            constraint = pyro.distributions.constraints.positive)
    
    # Sample from density to produce sample of mu_post
    mu_post_dist = pyro.distributions.Normal(mean_post, sigma_post)
    mu_post = pyro.sample('mu', mu_post_dist)
    
    return mu_post

# The above guide produces a single sample of a scalar random variable mu_post.
# The random variable is normally distributed with the paramters mu_post, sigma_post
# Sample sites with identical names in model and guide (like mu here) communicate
# to pyro that the samples drawn in the guide should represent the posterior density
# of mu with mu distributed as indicated in the model. Subsequently the following
# happens during optimization:
#   1. mu_post, sigma post are the only adjustable parameters. Scalar random var mu 
#       is unobserved and its posterior needs to be constructed.
#   2. mu_post, sigma post are interpreted as mean and std of variational distribution
#   3. mu_post, sigma_post are adjusted to be close to the true posterir as measured by KLD
# The true posterior is incidentally also a normal distribution, but with mean
# equal to the arithmetic mean and stt = sigma/sqrt(n_data). This allows checking
# the results easily.


"""
    5. Perform inference
"""


# i) Set up optimization

# Optimization options
adam = pyro.optim.Adam({"lr": 0.01})
elbo = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(model, guide, adam, elbo)


# ii) Run Optimization

loss_sequence = []
n_steps = 1000
for step in range(n_steps):
    loss = svi.step(data)
    loss_sequence.append(loss)
    
    # Print out loss
    if step % 100 == 0:
        print('epoch: {} ; loss : {}'.format(step, loss))



"""
    6. Investigate and compare
"""


# i) Compute distribution of estimator - analytically

mu_analytical = torch.mean(data)
sigma_analytical = sigma_true / torch.sqrt(torch.tensor(n_data))
estimator_distribution_analytical = pyro.distributions.Normal(mu_analytical, sigma_analytical).log_prob(interval).exp()

# ii) Compute distribution of estimator - sample from guide

guide_samples = []
for k in range(10000):
    guide_samples.append(guide().detach())
guide_samples = torch.tensor(guide_samples).numpy()


# iii) Compute distribution of estimator - use the params

mu_param = pyro.get_param_store()['mu_post'].detach()
sigma_param = pyro.get_param_store()['sigma_post'].detach()
estimator_distribution_param = pyro.distributions.Normal(mu_param, sigma_param).log_prob(interval).exp()



"""
    7. Plots and illustrations
"""


# i) Showcase the optimization

plt.figure(1, dpi = 300)
plt.plot(loss_sequence)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss during optimization')


# ii) Print out the inferred and true values

# True values and analytical estimates
print('mu_true = ', mu_true)
print('mu_estimator_analytical = ', mu_analytical)
print('mu_estimator_sigma_analytical = ', sigma_analytical)

# Parameters from optimization
for name, param in pyro.get_param_store().items():
    print(name, param)


# iii) Compare the three densities

plt.figure(2, dpi = 300)
plt.hist(guide_samples, bins = 30, density = True, label = 'guide samples')
plt.plot(interval, estimator_distribution_analytical, label = 'analytical')
plt.plot(interval, estimator_distribution_param, label = 'parameters')

plt.xlabel('mu')
plt.ylabel('Probability')
plt.title('Posterior densities')
plt.legend()






























