#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is part of a series of tutorials that showcase bayesian concepts
and use pyro to perform inference and more complicated computations. During these
tutorials, we will tackle a sequence of problems of increasing complexity that 
will take us from fitting of distributional parameters to the computation of 
nontrivial posterior distributions.
The tutorials consist in the following:
    - fit a mean                            (tutorial_1) 
    - compute simple posterior              (tutorial_2)
    - fit mean and compute posterior        (tutorial_3)
    - fit params, multivariate posterior    (tutorial_4)


This script is to showcase a problem in which temperature measurements are 
performed with an unknown bias that is to be estimated. We will do this by
providing the analytical solution and also by emplying pyros inference 
machinery.
For this, do the following:
    1. Imports and definitions
    2. Generate synthetic data
    3. Analytic Maximum Likelihood
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
mu_disc = torch.linspace(-3, 3, n_disc)

torch.manual_seed(0)
pyro.set_rng_seed(0)



"""
    2. Generate synthetic data
"""


# i) Define params

T_true = 10         # 10 degree celsius (deg c) is underlying true termperature
mu_T = 1            # 1 deg c is underlying true bias of measurements
sigma_T = 1         # 1 deg c is standard deviation of measurements


# ii) Simulate data

dist_data = pyro.distributions.Normal(loc = T_true + mu_T, scale = sigma_T).expand([n_data])
data = dist_data.sample()



"""
    3. Analytic Maximum Likelihood
"""


# i)  Compute log likelihood as a function of bias mu

def neg_log_likelihood(mu):
    # Compute the shifted negative of the log likelihood
    # additive constants and scales dont matter for optimization
    residuals = data - T_true - mu
    squares =  torch.square(residuals)
    sum_of_squares = torch.sum(squares)
    
    return sum_of_squares


# ii) Find minimum; from analysis it is known to coincide with arithmetic mean

nll = torch.zeros([n_disc])
for k in range(n_disc):
    nll[k] = neg_log_likelihood(mu_disc[k])

min_index = torch.argmin(nll)
mu_mle_graphical = mu_disc[min_index]
mu_mle_analytical = torch.mean(data - T_true)



"""
    4. Pyro model and inference
"""


# i) Pyro model

def model(observations = None):
    bias_mu = pyro.param('bias_mu', torch.zeros([1]))
    dist_obs = pyro.distributions.Normal(loc = T_true + bias_mu, scale = 1)
    with pyro.plate('batch_plate', size = n_data, dim = -1):
        sample = pyro.sample('sample', dist_obs, obs = observations)
    return sample

# ii) Pyro guide

def guide(observations = None):
    pass


# iii) Pyro inference

adam = pyro.optim.NAdam({"lr": 0.01})
elbo = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(model, guide, adam, elbo)

loss_sequence = []
mu_sequence = []
for step in range(500):
    loss = svi.step(data)
    if step % 100 == 0:
        print('epoch: {} ; loss : {}'.format(step, loss))
    else:
        pass
    loss_sequence.append(loss)
    mu_sequence.append(pyro.get_param_store()['bias_mu'].item())



"""
    5. Plots and illustrations
"""

# i) Plot negative log likelihood 

plt.figure(1, dpi = 300)
plt.plot(mu_disc, nll)
plt.title('Shifted negative log likelihood')
plt.xlabel('bias mu')
plt.ylabel('implausibility of bias mu')


# ii) Training process

fig, ax = plt.subplots(1,2, figsize = (10,5))
ax[0].plot(loss_sequence)
ax[0].set_title('Loss during computation')

ax[1].plot(mu_sequence)
ax[1].set_title('Estimated mu during computation')


# iii) Print results

pyro_mle = pyro.get_param_store()['bias_mu'].item()
print(' True bias = {} \n MLE bias = {} \n pyro bias = {}'
      .format(mu_T, mu_mle_analytical.item(), pyro_mle))

