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

This script will build a stochastic model featuring control flow that depends
on an unobserved variable. We will perform inference on this model to derive 
the posterior density on the unobserved random variable.

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
import copy
import matplotlib.pyplot as plt



# ii) Definitions

n_data = 1000
torch.manual_seed(0)
pyro.set_rng_seed(0)



"""
    2. Generate data
"""


# i) Set up data distributions

mu_dec_true = torch.tensor([1.0])
mu_high_true = torch.tensor([3.0])
mu_low_true = torch.tensor([-3.0])
sigma = torch.tensor([1.0])

dist_decision_true = pyro.distributions.Normal(loc = mu_dec_true, scale = sigma)
dist_low_true = pyro.distributions.Normal(loc = mu_low_true, scale = sigma)
dist_high_true = pyro.distributions.Normal(loc = mu_high_true, scale = sigma)


# ii) Sample from dist to generate data. Decision sample decides on which 
# distribution to use next - then sample from dist_low or dist_high dependent 
# on decision_sample >=0 / <0

data = torch.zeros([n_data])
decision_true = torch.zeros([n_data])
for k in range(n_data):
    decision_true[k] = dist_decision_true.sample() >= 0
    if decision_true[k] == True:
        data[k] = dist_high_true.sample()
    else:
        data[k] = dist_low_true.sample()    


"""
    3. Define stochastic model
"""


# i) Define model by converting the above control flow into a function acting on
# tensors using pytorch and pyro primitives.

def model(observations = None):
    # Define parameters to be estimated
    mu_dec = pyro.param(name = 'mu_dec', init_tensor = torch.tensor([0.0]))
    # mu_high = pyro.param(name = 'mu_high', init_tensor = torch.tensor([3.0]))
    # mu_low = pyro.param(name = 'mu_low', init_tensor = torch.tensor([-3.0]))
    mu_high = mu_high_true
    mu_low = mu_low_true
    
    # Distributions
    dist_decision = pyro.distributions.Normal(loc = mu_dec, scale = sigma)
    dist_low = pyro.distributions.Normal(loc = mu_low, scale = sigma)
    dist_high = pyro.distributions.Normal(loc = mu_high, scale = sigma)

    # Control flow inside of independence context
    decision = torch.zeros([n_data])
    sample = torch.zeros([n_data])
    for index in pyro.plate('data_plate', size = n_data):
        # Unobserved random variable decision and observations
        decision[index] = pyro.sample('decision_{}'.format(index), dist_decision)
        observations_or_None = observations[index] if observations is not None else None
        
        # Control flow
        if decision[index] >= 0:
            sample[index] = pyro.sample('sample_{}'.format(index), dist_high, obs = observations_or_None)
        else: 
            sample[index] = pyro.sample('sample_{}'.format(index), dist_low, obs = observations_or_None)
    
    return sample


# Illustrate model by plotting symbolic representation
# pyro.render_model(model, model_args=(), render_distributions=True, render_params=True)

untrained_sample = copy.copy(model())



"""
    4. Perform inference
"""


# i) Set up guide

def guide(observations = None):
    # Define parameters to be estimated
    mu_dec_guide = pyro.param(name = 'mu_dec_guide', init_tensor = torch.tensor([0.0]))
    
    # Distributions
    dist_decision = pyro.distributions.Normal(loc = mu_dec_guide, scale = sigma)
    
    # Sampling decisions
    decision = torch.zeros([n_data])
    for index in pyro.plate('data_plate', n_data):
        decision[index] = pyro.sample('decision_{}'.format(index), dist_decision)
    
    return decision


# ii) Set up inference


adam = pyro.optim.NAdam({"lr": 0.1})
elbo = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(model, guide, adam, elbo)


# iii) Perform svi

for step in range(100):
    loss = svi.step(data)
    if step % 10 == 0:
        print('epoch: {} ; loss : {}'.format(step, loss))
    else:
        pass


# iv) Sample trained model

trained_sample = copy.copy(model())



"""
    5. Plots and illustrations
"""


# i) Print results

for name, value in pyro.get_param_store().items():
    print(name, pyro.param(name).data.cpu().numpy())
    
print('mu_dec_true = {} \n mu_high_true = {} \n mu_low_true = {}'
      .format(mu_dec_true, mu_high_true, mu_low_true))


# ii) Plot distributions

# Creating the figure and subplots
fig, axs = plt.subplots(3, 1, figsize=(8, 12), dpi = 300)

axs[0].hist(data.detach().numpy(), bins=30, color='skyblue', edgecolor='black')
axs[0].set_title('Histogram of data')
axs[1].hist(untrained_sample.detach().numpy(), bins=30, color='lightgreen', edgecolor='black')
axs[1].set_title('Sampling from untrained model')
axs[2].hist(trained_sample.detach().numpy(), bins=30, color='salmon', edgecolor='black')
axs[2].set_title('Sampling from trained model')

plt.tight_layout()
plt.show()






def model(data = None):
    mu_dec = pyro.param("mu_dec", torch.tensor([1.0]))
    mu_high = torch.tensor([3.0])
    mu_low = torch.tensor([-3.0])
    sigma = torch.tensor([1.0])
    
    with pyro.plate("data_plate", size=n_data):
        # First sample the decision variable
        decision = pyro.sample("decision", pyro.distributions.Normal(mu_dec, sigma)) >= 0
        
        # Use the decision to choose the distribution from which to sample the observed data
        data_dist = pyro.distributions.Normal(mu_high * decision.float() + mu_low * (~decision).float(), sigma)
        obs = pyro.sample("obs", data_dist, obs=data)
# pyro.render_model(model, model_args=(), render_distributions=True, render_params=True)
def guide(data = None):
    mu_dec_guide = pyro.param("mu_dec_guide", torch.tensor([0.0]))
    
    with pyro.plate("data_plate", size=n_data):
        decision_guide = pyro.sample("decision", pyro.distributions.Normal(mu_dec_guide, sigma))
# pyro.render_model(guide, model_args=(), render_distributions=True, render_params=True)      
 
# guide = pyro.infer.autoguide.AutoNormal(model)
        
adam_params = {"lr": 0.01}
optimizer = pyro.optim.Adam(adam_params)

# Setup the inference algorithm
svi = pyro.infer.SVI(model, guide, optimizer, loss=pyro.infer.Trace_ELBO())

n_steps = 1000
# Do gradient steps
for step in range(n_steps):
    loss = svi.step(data)
    if step % 100 == 0:
        print(f"Step {step} : loss = {loss}")    
        
for name, value in pyro.get_param_store().items():
    print(name, pyro.param(name).data.cpu().numpy())
    
print('mu_dec_true = {} \n mu_high_true = {} \n mu_low_true = {}'
      .format(mu_dec_true, mu_high_true, mu_low_true))