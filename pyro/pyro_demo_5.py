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

This script will build a stochastic model that involves a multivariate Gaussian
distribution whose mean and variance depend on the output of an ANN. A dataset
with complicated shape is used to train the composition of ANN and Gaussian. The
result of the training is a complex stochastic model showcasing dependence of
mean and variance on explanatory data. 

For this, do the following:
    1. Imports and definitions
    2. Generate data
    3. Define ANN and helper functions
    4. Define stochastic model
    5. Perform inference
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

n_dim = 2
n_data = 500

torch.manual_seed(0)
pyro.set_rng_seed(0)

torch.set_default_tensor_type(torch.DoubleTensor)



"""
    2. Generate data
"""
 

# i) Prepare data generation by initializing nonlinear functions

def complicated_function(random_input):
    first_transform = torch.sin(random_input) 
    second_transform = torch.tanh(first_transform)
    result_1 = second_transform
    result_2 = torch.relu(second_transform) + 1e-3
    return result_1, torch.diag_embed(result_2)


# ii) Define distributional parameters and simulate

# dist_random_input = pyro.distributions.Uniform(-1, 1)
# random_input = dist_random_input.sample(sample_shape = [n_data,n_dim])
dist_random_input = pyro.distributions.Normal(loc = torch.zeros([n_data, n_dim]), scale = torch.ones([1, n_dim]))
random_input = dist_random_input.sample()
mean, cov = complicated_function(random_input)

dist_output = pyro.distributions.MultivariateNormal(loc = mean, covariance_matrix = cov)
data = dist_output.sample()



"""
    3. Define ANN and helper functions
"""


# i) ANN class

class ANN(pyro.nn.PyroModule):
    # Initialize by invoking base class
    def __init__(self, dim_input, dim_hidden, dim_output):
        super().__init__()
        
        # Integration basic data
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        
        self.mean_output_dims = range(dim_output[0])
        self.cov_output_dims = range(dim_output[0],sum(dim_output))
        
        # Linear transforms
        self.fc_1 = torch.nn.Linear(self.dim_input, self.dim_hidden)
        self.fc_2 = torch.nn.Linear(self.dim_hidden, self.dim_hidden)
        self.fc_3 = torch.nn.Linear(self.dim_hidden, self.dim_output[0] + self.dim_output[1])
        # nonlinear transforms
        self.nonlinear = torch.nn.ReLU()
        
    def forward(self, x):
        # Define forward computation on the input data x
        # Shape the minibatch so that batch_dims are on left, argument_dims on right
        x = x.reshape([-1, self.dim_input])
        
        # Then compute hidden units and output of nonlinear pass
        hidden_units_1 = self.nonlinear(self.fc_1(x))
        hidden_units_2 = self.nonlinear(self.fc_2(hidden_units_1))
        feature_mat = self.fc_3(hidden_units_2)
        mean = feature_mat[:,self.mean_output_dims]
        cov = torch.diag_embed(torch.exp(feature_mat[:,self.cov_output_dims]))
        return mean, cov


# ii) Invocations and constructions

dim_latent_z = 1
model_decoder = ANN(dim_latent_z, 5, [n_dim,n_dim])
guide_encoder = ANN(n_dim, 5, [dim_latent_z, dim_latent_z])



"""
    4. Define stochastic model
"""


# i) Define model

def model(observations = None):
    # Register ANN module
    pyro.module("model_decoder", model_decoder)
        
    # Define distribution and sample latent_z
    latent_z_dist = pyro.distributions.Normal(loc = torch.zeros([n_data, dim_latent_z]),
                                              scale = torch.ones([n_data, dim_latent_z])).to_event(1)
    
    # Define sample procedure: Sample z, transform it, sample observations
    with pyro.plate("batch_plate", size = n_data, dim = -1):
        latent_z = pyro.sample("latent_z", latent_z_dist)
        transformed_mu, transformed_sigma = model_decoder(latent_z)
        
        sample_dist = pyro.distributions.MultivariateNormal(loc = transformed_mu, 
                            covariance_matrix = transformed_sigma)     
        sample = pyro.sample("sample", sample_dist, obs = observations)
    
    return sample

untrained_sample = copy.copy(model())


# iii) Define guide

def guide(observations = None):
    # Register ANN module
    pyro.module("guide_encoder", guide_encoder)
    
    # Apply ANN to define latent_z distribution parameters
    latent_z_mu, latent_z_sigma = guide_encoder(observations)
    latent_z_dist = pyro.distributions.MultivariateNormal(loc = latent_z_mu,
                        covariance_matrix = latent_z_sigma)
    
    # Define sample procedure for latent variables
    with pyro.plate("batch_plate", size = n_data, dim = -1):
        latent_z = pyro.sample("latent_z", latent_z_dist)
    
    return latent_z



"""
    5. Perform inference
"""


# i) Set up inference


adam = pyro.optim.NAdam({"lr": 0.03})
elbo = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(model, guide, adam, elbo)


# ii) Perform svi

for step in range(1000):
    loss = svi.step(data)
    if step % 100 == 0:
        print('epoch: {} ; loss : {}'.format(step, loss))
    else:
        pass


# iii) Sample trained model

trained_sample = copy.copy(model())



"""
    6. Plots and illustrations
"""


# i) Illustrate model and guide 
# (execute following lines one by one in console to render)

pyro.render_model(model, model_args=(), render_distributions=True, render_params=True)
pyro.render_model(guide, model_args=(data,), render_distributions=True, render_params=True)


# ii) Plot distributions

# Plotting the realizations
fig1, axs = plt.subplots(3, 1, figsize=(8, 12), dpi=300)

axs[0].scatter(data[:,0].detach(), data[:,1].detach())
axs[0].set_title('Original data')
axs[0].set_xlabel('x_1')
axs[0].set_ylabel('x_2')

axs[1].scatter(untrained_sample[:,0].detach(), untrained_sample[:,1].detach())
axs[1].set_title('Untrained model')
axs[1].set_xlabel('x_1')
axs[1].set_ylabel('x_2')

axs[2].scatter(trained_sample[:,0].detach(), trained_sample[:,1].detach())
axs[2].set_title('Trained model')
axs[2].set_xlabel('x_1')
axs[2].set_ylabel('x_2')

plt.tight_layout()
plt.show()


# Plotting the distribution of latent_z's
x_obs_0 = torch.tensor([0.0,0.0]).reshape([1,2])
x_obs_1 = torch.tensor([1.0,1.0]).reshape([1,2])
x_obs_2 = torch.tensor([-1.0,-1.0]).reshape([1,2])
latent_z_0 = guide(x_obs_0)
latent_z_1 = guide(x_obs_1)
latent_z_2 = guide(x_obs_2)


fig2 = plt.figure(figsize=(6, 4), dpi=300)

plt.hist(latent_z_0.detach().numpy(), density = True, label = 'observed x = [0,0]')
plt.hist(latent_z_1.detach().numpy(), density = True, label = 'observed x = [1,1]')
plt.hist(latent_z_2.detach().numpy(), density = True, label = 'observed x = [-1,-1]')
plt.title("Histograms for latent_z's conditioned on observations")
plt.legend()

plt.tight_layout()
plt.show()



