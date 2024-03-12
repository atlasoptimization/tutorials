#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script provides a task related to the training of a simple neural network
for the purpose of classifying some timeseries. An exemplary solution is also 
provided; however the idea of the script is to use the commands and building
blocks found in the pytorch_tutorial series to assemble a solution. The goal
is for you to learn to set up a neural network, define a loss function, train
the resulting net, investigate performance, and illustrate the features learned
by the net.

For this, you will do the following:
    1. Imports and definitions
    2. Generate data
    3. Setup neural network class       <-- your task
    4. Train neural network             <-- your task
    5. Investigate activation triggers  <-- your task
    6. Plots and illustrations
    
The script is meant solely for educational and illustrative purposes. Written by
Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.com.
"""



"""
    1. Imports and definitions
"""


# i) Imports

import torch
import numpy as np
import copy
import matplotlib.pyplot as plt


# ii) Definitions

n_time = 100
n_data = 10
time = torch.linspace(0,1,n_time)


# iii) Global provisions

torch.manual_seed(0)
torch.set_default_tensor_type(torch.DoubleTensor)



"""
    2. Generate data
"""


# The data consists of timeseries that feature a randomly chosen slope and some
# noise added on top. These timeseries can be interpreted as representing a 
# phenomenon of your own choosing (stock prices, deformations, workhours, ...). 
# We assign class labels depending on the slope of the timeseries.

# i) Generate and classify timeseries

timeseries = torch.zeros([n_data, n_time])
labels = torch.zeros([n_data, 1])

for k in range(n_data):
    random_slope = torch.distributions.Uniform(-2,2).sample()
    random_noise = torch.distributions.Normal(0, 0.5).sample([n_time])
    
    labels[k,0] = random_slope >=0
    timeseries[k,:] = random_slope * time + random_noise


# ii) Classify timeseries




# The rule for classifying the timeseries is not available in practice and we 
# employ it here only to have some simple synthetic dataset that we fully understand.
# From now on the rule is considered unknown and has to be learned from data. 


# Consider everything that has happened up till this line to be result of some
# black box process found in nature. Only the data is available to you.
# ----------------------------------------------------------------------------




"""
    3. Auxiliary definitions VAE
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
        
        # Linear transforms
        self.fc_1 = torch.nn.Linear(self.dim_input, self.dim_hidden)
        self.fc_2 = torch.nn.Linear(self.dim_hidden, self.dim_hidden)
        self.fc_3 = torch.nn.Linear(self.dim_hidden, self.dim_output)
        # nonlinear transforms
        self.nonlinear = torch.nn.ReLU()
        
    def forward(self, x):
        # Define forward computation on the input data x
        # Shape the minibatch so that batch_dims are on left, argument_dims on right
        x = x.reshape([-1, self.dim_input])
        
        # Then compute hidden units and output of nonlinear pass
        hidden_units_1 = self.nonlinear(self.fc_1(x))
        hidden_units_2 = self.nonlinear(self.fc_2(hidden_units_1))
        output = self.fc_3(hidden_units_2)

        return output





"""
    4. Set up entire VAE
"""



"""
    5. Perform inference
"""




"""
    6. Plots and illustrations
"""


# i) Illustrate model and guide 
# (execute following lines one by one in console to render)

pyro.render_model(vae.model, model_args=(data,), render_distributions=True, render_params=True)
pyro.render_model(vae.guide, model_args=(data,), render_distributions=True, render_params=True)


# ii) Plot training progress

fig = plt.figure(1, figsize = (10,5))
plt.plot(loss_sequence)
plt.title('Loss during computation')


# iii) Showcase data

fig, ax = plt.subplots(3,3, figsize = (10,10), dpi = 300)
ax[0,0].imshow(data[0,:].reshape([n_space,n_space]))
ax[0,0].set_title('Original data - type 1')
ax[0,1].imshow(data[1,:].reshape([n_space,n_space]))
ax[0,2].imshow(data[2,:].reshape([n_space,n_space]))

ax[1,0].imshow(data[n_data,:].reshape([n_space,n_space]))
ax[1,0].set_title('Original data - type 2')
ax[1,1].imshow(data[n_data+1,:].reshape([n_space,n_space]))
ax[1,2].imshow(data[n_data+2,:].reshape([n_space,n_space]))

ax[2,0].imshow(data[2*n_data,:].reshape([n_space,n_space]))
ax[2,0].set_title('Original data - type 3')
ax[2,1].imshow(data[2*n_data+1,:].reshape([n_space,n_space]))
ax[2,2].imshow(data[2*n_data+2,:].reshape([n_space,n_space]))

plt.tight_layout()
plt.show()


# iv) Showcase trained and untrained resimulations

fig, ax = plt.subplots(3,7, figsize = (15,5), dpi = 300)
# first row, type I
ax[0,0].imshow(data[0,:].reshape([n_space,n_space]))
ax[0,0].set_title('Original data - type 1')
ax[0,1].imshow(images_resimu_untrained_type1[0,:,:])
ax[0,1].set_title('Resimulations untrained')
ax[0,2].imshow(images_resimu_untrained_type1[1,:,:])
ax[0,3].imshow(images_resimu_untrained_type1[2,:,:])
ax[0,4].imshow(images_resimu_trained_type1[0,:,:])
ax[0,4].set_title('Resimulations trained')
ax[0,5].imshow(images_resimu_trained_type1[1,:,:])
ax[0,6].imshow(images_resimu_trained_type1[2,:,:])

# second row, type II
ax[1,0].imshow(data[n_data,:].reshape([n_space,n_space]))
ax[1,0].set_title('Original data - type 2')
ax[1,1].imshow(images_resimu_untrained_type2[0,:,:])
ax[1,1].set_title('Resimulations untrained')
ax[1,2].imshow(images_resimu_untrained_type2[1,:,:])
ax[1,3].imshow(images_resimu_untrained_type2[2,:,:])
ax[1,4].imshow(images_resimu_trained_type2[0,:,:])
ax[1,4].set_title('Resimulations trained')
ax[1,5].imshow(images_resimu_trained_type2[1,:,:])
ax[1,6].imshow(images_resimu_trained_type2[2,:,:])

# third row, type III
ax[2,0].imshow(data[2*n_data,:].reshape([n_space,n_space]))
ax[2,0].set_title('Original data - type 3')
ax[2,1].imshow(images_resimu_untrained_type3[0,:,:])
ax[2,1].set_title('Resimulations untrained')
ax[2,2].imshow(images_resimu_untrained_type3[1,:,:])
ax[2,3].imshow(images_resimu_untrained_type3[2,:,:])
ax[2,4].imshow(images_resimu_trained_type3[0,:,:])
ax[2,4].set_title('Resimulations trained')
ax[2,5].imshow(images_resimu_trained_type3[1,:,:])
ax[2,6].imshow(images_resimu_trained_type3[2,:,:])

plt.tight_layout()
plt.show()


# vi) Showcase distribution of hidden variables

z_loc, z_scale = vae.encoder(data)[:,0:dim_bottleneck].detach().numpy(),\
        vae.encoder(data)[:,dim_bottleneck : 2*dim_bottleneck].detach().numpy()
fig, ax = plt.subplots(2,1, figsize = (10,10), dpi = 300)
ax[0].hist(z_loc)
ax[0].set_title('Distribution of latent variables; colors indicating dimension')
ax[1].scatter(z_loc[:,0], z_loc[:,1])
ax[1].set_title('Scatterplot of latent variables')
