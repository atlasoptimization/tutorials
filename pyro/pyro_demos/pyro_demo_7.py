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

This script will train a variational autoencoder to learn the structure inherent
in a dataset that contains images of three different types of classes: smooth,
oscillating, and noise. The VAE will be used to generate new images based on old
images, the internal rule for classification and the discriminative features used
inside of the VAE will be investigated.

For this, do the following:
    1. Imports and definitions
    2. Generate data
    3. Auxiliary definitions VAE
    4. Set up entire VAE
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

n_space = 10
n_total = n_space ** 2
n_data = 100
x = torch.linspace(0,1,n_space)
y = torch.linspace(0,1,n_space)
xx,yy = torch.meshgrid((x,y), indexing = 'ij')
locs = torch.vstack((xx.unsqueeze(0),yy.unsqueeze(0))).reshape([2,n_total])


# iii) Global provisions

torch.manual_seed(0)
pyro.set_rng_seed(0)

torch.set_default_tensor_type(torch.DoubleTensor)



"""
    2. Generate data
"""
 
# i) Distributional parameters for smooth, oscillation, noise random fields

mu_true = torch.zeros([1,n_total])
sigma_smooth = torch.zeros([n_total,n_total])
sigma_osc = torch.zeros([n_total,n_total])
sigma_noise = torch.zeros([n_total,n_total])

def cov_fun_smooth(s,t):
    cov_val = torch.exp(-(torch.norm(s-t)/0.4)**2)
    return cov_val
def cov_fun_osc(s,t):
    cov_val = torch.cos(3*torch.pi*torch.norm(s)) * torch.cos(3*torch.pi*torch.norm(t))
    # cov_val = torch.cos(3*torch.pi*s[0]) * torch.cos(3*torch.pi*t[0]) \
    #     + torch.cos(3*torch.pi*s[1]) * torch.cos(3*torch.pi*t[1])
    # cov_val = torch.exp(-(torch.norm(s-t)/0.2)**2)
    return cov_val

for k in range(n_total):
    for l in range(n_total):
        sigma_smooth[k,l] = cov_fun_smooth(locs[:,k], locs[:,l])
        sigma_osc[k,l] = cov_fun_osc(locs[:,k], locs[:,l])
sigma_noise = torch.eye(n_total)
sigma_reg = 1e-3*torch.eye(n_total)


# ii) Define and sample from distribution

extension_tensor = torch.ones([n_data,1])
dist_smooth = pyro.distributions.MultivariateNormal(loc = mu_true * extension_tensor,
                                                  covariance_matrix = sigma_smooth + sigma_reg)
dist_osc = pyro.distributions.MultivariateNormal(loc = mu_true * extension_tensor,
                                                  covariance_matrix = sigma_osc + sigma_reg)
dist_noise = pyro.distributions.MultivariateNormal(loc = mu_true * extension_tensor,
                                                  covariance_matrix = sigma_noise + sigma_reg)
data_smooth = dist_smooth.sample()
data_osc = dist_osc.sample()
data_noise = dist_noise.sample()

data = torch.vstack((data_smooth, data_osc, data_noise))



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


# ii) Set up encoder and decoder

dim_hidden = 10
dim_bottleneck = 2
encoder = ANN(n_total, dim_hidden, 2*dim_bottleneck)
decoder = ANN(dim_bottleneck, dim_hidden, n_total)






"""
    4. Set up entire VAE
"""


# i) VAE class

class VAE(pyro.nn.PyroModule):
    # Initialize VAE by invoking superclass and integrating basic data
    def __init__(self):
        super().__init__()
                
        # integrate ann's
        self.encoder = encoder
        self.decoder = decoder
        
        
    # ii) Define the model
    
    def model(self, input_images):
        # Register the parameters inside of the decoder module
        pyro.module("decoder", self.decoder)
        
        # Define prior for latent z
        mu_z_prior = torch.zeros([input_images.shape[0], dim_bottleneck])
        sigma_z_prior = torch.ones([input_images.shape[0], dim_bottleneck])
        dist_z_prior = pyro.distributions.Normal(loc = mu_z_prior, scale = sigma_z_prior).to_event(1)
        
        # Sample from prior and decode latent to image
        with pyro.plate('batch_plate', size = input_images.shape[0], dim = -1):
            z_sample = pyro.sample('latent_z', dist_z_prior)
            decoded_z = self.decoder(z_sample)
            
            dist_image = pyro.distributions.Normal(loc = decoded_z, scale = 1e-3).to_event(1)
            images = pyro.sample('image', dist_image, obs = input_images)
        
        return images
        
    
    # iii) Define the guide
    
    def guide(self, input_images):
        # Register the parameters inside of the encoder module
        pyro.module("encoder", self.encoder)
        with pyro.plate('batch_plate', size = input_images.shape[0], dim = -1):
            # Define latent distribution and sample
            latent_code = self.encoder(input_images)
            mu_z_post = latent_code[:,0:dim_bottleneck]
            # sigma_z_post = torch.exp(latent_code[:,dim_bottleneck:2*dim_bottleneck])
            sigma_z_post = 0.1*torch.ones(mu_z_post.shape)
            dist_z_post = pyro.distributions.Normal(loc = mu_z_post, scale = sigma_z_post).to_event(1)
            z_samples = pyro.sample('latent_z', dist_z_post)
        
        return z_samples
    
    
    # iv) Design function to resimulate image
    
    def resimulate_image(self, input_image, n_simus):
        # Draw n_simu independent realizations of encoding, decoding sequence
        
        # Define distribution of latents (encoding)
        mu_z_resimu, sigma_z_resimu = self.encoder(input_image)[:, 0:dim_bottleneck], \
            torch.exp(self.encoder(input_image)[:, dim_bottleneck:2*dim_bottleneck])
        
        # Sample latents and convert to image (decoding)
        images_resimu = torch.zeros([n_simus, n_total]) 
        for k in range(n_simus):      
            z_resimu = pyro.sample('z_reconstruct', pyro.distributions.Normal(loc = mu_z_resimu, scale = sigma_z_resimu))
            images_resimu[k, :] = self.decoder(z_resimu)
        
        return images_resimu.detach().reshape([n_simus, n_space,n_space])
     

# v) Instantiate VAE & sample untrained results

vae = VAE()

n_simus = 3
z_samples_untrained = copy.copy(vae.guide(data))
images_resimu_untrained_type1 = copy.copy(vae.resimulate_image(data[0,:], n_simus))
images_resimu_untrained_type2 = copy.copy(vae.resimulate_image(data[n_data,:], n_simus))
images_resimu_untrained_type3 = copy.copy(vae.resimulate_image(data[2*n_data,:], n_simus))


        


"""
    5. Perform inference
"""


# i) Set up inference


adam = pyro.optim.NAdam({"lr": 0.003})
elbo = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(vae.model, vae.guide, adam, elbo)


# ii) Perform svi

loss_sequence = []
for step in range(10000):
    loss = svi.step(data)
    loss_sequence.append(loss)
    if step % 100 == 0:
        print('epoch: {} ; loss : {}'.format(step, loss))
    else:
        pass


# iii) Sample trained model

z_samples_untrained = copy.copy(vae.guide(data))
images_resimu_trained_type1 = copy.copy(vae.resimulate_image(data[0,:], n_simus))
images_resimu_trained_type2 = copy.copy(vae.resimulate_image(data[n_data,:], n_simus))
images_resimu_trained_type3 = copy.copy(vae.resimulate_image(data[2*n_data,:], n_simus))



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
