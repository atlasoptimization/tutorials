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
performed on nultiple objects at multiple locations. The temperature of the
objects is random exhibiting spatial correlation and the measurements are
are affected by randomness, too. The goal is to compute distributional parameters 
and the posterior of the true temperature at some location given observations at 
other locations. We will do this by computing the true posterior and compare it 
to the results of pyros inference machinery.
For this, do the following:
    1. Imports and definitions
    2. Generate synthetic data
    3. True posterior
    4. Pyro model and inference
    5. Plots and illustrations
    
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
n_locs = 3
n_disc = 100

z_disc = torch.linspace(4,16,n_disc)
zz1, zz2 = torch.meshgrid((z_disc, z_disc), indexing = 'ij')


torch.manual_seed(0)
pyro.set_rng_seed(0)



"""
    2. Generate synthetic data
"""


# i) Define params

# mean of [10,8,12] degree celsius (deg c) at water, mountain, forest locations
# measurements are unbiased -> mu_epsion = 0 deg c
mu_z = 1.0 * torch.tensor([[10, 8, 12]])               
mu_epsilon = torch.zeros([1,3])

# autocorrelated true temperatures and i.i.d. noise
sigma_z = 1.0 * torch.tensor([[2,-1.5,2],[-1.5,4,-1.5],[2,-1.5,3]])
sigma_epsilon = 1 * torch.eye(n_locs)


# ii) Simulate data

# sample underlying true temperature in deg c, then sample subsequent measurement
dist_z = pyro.distributions.MultivariateNormal(mu_z, sigma_z).expand([n_data])
z_true = dist_z.sample()         
dist_data = pyro.distributions.MultivariateNormal(loc = z_true + mu_epsilon ,
                        covariance_matrix = sigma_epsilon)
data = dist_data.sample()


# iii) Observation for conditioning

# Measurement in water location is given, temperature at mountain and forest
# location is to be estimated
data_observation = torch.tensor([6])
index_observation = [0]
index_to_estimate = [1,2]



"""
    3. Analytic MLE & Posterior
"""


# i)  Compute conditional mean and covariance

# means for water location (w), mountain location (m), forest location (f)
mu_w = mu_z[0,0]
mu_mf = mu_z[0,1:3]

# covariances for locations (w,m,f) and measurements at location w (Tw)
sigma_T_T = sigma_z + sigma_epsilon
sigma_Tw_Tw = sigma_T_T[0,0]
sigma_mf_w = sigma_z[1:3,0]
sigma_mf_mf = sigma_z[1:3, 1:3]

# use equations for conditional mean and conditional covariance of Gaussian
conditional_mu = mu_mf + sigma_mf_w * (1/(sigma_Tw_Tw)) * (data_observation - mu_w) 
conditional_sigma = sigma_mf_mf -  sigma_mf_w.unsqueeze(1) * (1/(sigma_Tw_Tw)) *sigma_mf_w.unsqueeze(1).permute((1,0))


# ii) Construct posterior from mean and covariance

dist_conditional = pyro.distributions.MultivariateNormal(loc = conditional_mu,
                                        covariance_matrix = conditional_sigma)
dist_unconditional = pyro.distributions.MultivariateNormal(loc = mu_mf,
                                        covariance_matrix = sigma_mf_mf)


density_conditional = torch.zeros([n_disc,n_disc])
density_unconditional = torch.zeros([n_disc,n_disc])
for k in range(n_disc):
    for l in range(n_disc):
        temp_input = torch.tensor([[zz1[k,l]],[zz2[k,l]]])
        density_conditional[k,l] = torch.exp(dist_conditional.log_prob(temp_input.flatten()))
        density_unconditional[k,l] = torch.exp(dist_unconditional.log_prob(temp_input.flatten()))



"""
    4. Pyro model and inference
"""


# i) Pyro model

def model(observations = None):    
    # Define parameters
    n_obs = 1 if observations is None else observations.shape[0]
    mu = pyro.param('mu', init_tensor = torch.zeros([1,n_locs]))
    sigma = pyro.param('sigma', init_tensor = torch.eye(n_locs), 
                       constraint = pyro.distributions.constraints.positive_definite)
    sigma_noise = pyro.param('sigma_noise', init_tensor = torch.ones([1]),
                             constraint = pyro.distributions.constraints.positive)
    
    # Sample z, then sample measurement and subsample it
    with pyro.plate('batch_plate', size = n_obs, dim = -1):
        # Sample z
        dist_z = pyro.distributions.MultivariateNormal(loc = mu, 
                            covariance_matrix = sigma).expand([n_obs])
        z = pyro.sample('latent_z', dist_z)
        
        # Sample measurement
        dist_measurement = pyro.distributions.MultivariateNormal(loc =  z, 
                                covariance_matrix = sigma_noise * torch.eye(n_locs))
        measurement = pyro.sample('measurement', dist_measurement, obs = observations)
        
        # Subsample measurement for conditioning
        dist_submeasurement = pyro.distributions.Normal(loc = measurement[:,0]
                                            .reshape([n_obs,1]), scale = 0.01).to_event(1)
        measurement_water = pyro.sample('measurement_water', dist_submeasurement)
    return z, measurement, measurement_water


# ii) Pyro guide

def guide(observations = None):
    # Parameters
    coeffs_post = pyro.param('coeffs_post', torch.zeros([n_locs, n_locs]))
    trans_post = pyro.param('trans_post', torch.zeros([1, n_locs]))
    sigma_z_post = pyro.param('sigma_z_post', torch.eye(n_locs),
                              constraint = pyro.distributions.constraints.positive)
    sigma_water_post = pyro.param('sigma_water_post', torch.ones([1]),
                                  constraint = pyro.distributions.constraints.positive)
    
    # Variational distribution for latent z
    mu_latent = observations @ coeffs_post + trans_post
    sigma_latent = sigma_z_post
    dist_latent = pyro.distributions.MultivariateNormal(mu_latent, sigma_latent)
    
    with pyro.plate('batch_plate', size = observations.shape[0], dim = -1):    
        latent_z = pyro.sample('latent_z', dist_latent)
        
        # Variational distribution for measurement water
        mu_water = observations[:,0].reshape([observations.shape[0],1])
        sigma_water = sigma_water_post
        dist_water = pyro.distributions.Normal(mu_water, sigma_water).to_event(1)
        measurement_water = pyro.sample('measurement_water', dist_water)
    
    return latent_z, measurement_water


# iii) Pyro diagnostics

# Render model and guide
pyro.render_model(model, model_args=(), render_distributions=True, render_params=True)  
pyro.render_model(guide, model_args=(data,), render_distributions=True, render_params=True)

# Showcase the execution traces
model_trace_data = pyro.poutine.trace(model).get_trace(data)
model_trace_data.compute_log_prob()
print(model_trace_data.format_shapes())

guide_trace_data = pyro.poutine.trace(guide).get_trace(data)
guide_trace_data.compute_log_prob()
print(guide_trace_data.format_shapes())


# iv) Pyro inference

adam = pyro.optim.NAdam({"lr": 0.01})
elbo = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(model, guide, adam, elbo)

loss_sequence = []
for step in range(5000):
    loss = svi.step(data)
    if step % 100 == 0:
        print('epoch: {} ; loss : {}'.format(step, loss))
    else:
        pass
    loss_sequence.append(loss)


# v) Conditional model and guide and training
    
# conditioned model is the model but the measurement water site has fixed value
conditioned_model = pyro.condition(model, data = {'measurement_water' : data_observation})


# conditioned guide is model for the posterior distributions of the rv's unobserved
# in the conditioned model, i.e. latent_z and measurements
def conditioned_guide(observations = None):
    # Parameters
    conditional_mu_pyro = pyro.param('conditional_mu_pyro', torch.zeros([1,3]))
    conditional_sigma_pyro = pyro.param('conditional_sigma_pyro', torch.eye(3), 
                                constraint = pyro.distributions.constraints.positive)
    
    # Distribution and sampling
    dist_conditional_latent_z = pyro.distributions.MultivariateNormal(loc = conditional_mu_pyro, 
                                                    covariance_matrix = conditional_sigma_pyro)
    conditional_latent_z = pyro.sample('latent_z', dist_conditional_latent_z)
    
    dist_measurement = pyro.distributions.MultivariateNormal(loc =  conditional_latent_z, 
                            covariance_matrix = pyro.param('sigma_noise') * torch.eye(n_locs))
    measurement = pyro.sample('measurement', dist_measurement, obs = observations)
    
    return conditional_latent_z

pyro.render_model(conditioned_model, model_args=(), render_distributions=True, render_params=True)  
pyro.render_model(conditioned_guide, model_args=(), render_distributions=True, render_params=True)  

# Define callback that gives nonzero updates only for conditional distribution params
def selective_optimization(param_name):
    if "conditional" in param_name:
        return {"lr": 0.01}
    else:
        return {"lr": 0}

# Execute upates on conditioned model and guide
adam = pyro.optim.NAdam(selective_optimization)
svi = pyro.infer.SVI(conditioned_model, conditioned_guide, adam, elbo)
loss_sequence_conditional = []
for step in range(5000):
    loss_conditional = svi.step(None)
    if step % 100 == 0:
        print('epoch: {} ; loss : {}'.format(step, loss_conditional))
    loss_sequence_conditional.append(loss_conditional)



"""
    5. Plots and illustrations
"""


# i) Parameter comparison

sigma_trained = pyro.get_param_store()['sigma'].detach()
sigma_noise_trained = torch.eye(n_locs)*pyro.get_param_store()['sigma_noise'].detach()

fig, ax = plt.subplots(2,2, figsize = (10,10), dpi = 300)

ax[0,0].imshow(sigma_z, vmin = 0, vmax = 5)
ax[0,0].set_title('True covariance for temperature')

ax[1,0].imshow(sigma_T_T, vmin = 0, vmax = 5)
ax[1,0].set_title('True covariance for measurements')

ax[0,1].imshow(sigma_trained, vmin = 0, vmax = 5)
ax[0,1].set_title('Inferred covariance for temperature')

ax[1,1].imshow(sigma_trained + sigma_noise_trained, vmin = 0, vmax = 5)
ax[1,1].set_title('Inferred covariance for measurements')

plt.tight_layout()
plt.show()


# ii) Density comparison

# Compute conditional and unconditional densities pyro

conditional_mu_pyro = pyro.param('conditional_mu_pyro')
conditional_sigma_pyro = pyro.param('conditional_sigma_pyro')
unconditional_mu_pyro = pyro.param('mu')
unconditional_sigma_pyro = pyro.param('sigma')

dist_conditional_pyro = pyro.distributions.MultivariateNormal(loc = conditional_mu_pyro[0,1:3],
                                        covariance_matrix = conditional_sigma_pyro[1:3,1:3])
dist_unconditional_pyro = pyro.distributions.MultivariateNormal(loc = unconditional_mu_pyro[0,1:3],
                                        covariance_matrix = unconditional_sigma_pyro[1:3,1:3])

density_conditional_pyro = torch.zeros([n_disc,n_disc])
density_unconditional_pyro = torch.zeros([n_disc,n_disc])
for k in range(n_disc):
    for l in range(n_disc):
        temp_input = torch.tensor([[zz1[k,l]],[zz2[k,l]]])
        density_conditional_pyro[k,l] = torch.exp(dist_conditional_pyro.log_prob(temp_input.flatten()))
        density_unconditional_pyro[k,l] = torch.exp(dist_unconditional_pyro.log_prob(temp_input.flatten()))

# Plot densities
fig, ax = plt.subplots(2,2, figsize = (10,10), dpi = 300)

ax[0,0].imshow(density_unconditional, extent = [z_disc[0], z_disc[-1], z_disc[-1], z_disc[0]])
ax[0,0].set_title('True underlying density for m, f')
ax[0,0].set_xlabel('T forest location')
ax[0,0].set_ylabel('T mountain location')


ax[1,0].imshow(density_conditional, extent = [z_disc[0], z_disc[-1], z_disc[-1], z_disc[0]])
ax[1,0].set_title('True conditional density for observed w = 8')
ax[1,0].set_xlabel('T forest location')
ax[1,0].set_ylabel('T mountain location')

ax[0,1].imshow(density_unconditional_pyro.detach(), extent = [z_disc[0], z_disc[-1], z_disc[-1], z_disc[0]])
ax[0,1].set_title('Estimated density for m, f')
ax[0,1].set_xlabel('T forest location')
ax[0,1].set_ylabel('T mountain location')


ax[1,1].imshow(density_conditional_pyro.detach(), extent = [z_disc[0], z_disc[-1], z_disc[-1], z_disc[0]])
ax[1,1].set_title('Estimated conditional density for observed w = 8')
ax[1,1].set_xlabel('T forest location')
ax[1,1].set_ylabel('T mountain location')

plt.tight_layout()
plt.show()


# iii) Training process

fig, ax = plt.subplots(2,1, figsize = (15,5))
ax[0].plot(loss_sequence)
ax[0].set_title('Loss during computation model fit')

ax[1].plot(loss_sequence_conditional)
ax[1].set_title('Loss during computation conditional')

plt.tight_layout()
plt.show()


