#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is a compact version of pyro_cc_4_model_2 thatshowcases pyro inference
yielding correctly inferred posterior standard deviations if the right model (a
a multivariate Normal) is used.
    
The script is meant solely for educational and illustrative purposes. Written by
Dr. Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.com.
"""



"""
    1. Imports and definitions
"""


# i) Imports

import pyro
import torch
import pandas
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)


# ii) Definitions

measurement_df = pandas.read_csv("sensor_measurements.csv")
n_device = measurement_df["sensor_id"].nunique()
n_measure = measurement_df["time_step"].nunique()

T_true = torch.tensor(measurement_df["T_true"].values).reshape(n_device, n_measure)
T_meas = torch.tensor(measurement_df["T_measured"].values).reshape(n_device, n_measure)
sigma_T_meas = 0.3

# Fix random seed
pyro.set_rng_seed(0)



"""
    2. Build model and guide
"""


# i) Define the model

# Define priors
mu_alpha = torch.tensor([0.0, 1.0])
Sigma_alpha = torch.tensor([[0.1**2,0], [0, 0.1**2]])

# Build the model in pyro
def model(input_vars = T_true, observations = None):
    alpha_prior_dist = pyro.distributions.MultivariateNormal(loc = mu_alpha,
                                            covariance_matrix = Sigma_alpha)
    alpha = pyro.sample('alpha', alpha_prior_dist)
    
    obs_dist = pyro.distributions.Normal(loc = alpha[0] + alpha[1] * T_true,
                                         scale = sigma_T_meas)
    
    with pyro.plate('device_plate', dim = -2):
        with pyro.plate('measure_plate', dim = -1):
            obs = pyro.sample('observations', obs_dist, obs = observations)
    
    return obs
                      

# ii) Build the guide


# Build the guide
def guide(input_vars = T_true, observations = None):
    alpha_mu = pyro.param('alpha_mu', init_tensor = torch.ones([2]))
    alpha_sigma = pyro.param('alpha_sigma', init_tensor = 0.1*torch.eye(2),
                             constraint = pyro.distributions.constraints.positive_definite)
    alpha_dist_post = pyro.distributions.MultivariateNormal(loc = alpha_mu, 
                                            covariance_matrix = alpha_sigma)
    alpha = pyro.sample('alpha', alpha_dist_post)

    return alpha


# iii) Illustrate model and guide

graphical_model = pyro.render_model(model = model, model_args= (T_true,),
                                    render_distributions=True,
                                    render_params=True)
graphical_guide = pyro.render_model(model = guide, model_args= (T_true,),
                                    render_distributions=True,
                                    render_params=True)

graphical_model
graphical_guide



n_model_samples = 30
n_guide_samples = 1000  

predictive = pyro.infer.Predictive
prior_predictive_pretrain = predictive(model, num_samples = n_model_samples)()['observations']
posterior_pretrain_dict = predictive(guide, num_samples = n_guide_samples)()
posterior_pretrain = posterior_pretrain_dict['alpha']
posterior_predictive_pretrain = predictive(model, guide = guide, num_samples = n_model_samples)()['observations']

    

"""
    3. Perform inference (SVI version)
"""


# # i) Set up inference

# adam = pyro.optim.Adam({"lr": 0.01})
# elbo = pyro.infer.Trace_ELBO(num_particles = 5)
# svi = pyro.infer.SVI(model, guide, adam, elbo)


# # ii) Perform svi

# data = (T_true, T_meas)
# loss_sequence = []
# for step in range(5000):
#     loss = svi.step(*data)
#     loss_sequence.append(loss)
#     if step %50 == 0:
#         print(f'epoch: {step} ; loss : {loss}')
                

"""
    3. Perform inference (MCMC version)
"""

import pyro.infer.mcmc as mcmc

# i) Set up NUTS kernel and MCMC sampler
n_mcmc = 1000
nuts_kernel = mcmc.NUTS(model)
mcmc_sampler = mcmc.MCMC(nuts_kernel, 
                         num_samples=n_mcmc, 
                         warmup_steps=500, 
                         num_chains=1)  # Increase chains for robustness

# ii) Run MCMC
mcmc_sampler.run(T_true, T_meas)

# iii) Extract samples
posterior_samples = mcmc_sampler.get_samples()
posterior_alpha = posterior_samples["alpha"]

# iv) Compute posterior means and stds
alpha_mean = posterior_alpha.mean(dim=0)
alpha_mu = alpha_mean.reshape([2,1])
alpha_sigma = (1/n_mcmc)*posterior_alpha.T@posterior_alpha - alpha_mu@alpha_mu.T

print("alpha_mu_post = \n{} \n alpha_Sigma_post = \n{}".format(alpha_mean, alpha_sigma))



# iii) Record example outputs of model and guide after training

prior_predictive_posttrain = predictive(model, num_samples = n_model_samples)()['observations']
posterior_posttrain_dict = predictive(guide, num_samples = n_guide_samples)()
posterior_posttrain = posterior_posttrain_dict['alpha']
posterior_predictive_posttrain = predictive(model, guide = guide, num_samples = n_model_samples)()['observations']


# iv) Additional investigations

# We now inspect the model and guide using pyros inbuilt functionality. This allows
# us insight into interior details of model and guide; but we only plot the shapes.
model_trace = pyro.poutine.trace(model).get_trace(T_true, observations = T_meas)
guide_trace = pyro.poutine.trace(guide).get_trace(T_true, observations = T_meas)
print('These are the shapes inside of the model : \n {}'.format(model_trace.format_shapes()))
print('These are the shapes inside of the guide : \n {}'.format(guide_trace.format_shapes()))

# The parameters of the posterior are again stored in pyro's param_store
for name, value in pyro.get_param_store().items():
    print('Param : {}; Value : {}'.format(name, value))
   

A = torch.vstack((torch.ones([n_device*n_measure]), T_true.flatten())).T
Sigma_noise = (sigma_T_meas**2)*torch.eye(n_device*n_measure)
Sigma_y = A@Sigma_alpha@A.T + Sigma_noise

mu_alpha_cf = mu_alpha + Sigma_alpha @ A.T@torch.linalg.pinv((A@Sigma_alpha@A.T
                        +  Sigma_noise)) @(T_meas.flatten() - T_true.flatten())
Sigma_alpha_cf = Sigma_alpha - Sigma_alpha @ A.T@torch.linalg.pinv((A@Sigma_alpha@A.T
                        + Sigma_noise)) @A@Sigma_alpha

print('The closed form solution is \n alpha_mu_post = \n{} \n alpha_Sigma_post = \n{}'.format(mu_alpha_cf, Sigma_alpha_cf))
