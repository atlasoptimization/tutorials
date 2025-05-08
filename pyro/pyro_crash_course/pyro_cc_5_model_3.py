#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is part of a series of notebooks that showcase the pyro probabilistic
programming language - basically pytorch + probability. In this series we explore
a sequence of increasingly complicated models meant to represent the behavior of
some measurement device. 
The crash course consists in the following:
    - minimal pyro example                      (cc_0_minimal_inference)
    - generating data and exploring it          (cc_1_hello_dataset)
    - Model with no trainable params            (cc_2_model_0)
    - Model with deterministic parameters       (cc_2_model_1)
    - Model with latent variables               (cc_2_model_2)
    - Model with hierarchical randomness        (cc_2_model_3)
    - Model with discrete random variables      (cc_2_model_4)
    - Model with neural net                     (cc_2_model_5)
    

This script will expand on the previous model_2 and feature latent variables
offset and scale that are different for each measurement device. This model_3
is therefore more finegrained than model_2 and features more latent variables
(2 per device) but is not fundamentally different in terms of theoretical 
background. To allow more flexibility, we will make the variational family be
a multivariate gaussian (thereby allowing for correlations in the posterior).
This also simplifies the guide, since we only need one distribution per device.
We will infer n_device latent distributions and therefore n_device x 2 parameters;
the for each device a mean and a covariance matrix representing the posterior.
This model will allow us to treat the different thermistors differently and will
do so in an automatic manner.

# Our model assumes the thermistors generate measurements T_meas which are made
made up from scaled T_true and some offset (and noise).  We will assume that the
scale and offset are unknown random variables. They are different for each sensor
in contrast to model_1 where only one scale and offset is shared among all the 
devices. The motivation for this model is the same as for model_2, i.e. the scale
and offset being known from some spec sheet up to some uncertainty while we allow
the additional flexibility of the devices being manufactured with slight differences.
Since we will assume the standard deviation of the noise to be fixed, the model
has again no trainable parameters.

The guide will again be nontrivial. We will prescribe as variational family to
be adjusted towards the posterior a multivariate Gaussian with unknown mean and
unknown covariance matrix (these are the trainable parameters). We will after 
training be able to sample from the posterior distribution to get a guess of
scale and offset per device and also to sample from the posterior predictive
distribution, which should yield now data that fits the measurement data better
since trained posterior scales and offsets are different for each device.

In the end, this model_3 showcases the basics of building a (model, guide) pair
featuring a hierarhcy of latent random variables and how the associated posterior
distributions are inferred by pyro's inference machinery.

For this, do the following:
    1. Imports and definitions
    2. Build model and guide
    3. Perform inference
    4. Interpretations and illustrations
    
In this script we keep explanations of previously introduced building blocks short;
for explanations on pyro.sample, pyro.plate, pyro.param, pyro.infer.Predictive,
other basic commands, and general guide design, see the previous scripts.
    
The script is meant solely for educational and illustrative purposes. Written by
Dr. Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.com.
"""



"""
    1. Imports and definitions
"""


# i) Imports
# If you run this in Colab, you might get an import error as pyro is not
# installed by default. In that case, uncomment the following command.
# !pip install pyro-ppl

import pyro
import torch
import pandas
import matplotlib.pyplot as plt


# ii) Definitions

# Read csv, infer dimensions
# !wget https://raw.githubusercontent.com/atlasoptimization/tutorials/master/pyro/pyro_crash_course/sensor_measurements.csv
measurement_df = pandas.read_csv("sensor_measurements.csv")
n_device = measurement_df["sensor_id"].nunique()
n_measure = measurement_df["time_step"].nunique()

# Read out T_true and T_meas: the true temperature and the measured temperature
T_true = torch.tensor(measurement_df["T_true"].values).reshape(n_device, n_measure)
T_meas = torch.tensor(measurement_df["T_measured"].values).reshape(n_device, n_measure)

# Assume standard deviation
sigma_T_meas = 0.3

# Fix random seed
pyro.set_rng_seed(0)



"""
    2. Build model and guide
"""


# i) Define the model
# We implement a simple model assuming that our measurements T_meas are scaled,
# biased, noisy versions of T_true where scale and offset are random/ known only
# with some uncertainty.  But now scale and offset are latent variables different
# for each measurement device.
# We might write this as: T_meas = a_0 + a_1*T_true + noise.
# Or similary as the probabilistic model
# a ~ N([0,1], 0.1 * I)                     a shape [n_device,2]
# mu_T_meas = a[:,0] + a[:,1] * T_true      mu_T_meas shape [n_device, n_meas]
# T_meas ~ N(mu_T_meas, sigma_T_meas)       T_meas shape [n_device, n_meas]

# Define priors
mu_alpha = torch.tensor([0.0, 1.0])
Sigma_alpha = torch.tensor([[0.1**2,0], [0, 0.1**2]])


# Build the model in pyro
def model(input_vars = T_true, observations = None):
    # We sample again in the two different independence contexts 'device_plate'
    # and 'measure_plate'. But now sampling the different latents alpha happens
    # already in the device plate (as they are different for each device).

    alpha_dist = pyro.distributions.MultivariateNormal(loc = mu_alpha,
                                                covariance_matrix = Sigma_alpha)
    
    # Independence in first dim of alpha_dist, sample n_device times
    with pyro.plate('device_plate', size = n_device, dim = -1):
        # We sample alpha from a multivariate normal prior
        alpha = pyro.sample('alpha', alpha_dist)
    
    # Build the observation distribution
    # This requires a bit of reshaping to achieve broadcastable shapes.
    mean_obs = (alpha[:,0].unsqueeze(1)) + (alpha[:,1].unsqueeze(1))*T_true
    obs_dist = pyro.distributions.Normal(loc = mean_obs, scale = sigma_T_meas)
    
    # Sample from this distribution and declare the samples independent in the
    # first two dims. 
    with pyro.plate('device_plate', dim = -2):
        with pyro.plate('measure_plate', dim = -1):
            obs = pyro.sample('observations', obs_dist, obs = observations)
    
    return obs
                      

# ii) Build the guide
# The guide ( = variational distribution) now needs to encode several posterior
# distributions, one multivariate Gaussian for each device.

# Build the guide
def guide(input_vars = T_true, observations = None):
    # Per-device means and scales (shape [n_device, 2])
    mu_alpha_post = pyro.param('mu_alpha_post', torch.tensor([0,1.0]).unsqueeze(0).expand([n_device,2]))
    sigma_alpha_post = (pyro.param('sigma_alpha_post', 0.1 * (torch.eye(2).unsqueeze(0)).expand([n_device,2,2]),
                             constraint=pyro.distributions.constraints.positive_definite) 
                        + 0.001 * torch.eye(2))
    # We add 1e-3 on the diagonal of the covariance matrix to avoid numerical issues
    # related to positive definiteness tests.

    with pyro.plate('device_plate', size=n_device, dim=-1):
        # Multivariate Gaussian for each device (allow for correlations)
        alpha = pyro.sample('alpha', pyro.distributions.MultivariateNormal(loc = mu_alpha_post,
                                        covariance_matrix = sigma_alpha_post))

    return alpha


# iii) Illustrate model and guide

# Now the illustration of the model shows alpha inside of the device plate as
# a multivariate normal random variable. In the guide we now have also alpha
# as a member of the device plate and affected by the posterior mean and covariance.
graphical_model = pyro.render_model(model = model, model_args= (T_true,),
                                    render_distributions=True,
                                    render_params=True)
graphical_guide = pyro.render_model(model = guide, model_args= (T_true,),
                                    render_distributions=True,
                                    render_params=True)

graphical_model
graphical_guide


# iv) Record example outputs of model and guide prior to training

n_model_samples = 10
n_guide_samples = 1000  

predictive = pyro.infer.Predictive
prior_predictive_pretrain = predictive(model, num_samples = n_model_samples)()['observations']
posterior_pretrain = predictive(guide, num_samples = n_guide_samples)()['alpha']
posterior_predictive_pretrain = predictive(model, guide = guide, num_samples = n_model_samples)()['observations']



"""
    3. Perform inference
"""


# i) Set up inference

adam = pyro.optim.Adam({"lr": 0.01})
elbo = pyro.infer.Trace_ELBO(num_particles = 20)
svi = pyro.infer.SVI(model, guide, adam, elbo)


# ii) Perform svi

data = (T_true, T_meas)
loss_sequence = []
for step in range(1000):
    loss = svi.step(*data)
    loss_sequence.append(loss)
    if step %50 == 0:
        print(f'epoch: {step} ; loss : {loss}')
                
        
# iii) Record example outputs of model and guide after training

prior_predictive_posttrain = predictive(model, num_samples = n_model_samples)()['observations']
posterior_posttrain = predictive(guide, num_samples = n_guide_samples)()['alpha']
posterior_predictive_posttrain = predictive(model, guide = guide, num_samples = n_model_samples)()['observations']


# iv) Additional investigations

# The shapes inside of the model and guide are now slightly more complex. We can
# now see that alpha has a batch shape of 5 (independent samples from the alpha
# distribution) and an event shape of 2 (since its multivariate and each draw
# delivers alpha = (offset, scale). Notice that the observation dims have not 
# changed and still have batch_shape [5,100]. The guide parameter sigma_alpha_post
# (The posterior covariance matrices per device) has shape 5 x 2 x 2 where 5 = n_device.
model_trace = pyro.poutine.trace(model).get_trace(T_true, observations = T_meas)
guide_trace = pyro.poutine.trace(guide).get_trace(T_true, observations = T_meas)
print('These are the shapes inside of the model : \n {}'.format(model_trace.format_shapes()))
print('These are the shapes inside of the guide : \n {}'.format(guide_trace.format_shapes()))

# The parameters of the posterior are again stored in pyro's param_store
for name, value in pyro.get_param_store().items():
    print('Param : {}; Value : {}'.format(name, value))


"""
    4. Interpretations and illustrations
"""

# i) Plot and print ELBO loss

# The ELBO loss can now only be sampled, so it is noisy.
plt.figure(1, dpi = 300)
plt.plot(loss_sequence)
plt.yscale("log")
plt.title('ELBO loss during training (log scale)')
plt.xlabel('Epoch nr')
plt.ylabel('value')


# The ELBO loss decreases on average but from epoch to epoch it may jump quite
# significantly. A better, less noisy estimation of the ELBO can be done by 
# increasing the num_particles - that is the number of latent samples used for
# ELBO estimation.

# ii) Compare model output and data

# Create the figure and 2x5 subplot grid
fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=False, sharey=False)
# Global y-axis limits
y_min = T_meas.min()
y_max = T_meas.max()

# FIRST ROW
# First plot: measurement data
for i in range(n_device):
    axes[0,0].scatter(T_true[i,:], T_meas[i,:], s=10, label=f"T_meas: sensor[{i}]")
axes[0,0].set_title("Measurement data")
axes[0,0].set_xlabel("T_true")
axes[0,0].set_ylabel("T_meas")
axes[0,0].set_ylim(y_min, y_max)
axes[0,0].legend()

# Second plot: data produced by model pre-training
for k in range(n_model_samples):
    for i in range(n_device):
        axes[0,1].scatter(T_true[i,:], prior_predictive_pretrain[k,i,:], s=10, label=f"T_meas: sensor[{i}]")
axes[0,1].set_title("Data from model pre-training")
axes[0,1].set_xlabel("T_true")
axes[0,1].set_ylabel("T_meas")
axes[0,1].set_ylim(y_min, y_max)

# Third plot: data produced by posterior_predictive pre-training
for k in range(n_model_samples):
    for i in range(n_device):
        axes[0,2].scatter(T_true[i,:], posterior_predictive_pretrain[k,i,:], s=10, label=f"T_meas: sensor[{i}]")
axes[0,2].set_title("Data from posterior predictive pre-training")
axes[0,2].set_xlabel("T_true")
axes[0,2].set_ylabel("T_meas")
axes[0,2].set_ylim(y_min, y_max)

# Fourth plot: data produced by guide pre-training
axes[0,3].hist2d(posterior_pretrain[:,:,0].flatten().numpy(),
                 posterior_pretrain[:,:,1].flatten().numpy(),
                 bins=10, cmap='viridis')
axes[0,3].set_title("2D Histogram of parameters pre-train")
axes[0,3].set_xlabel("alpha_0")
axes[0,3].set_ylabel("alpha_1")

# SECOND ROW
# Second plot: data produced by model post-training
for k in range(n_model_samples):
    for i in range(n_device):
        axes[1,1].scatter(T_true[i,:], prior_predictive_posttrain[k,i,:], s=10, label=f"T_meas: sensor[{i}]")
axes[1,1].set_title("Data from model post-training")
axes[1,1].set_xlabel("T_true")
axes[1,1].set_ylabel("T_meas")
axes[1,1].set_ylim(y_min, y_max)

# Third plot: data produced by posterior_predictive post-training
for k in range(n_model_samples):
    for i in range(n_device):
        axes[1,2].scatter(T_true[i,:], posterior_predictive_posttrain[k,i,:], s=10, label=f"T_meas: sensor[{i}]")
axes[1,2].set_title("Data from posterior predictive post-training")
axes[1,2].set_xlabel("T_true")
axes[1,2].set_ylabel("T_meas")
axes[1,2].set_ylim(y_min, y_max)

# Fourth plot: data produced by guide post-training
axes[1,3].hist2d(posterior_posttrain[:,:,0].flatten().numpy(),
                 posterior_posttrain[:,:,1].flatten().numpy(),
                 bins=10, cmap='viridis')
axes[1,3].set_title("2D Histogram of parameters post-train")
axes[1,3].set_xlabel("alpha_0")
axes[1,3].set_ylabel("alpha_1")

plt.tight_layout()
plt.show()


# The results are already quite interesting. We see that the posterior distribution
# has been trained to better match the overall spread found in the data. A detailed
# view of the histogram of the latent variables offset, scale also reveals that
# the distribution of latents is multimodal - one of the measurement devices behaves
# quite differently from the others. It has a similar scale but an offset that 
# differs by at least one degree C from all the other sensors. This might be a
# faulty sensor, that we could categorize as an outliner - and in fact we will 
# look into this in the next model that tries its hand at introducing discrete
# random variables for outlier detection.



    