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
    

This script will expand on the previous model_3 and feature the hierarchical
approach but also introduce a discrete latent variable to classify if a measurement
device is faulty. This will add another powerful tool to our arsenal, since 
discrete random variables are incredibly flexible for modelling class membership,
hidden switches for qualitatively different model behavior and even logical
constructions. Just as with Integer optimization though, the cost is a nondifferentiable
objective and some type of enumeration. The can increase by a lot the computational
load. The benefit for our model is not as big as the step from a single global 
alpha to a per-device alpha (model_2 -> model_3) but the insight into the hidden
state of the measurement device is quite nice.  

Our model assumes the thermistors generate measurements T_meas which are made
made up from scaled T_true and some offset (and noise) with scale and offset being
known from a spec sheet up to some uncertainty. Scale and offset are different
for each device. In contrast to model_3, we introduce a new discrete variable 
is_faulty that indicates if the device behaves atypically. The generative process
producing devices that then produce data is assumed as:
    1. A random decision if the device is built faulty
    2. Depending on is_faulty, sample alpha from different distribtutions
    3. Use alpha to construct the mean, then add noise to generate observations
Since we will assume the standard deviation of the noise to be fixed, the model
has again no trainable parameters. 
Having the outlier/faulty property as explicit part in our model also means that
we could learn outlier properties: not only what features make an outlier an outlier
but also how outlier variance behaves etc. This would be more difficult in a manual
computation scheme where we filter out outliers or other subpopulations of the
measurements and then would need to perform inference on these as well.


The guide will be very similar as in the previous scripts. 

In the end, this model_4 showcases the basics of building a (model, guide) pair
featuring a latent discrete random variable and how the associated posterior
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

# We need an additional decoration config_enumerate to declare inference over
# a discrete latent variable - then the appropriate sample sites are automatically
# exxhaustively enumerated.
import pyro
import torch
import pandas
import matplotlib.pyplot as plt
from pyro.infer import config_enumerate


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
# We implement the same model as in the previous notebook but add an additional
# generative step in the beginning, where we sample from the device production
# distribution to determine if the device is being produced faulty.
# This may be written as the probabilistic model:
# is_faulty ~ Bernoulli(p_faulty)           coin flip to decide production fault
# Sigma = Sigma_normal or Sigma_faulty      dependent on is_faulty
# a_dist = N([0,1], Sigma)
# a ~ N([0,1], 0.1 * I)                     a shape [n_device,2]
# mu_T_meas = a[:,0] + a[:,1] * T_true      mu_T_meas shape [n_device, n_meas]
# T_meas ~ N(mu_T_meas, sigma_T_meas)       T_meas shape [n_device, n_meas]
# This model is actually quite interesting and may be a bit hard to grasp initially.
# We have two different stochastic models essentially: One for normal devices and
# another, way more uncertain one for faulty devices. The latter one is only to
# be used sparsely though (encoded by low p_faulty). Otherwise, the inference would
# just declare every device faulty since this would lead to higher probabilities
# since the faulty model has higher variance = more likelihood for residuals.
# Overall, there is a possibility to interpret this as a thresholding test where
# the threshold comes from p_faulty and sigma_alpha, sigma_alpha_faulty.


# Define priors and fixed params
mu_alpha = torch.tensor([0.0, 1.0]).expand(n_device, -1)
Sigma_alpha = torch.tensor([[0.1**2,0], [0, 0.1**2]])
p_faulty = 0.05
Sigma_faulty = 100*Sigma_alpha

# Build the model in pyro; use decorator to mark discrete variables for enumeration.
@config_enumerate
def model(input_vars = T_true, observations = None):
    # Build reusable independence context device_plate.
    device_plate = pyro.plate('device_plate', size=n_device, dim=-1)

    with device_plate:
        is_faulty = pyro.sample('is_faulty', pyro.distributions.Bernoulli(p_faulty))

    # Convert boolean indicator to tensor of shape [n_device, 1, 1] for broadcasting
    is_faulty_tensor = is_faulty.unsqueeze(-1).unsqueeze(-1)

    # Build Sigma_device via interpolation
    Sigma_device = Sigma_alpha + is_faulty_tensor * (Sigma_faulty - Sigma_alpha)  # shape: [n_device, 2, 2]

    # Make alpha_dist have a sigma dependent on  device and is_faulty
    alpha_dist = pyro.distributions.MultivariateNormal(
        loc=mu_alpha,                   # shape: [2, n_device, 2]
        covariance_matrix=Sigma_device)     # shape: [2, n_device, 2, 2]
            
    # Independence in first dim of alpha_dist, sample n_device times
    with device_plate:
        alpha = pyro.sample("alpha", alpha_dist)
    mean_obs = alpha[:, 0].unsqueeze(1) + alpha[:, 1].unsqueeze(1) * T_true
    obs_dist = pyro.distributions.Normal(loc=mean_obs, scale=sigma_T_meas)

    # Sample from this distribution and declare the samples independent in the
    with pyro.plate('device_plate', dim=-2):
        with pyro.plate('measure_plate', dim=-1):
            pyro.sample("observations", obs_dist, obs=observations)

# ii) Build the guide
# Since the guide needs to encode the posterior distribution of the is_faulty
# variable now as well, we build a distribution with trainable parameters. The
# probs_faulty params encode how much we believe the devices to be faulty (0.05)
# for all devices equally prior to learning. Sampling from Bernoulli(probs_faulty)
# delivers a sample on the faultyness and probs_faulty are adjusted during training
# such that that sample harmonizes with the associated probabilistic model.

# Build the guide
@config_enumerate
def guide(input_vars = T_true, observations = None):
    # Build reusable independence context device_plate.
    device_plate = pyro.plate('device_plate', size=n_device, dim=-1)
    
    # Learn a per-device probability of being faulty
    probs_faulty = pyro.param("probs_faulty", 0.05 * torch.ones(n_device),
                              constraint=pyro.distributions.constraints.unit_interval)
    
    # Sample discrete variable from learned probabilities
    with device_plate:
        is_faulty = pyro.sample("is_faulty", pyro.distributions.Bernoulli(probs_faulty))
    
    # Per-device means and scales (shape [n_device, 2])
    mu_alpha_post = pyro.param('mu_alpha_post', torch.tensor([0,1.0]).unsqueeze(0).expand([n_device,2]))
    sigma_alpha_post = (pyro.param('sigma_alpha_post', 0.1 * (torch.eye(2).unsqueeze(0)).expand([n_device,2,2]),
                             constraint=pyro.distributions.constraints.positive_definite) 
                        + 0.001 * torch.eye(2))
    # We add 1e-3 on the diagonal of the covariance matrix to avoid numerical issues
    # related to positive definiteness tests.

    with device_plate:
        # Multivariate Gaussian for each device (allow for correlations)
        alpha = pyro.sample('alpha', pyro.distributions.MultivariateNormal(loc = mu_alpha_post,
                                        covariance_matrix = sigma_alpha_post))

    return alpha, is_faulty

# Both model and guide are written in such a way that dimensions can be prepended
# easily. This is necessary as during inference, all possible values of the
# discrete variable are enumerated and that enumeration happens in a separate 
# dimension on the left. This can make writing (model, guide) pairs suitable for
# discrete inference tricky.


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
posterior_pretrain_faulty = predictive(guide, num_samples = n_guide_samples)()['is_faulty']
posterior_predictive_pretrain = predictive(model, guide = guide, num_samples = n_model_samples)()['observations']



"""
    3. Perform inference
"""


# i) Set up inference

# In terms of loss function, we need now TraceEnum_ELBO instead of Trace_ELBO,
# as the latter one is not designed to handle enumeration.
adam = pyro.optim.Adam({"lr": 0.01})
elbo = pyro.infer.TraceEnum_ELBO(num_particles = 20,
                                 max_plate_nesting = 2)
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
posterior_posttrain_faulty = predictive(guide, num_samples = n_guide_samples)()['is_faulty']
posterior_predictive_posttrain = predictive(model, guide = guide, num_samples = n_model_samples)()['observations']


# iv) Additional investigations

# The model traces do not show anything specific. Apart from random variable 
# is_faulty and probabilities probs_faulty, everything is as in model_3. The
# complicated part of the inference is hidden and perfomed by prepending an 
# enumeration dimension in front of some samples and enumerating all possible 
# results.
model_trace = pyro.poutine.trace(model).get_trace(T_true, observations = T_meas)
guide_trace = pyro.poutine.trace(guide).get_trace(T_true, observations = T_meas)
print('These are the shapes inside of the model : \n {}'.format(model_trace.format_shapes()))
print('These are the shapes inside of the guide : \n {}'.format(guide_trace.format_shapes()))

# The parameters of the posterior are again stored in pyro's param_store
for name, value in pyro.get_param_store().items():
    print('Param : {}; Value : {}'.format(name, value))

# Estimate the posterior probabilities of each device being faulty. For this,
# take the samples of is_faulty from the guide function and average them to see
# the probability of is_faulty being assigned to each device.
faulty_pretrain = torch.mean(posterior_pretrain_faulty,dim=0)
faulty_posttrain = torch.mean(posterior_posttrain_faulty,dim=0)
print('is_faulty ground truth : [1, 0, 0, 0, 0]')
print('Probabilities of being faulty pre-training : {} \n'
      'Probabilities of being faulty post-training : {} '
      .format(faulty_pretrain, faulty_posttrain))


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


# The results are interesting. The faulty device is recognized quite easily as
# device nr 1 (as we declared in the simulation stage of the tutorial, pyro_cc-1).




    