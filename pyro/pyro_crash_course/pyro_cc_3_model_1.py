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
    

This script will build a first simple model with trainable parameters to fit
the dataset previously introduced, thereby expanding upon the previous model
that had not trainable parameters. This model_1 is a simple model that assumes
the thermistors to scale and offset true temperatures T_true and subsequently add 
some noise to produce the measured temperatures T_meas. Since we will assume the
standard deviation of the noise to be fixed, the model has as trainable parameters
the scale and the offset; considered unknown deterministic values identical for 
all thermistors. Therefore inference produces best guesses for scale and offset
where best is measured by the elbo and boils down to maximum likelihood estimation.
In the end, this model_1 showcases the basics of building a (model, guide) pair
featuring some unknown parameters  and how these are inferred by pyro's inference
machinery.

For this, do the following:
    1. Imports and definitions
    2. Build model and guide
    3. Perform inference
    4. Interpretations and illustrations
    
In this script we keep explanations of previously introduced building blocks short;
for explanations on pyro.sample, pyro.plate, and other basic commands, see the
previous scripts.
    
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
import copy
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
# biased, noisy versions of T_true. 
# We might write this as: T_meas = a_0 + a_1*T_true + noise.
# Or similarly as the probabilistic model 
# mu_T_meas = a_0 + a_1*T_true
# T_meas ~ N(mu_T_meas, sigma_T_meas)

# Build the model in pyro
def model(input_vars = T_true, observations = None):
    # Call pyro.param to declare parameters alpha_0 and alpha_1, which are to
    # pyro deterministic unknown tensors marked by pyro for optimization during
    # later inference. Inputs to pyro.param are a unique name and a init value.
    alpha_0 = pyro.param('alpha_0', torch.zeros([]))
    alpha_1 = pyro.param('alpha_1', torch.ones([]))
    
    # Build the observation distribution using the parameters. Note that the 
    # alpha's are torch.tensors and broadcast over T_true. The distribution
    # obs_dist produces therefore values of shape [n_device, n_measure].
    obs_dist = pyro.distributions.Normal(loc = alpha_0 + alpha_1 * T_true,
                                         scale = sigma_T_meas)
    
    # Sample from this distribution and declare the samples independent in the
    # first two dims. Independence here means that sampling the noise is done 
    # independently of device or T_true. It does not mean that the observations
    # have "nothing to do" with each other - the parameters impact all of them.
    with pyro.plate('device_plate', dim = -2):
        with pyro.plate('measure_plate', dim = -1):
            obs = pyro.sample('observations', obs_dist, obs = observations)
    
    return obs
                      

# ii) Illustrate the model

graphical_model = pyro.render_model(model = model, model_args= (T_true,),
                                    render_distributions=True,
                                    render_params=True)
graphical_model


# iii) Build the guide

def guide(input_vars = T_true, observations = None):
    pass



"""
    3. Perform inference
"""


# i) Set up inference

adam = pyro.optim.Adam({"lr": 0.1})
elbo = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(model, guide, adam, elbo)

# Record example output of the model prior to training
model_data_pretraining = copy.copy(model(T_true)).detach().numpy()


# ii) Perform svi
# Now some meaningful optimization happens. The gradients of the ELBO w.r.t. the
# parameters are computed and the params are adjusted to decrease the ELBO loss.

data = (T_true, T_meas)
loss_sequence = []
for step in range(300):
    loss = svi.step(*data)
    loss_sequence.append(loss)
    if step %10 == 0:
        print(f'epoch: {step} ; loss : {loss}')
        
# Record example output of the model after training
model_data_posttraining = copy.copy(model(T_true)).detach().numpy()


# iii) Additional investigations

# We can inspect the model using pyro functionality. For this, we let the model
# run once and record the trace, a representation of everything important.
model_trace = pyro.poutine.trace(model).get_trace(T_true, observations = T_meas)
print('These are the shapes of the involved objects : \n{} \nFormat: batch_shape,'\
      ' event_shape'.format(model_trace.format_shapes()))
# All info is contained in the model nodes:
print(model_trace.nodes)

# The parameters are stored in pyro's param_store and can be accessed by name
alpha_0 = pyro.get_param_store()['alpha_0']
alpha_1 = pyro.get_param_store()['alpha_1']
   
# We can again compute the log probs by hand to evaluate the fit
obs_dist = pyro.distributions.Normal(loc = alpha_0 + alpha_1* T_true,
                                     scale = sigma_T_meas)
log_prob = torch.sum(obs_dist.log_prob(T_meas))

# Note: the log_prob values are also accessible by inspection of the model trace.
model_trace.compute_log_prob()
log_prob_from_trace = torch.sum(model_trace.nodes["observations"]["log_prob"])
    
# What can we expect in terms of results?
# Since there are still not latent variables, minimizing the ELBO loss becomes
# maximizing the likelihod p_theta(x) where theta = (alpha_0, alpha_1). As the
# model is linear in the parameters and the sole involved distribution is Gaussian,
# estimating alpha_0, alpha_1 via ELB) is equivalent to doing least squares.
# Since we start with initial values of alpha_0 = 0 and alpha_1 = 1, the starting
# ELBO loss should be equal to the loss for model_0 (0 offset, scale = 1). Then
# any further adjustments should decrease the loss and make the observations
# more likely. However, we dont expect gigantic leaps in quality - the model is
# still mostly incorrect.

# If we perform least squares to estimate the coefficients alpha_0, alpha_1,
# the result will coincide with what pyro produces.
A = torch.vstack((torch.ones([n_device*n_measure]), T_true.flatten())).T
alpha_ls = torch.linalg.pinv(A.T@A)@A.T@T_meas.flatten()
print('The least squares solution is alpha_0 = {}, alpha_1 = {};\n'
      'The pyro solution is alpha_0 = {}, alpha_1 = {}'.format(alpha_ls[0], alpha_ls[1], alpha_0, alpha_1))


"""
    4. Interpretations and illustrations
"""

# i) Plot and print ELBO loss

# The ELBO is simply the - log evidence = - log probability of the data for
# this simple model. As computed above by the product of Gaussian probability
# density values.
print(' ELBO loss : {} ,\n - log prob of data : {}'.format(loss, -log_prob))

# The training now adjusts the parameters in such a way that the loss decreases.
plt.figure(1, dpi = 300)
plt.plot(loss_sequence)
plt.yscale("log")
plt.title('ELBO loss during training (log scale)')
plt.xlabel('Epoch nr')
plt.ylabel('value')


# ii) Compare model output and data

# Create the figure and 1x5 subplot grid
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=False, sharey=False)
# Global y-axis limits
y_min = T_meas.min()
y_max = T_meas.max()

# First plot: measurement data
for i in range(n_device):
    axes[0].scatter(T_true[i,:], T_meas[i,:], s=10, label=f"T_meas: sensor[{i}]")
axes[0].set_title("Measurement data")
axes[0].set_xlabel("T_true")
axes[0].set_ylabel("T_meas")
axes[0].set_ylim(y_min, y_max)

# Second plot: data produced by model pre-training
for i in range(n_device):
    axes[1].scatter(T_true[i,:], model_data_pretraining[i,:], s=10, label=f"T_meas: sensor[{i}]")
axes[1].set_title("Data sampled from model pre-training")
axes[1].set_xlabel("T_true")
axes[1].set_ylabel("T_meas")
axes[1].set_ylim(y_min, y_max)

# Third plot: data produced by model post-training
for i in range(n_device):
    axes[2].scatter(T_true[i,:], model_data_posttraining[i,:], s=10, label=f"T_meas: sensor[{i}]")
axes[2].set_title("Data sampled from model post-training")
axes[2].set_xlabel("T_true")
axes[2].set_ylabel("T_meas")
axes[2].set_ylim(y_min, y_max)
axes[2].legend()
    
    
# iii) Illustrate residuals

# compute residuals
residuals_data_pretrain = (T_true - T_meas).detach().numpy()
residuals_data_posttrain = (alpha_0 + alpha_1 * T_true - T_meas ).detach().numpy()

# Residual histograms for model pre- and post-training
fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=120)

axes[0].hist(residuals_data_pretrain.flatten(), bins=30, edgecolor='black')
axes[0].set_title("Residuals: Pre-trained model - measured")
axes[0].set_xlabel("Residual")
axes[0].set_ylabel("Count")
axes[0].grid(True)

axes[1].hist(residuals_data_posttrain.flatten(), bins=30, edgecolor='black')
axes[1].set_title("Residuals: Post-trained model - measured")
axes[1].set_xlabel("Residual")
axes[1].set_ylabel("Count")
axes[1].grid(True)

plt.tight_layout()
plt.show()




    