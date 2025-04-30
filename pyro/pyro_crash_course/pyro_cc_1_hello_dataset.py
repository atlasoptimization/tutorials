#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is part of a series of notebooks that showcase the pyro probabilistic
programming language - basically pytorch + probability. In this series we explore
a sequence of increasingly complicated models meant to represent the behavior of
some measurement device. 
The crash course consist in the following:
    - minimal pyro example                      (cc_0_minimal_inference)
    - generating data and exploring it          (cc_1_hello_dataset)
    - Model with no trainable params            (cc_2_model_0)
    - Model with deterministic parameters       (cc_2_model_1)
    - Model with latent variables               (cc_2_model_2)
    - Model with hierarchical randomness        (cc_2_model_3)
    - Model with discrete random variables      (cc_2_model_4)
    - Model with neural net                     (cc_2_model_5)
    

This script will build a dataset that will be used for the subsequent models and
therefore already contains all necessary complexities. The dataset represents
measurement data gathered by a thermistor, a device designed to measure temperature.
By generating the data, we decide on the effects and thereby know the ground truth.
Consequently we can authoritatively argue already about which effect corresponds
to which feature in the data. Note that we exaggerate a lot of the effects to
generate a visual impact; the effect magnitudes are not to be mistaken as realistic.

The output of this script will be a simple dataset interpreted as numbers without
much context; we will discard any knowledge of thegenerative process behind it
in the fu ture modelling tasks.

For this, do the following:
    1. Imports and definitions
    2. Generate ground truth temperatures
    3. Generate linear responses per thermistor
    4. Add a gross error to one thermistor
    5. Add nonlinear trend function
    6. Add noise to dataset
    7. Plots and illustrations
    8. Preliminary interpretations
    
The script is meant solely for educational and illustrative purposes. Written by
Dr. Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.com.
"""



"""
    1. Imports and definitions
"""


# i) Imports
# We only use pytorch for generating data at random and matplotlib to visualize
# it. Since there is no inference going on in this script, we use pyro only to
# generate some random data according to some distributions but do not employ
# it in any way to train parameters or keep track of gradients.

import pyro
import torch
import matplotlib.pyplot as plt


# ii) Definitions
# We declare n_device different devices to be used and for simplicity, each of
# these devices were used for a testrun of n_measure temperature measurements.
# In each case, the temperature measurements are made of temperatures T_true
# that have been created in the climate chamber to form a linear progression
# between some minimum temp T_low and some maximum temp T_high. There are n_faulty
# faulty measurement devices that for some reason produce measurements deviating
# significantly from the true temp.

n_device = 5
n_measure = 100

T_low = 10
T_high = 20

n_faulty = 1

pyro.set_rng_seed(0)



"""
    2. Generate ground truth temperatures
"""


# i) Ground truth temperature
# This is kept super simple. As we assume the climate chamber to be absolutely
# trustworthy, our ground truth temperature is simply what we demand the chamber
# to deliver; in this case a linear progression from T_low to T_high.

T_true = torch.linspace(T_low, T_high, n_measure)
T_true = T_true.repeat([n_device,1])



"""
    3. Generate linear responses per thermistor
"""


# i) Define prior distributions for alpha_0, alpha_1
# We assume that each thermistor is governed by its own function mapping T_true
# to T_meas, the measured temperature. The functions are assumed to be simple
# affine equations with offset (alpha_0) and scale (alpha_1). Offset and scale
# are drawn randomly during the production process of the thermistor.

mu_alpha_0 = torch.zeros([n_device,1])
sigma_alpha_0 = 0.1*torch.ones([n_device,1])
mu_alpha_1 = torch.ones([n_device,1])
sigma_alpha_1 = 0.1*torch.ones([n_device,1])

alpha_0_dist = pyro.distributions.Normal(loc = mu_alpha_0, scale = sigma_alpha_0)
alpha_1_dist = pyro.distributions.Normal(loc = mu_alpha_1, scale = sigma_alpha_1)


# ii) Sample to generate offsets and scales
# We sample from the two distributions to get alpha_0 and alpha_1 for each device
# Consequently, alpha_0 and alpha_1 are vectors of shape [n_device]

alpha_0 = pyro.sample('alpha_0', alpha_0_dist)
alpha_1 = pyro.sample('alpha_1', alpha_1_dist)


# iii) Generate linear responses to T_true
# These are computed by passing alpha_0 and alpha_1 into the affine equation 
# T_meas_lin = alpha_0 + alpha_1 * T_true
# where T_true are the true temp values and T_meas_lin here are the values we would
# measure (if not for all the other effects that join the fray later)

T_meas_lin = alpha_0 + alpha_1 * T_true


"""
    4. Add a gross error to one thermistor
"""


# i) Dataset nr 1 contains a gross error
# Represent gross error as measured data T_meas_gross being T_meas_lin but with
# an added offset of 5 deg C.

index_gross = 0
impact_gross = torch.zeros([n_device, n_measure])
impact_gross[index_gross, : ] = 5 
T_meas_gross = T_meas_lin + impact_gross

"""
    5. Add nonlinear trend function
"""


# i) Overlay sine function
# The nonlinearity here is meant to represent some type of temperature dependent
# effect that dampens or increases the measurements dependent on the underlying
# true temperature.

nonlinear_trend = 0.5*torch.sin(T_true)
T_meas_nonlinear = T_meas_gross + nonlinear_trend


"""
    6. Add noise to dataset
"""


# i) Finally, add some noise
# Here we generate numbers chosen randomly from a univariate Gaussian distribution 
# and add them to the data generated previously. Our noise is simple in the sense
# that noise variance is constant and noise samples are uncorrelated.

noise_dist = pyro.distributions.Normal(loc = torch.zeros([n_device, n_measure]), scale = 0.3)
noise = pyro.sample('noise', noise_dist)
T_meas_noisy = T_meas_nonlinear + noise


# Now after this action we finally have some data that contains all the effects
# we want to have in our temperature measurement data: A linear response, slight
# variations per sensor, a gross error, some nonlinear trend, and noise. This
# dataset can now be passed as an input to our sequence of modelling challenges
# - and we will always know how wrong we are exactly.



"""
    7. Plots and illustrations
"""


# i) Plot all of the data
# Here we illustrate the progression from ground truth towards evermore distorted
# data.


# Create the figure and 1x5 subplot grid
fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharex=False, sharey=False)
# Global y-axis limits
y_min = min(T_true.min(), T_meas_noisy.min())
y_max = max(T_true.max(), T_meas_noisy.max())

# First plot: T_true vs T_true
axes[0].scatter(T_true[0,:], T_true[0,:], color='black', s=10)
axes[0].set_title("Case: error-free thermistor")
axes[0].set_xlabel("T_true")
axes[0].set_ylabel("T_true")
axes[0].set_ylim(y_min, y_max)

# Second plot: T_true vs T_meas_lin
for i in range(n_device):
    axes[1].scatter(T_true[i,:], T_meas_lin[i,:], s=10, label=f"T_meas[{i}]")
axes[1].set_title("Case: linear thermistor effects")
axes[1].set_xlabel("T_true")
axes[1].set_ylabel("T_meas")
axes[1].set_ylim(y_min, y_max)

# Third plot: T_true vs T_meas_gross
for i in range(n_device):
    axes[2].scatter(T_true[i,:], T_meas_gross[i,:], s=10, label=f"T_meas[{i}]")
axes[2].set_title("Case: + gross error")
axes[2].set_xlabel("T_true")
axes[2].set_ylabel("T_meas")
axes[2].set_ylim(y_min, y_max)


# Fourth plot: T_true vs T_meas_nonlinear
for i in range(n_device):
    axes[3].scatter(T_true[i,:], T_meas_nonlinear[i,:], s=10, label=f"T_meas[{i}]")
axes[3].set_title("Case: + nonlinear trend")
axes[3].set_xlabel("T_true")
axes[3].set_ylabel("T_meas")
axes[3].set_ylim(y_min, y_max)

# Fifth plot: T_true vs T_meas_noisy
for i in range(n_device):
    axes[4].scatter(T_true[i,:], T_meas_noisy[i,:], s=10, label=f"T_meas[{i}]")
axes[4].set_title("Case: + noise")
axes[4].set_xlabel("T_true")
axes[4].set_ylabel("T_meas")
axes[4].set_ylim(y_min, y_max)
axes[4].legend()


"""
    8. Preliminary interpretations
"""

# We can already make some predictions about how our future models might perform
# and what they will model well, or fail to uncover.
#
#    Model 0: Model with no trainable params
#       This will generate data that is mostly on the diagonal with some noise
#       on top. Most interesting structure will be ignored and data and model
#       will not fit well.
#    Model 1: Model with deterministic parameters
#       This will generate just one single line that will not actually look to 
#       different from model 0. The main aspects obviously not covered are the
#       device-dependend params and the nonlinearity.
#    Model 2: Model with latent variables
#       This will not look very different from Model 1 but at least we will have 
#       some measure of uncertainty regarding the parameters that are estimated.
#    Model 3: Model with hierarchical randomness
#       This will make visually the biggest impact since the single model predictions
#       will be replaced by the device-dependent model predictions.
#    Model 4: Model with discrete random variables
#       Visually this should not make a big impact at all, but the discrete
#       random variable should allow detection of the faulty device.
#    Model 5: Model with neural net
#       This should push the model to produce some data that captures that real
#       measurements almost perfectly. Nonetheless, the model will not be perfect 
#       as it generalizes very badly outside of the training range.







