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
    

This script will build a toy model that acts as a trivial baseline for comparing
and interpreting performance of subsequent models. This model_0 is a trivial
model that assumes measured temperatures T_meas to be true temperatures but with
a bit of noise on top. Since we will assume the standard deviation of the noise
to be fixed, the model has no trainable parameters. Therefore inference does not
produce any meaningful model adjustments but the elbo still tells us at least the
log evidence and we can use this as a proxy for model quality - even though we
need to be careful not to overinterpret. In the end, this model_0 showcases the 
basics of building a (model, guide) pair that can be passed to pyro's inference
machinery, and how to trigger that machinery.

For this, do the following:
    1. Imports and definitions
    2. Build model and guide
    3. Perform inference
    4. Interpretations and illustrations
    
The script is meant solely for educational and illustrative purposes. Written by
Dr. Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.com.
"""



"""
    1. Imports and definitions
"""


# i) Imports
# We need pyro for building model, guide and performing inference, this is our
# central package. torch is needed because the typical results of pyro sampling
# or declaring parameters are torch.tensors. pandas is used for importing the csv
# with the measurement data and matplotlib is used for visualization.

import pyro
import torch
import pandas
import matplotlib.pyplot as plt


# ii) Definitions
# Since we have a very simple model of T_meas being T_true with noise on top, we
# need almost no additional definitions apart of what we can derive from the 
# imported data. T_meas, and T_true are both of shape [n_device, n_measure] and
# come from importing the csv file we generated during the last script.

# Read csv, infer dimensions
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
# We ignore almost every complexity we might think of to model our data and start
# with (almost) the simplest model assuming that essentially our measurements are
# right but a bit noisy. We might write this as: T_meas = T_true + noise.
# Similarly, in terms of a probabilistic model directly referring to probability
# distributions, we might write: 
# mu_T_meas = T_true
# T_meas ~ N(mu_T_meas, sigma_T_meas)

# We now recreate this probabilistic model in pyro using the normal distribution
# from pyros distribution module and the sample primitive.
def model(input_vars = T_true, observations = None):
    # Distribution of observations is named obs_dist and has mean T_true and 
    # known standard deviation.
    obs_dist = pyro.distributions.Normal(loc = T_true, scale = sigma_T_meas)
    
    # Sampling from that distribution produces observations in this simple model,
    # i.e. obs = sample from obs_dist. For simulation, just calling pyro.sample
    # on the obs_dist would be enough. However, when we want to perform inference,
    # we have to be explicit about which dimensions are to be treated as independent.
    
    # This is done with the pyro.plate command. It produces a context with which
    # we declare independence, then pyro knows exactly over which dimensions it
    # can average during inference. Here we consider i.i.d noise so the normal
    # distribution samples are to be considered as independent in their two dims.
    with pyro.plate('device_plate', dim = -2):
        with pyro.plate('measure_plate', dim = -1):
            obs = pyro.sample('observations', obs_dist, obs = observations)
    # The above pyro.plate statement does the following:
    # It declares dim -2 and dim -1 (counted from the right) as independent.
    # As pyro.sample is executed in this context and produces a tensor of dim 2
    # with shape [n_device, n_measure], both of these dims are now considered 
    # independent. pyro.plate allows us to inform inference about our independence
    # assumptions.
    
    # The above pyro.sample statement does the following: 
    #   i) It samples from obs_dist, our distribution of observations and returns
    #       the randomly sampled numbers. 
    #   ii) These  observations are getting a unique name string 'observations'
    #       so that they can be uniquely identified wherever the pop up in some 
    #       later processing steps.
    #   iii) If some tensor with actual values is passed as the 'obs' argument 
    #       into the pyro.sample function, then sample returns these values.
    # The reason for the duality of behaviors described in i) and iii) is that
    # it allows simulation and inference to happen with the same function pyro.sample
    # depending on if the obs argument is None or some tensor. If obs = None, 
    # some new values are generated randomly. If instead obs = tensor, then 
    # pyro.sample does not produce new values by sampling from obs_dist. Instead
    # it plugs these values into obs_dist to evaluate their likelihood.
    
    return obs
                      


# ii) Illustrate the model
# We can use pyro to visualize the model we just built. For this, we call the 
# model with some arguments and let pyro trace the distributions and sample calls.
graphical_model = pyro.render_model(model = model, model_args= (T_true,),
                                    render_distributions=True,
                                    render_params=True)

# If you want to print that model illustration, run the following line in isolation.
graphical_model
# You will see our model represented in classical Buntine plate notation. The
# plates/rectangles represent independent contexts over which we iterate. The 
# ellipses represent random quantities that are sampled, here the observations
# according to the normal distribution.


# iii) Build the guide
# The guide function is supposed to specify a family q_phi of functions q parametrized
# by phi. The function q_phi represents the he posterior distribution with the
# parameters phi then learned in such a way that q_phi is as close to the true
# posterior distribution as possible.
# Since there are not latent variables in our model, there exists no posterior
# over latent variables that would need to be learned. Consequently the guide 
# can be the empty function since it does not need to represent anything and
# therefore does not need to contain any parameters of the posterior that need
# to be learned. To allow proper calling of the guide during inference, it needs
# to have the same input signature as the model, though.

def guide(input_vars = T_true, observations = None):
    pass



"""
    3. Perform inference
"""


# i) Set up inference
# This requires us specifying an optimizer, a loss function, and the inference
# machinery. All of this is not very interesting right now as training will be
# meaningless since there are no learnable parameters in our model.
adam = pyro.optim.Adam({"lr": 0.1})
elbo = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(model, guide, adam, elbo)

# The only thing worth noting and briefly discussing may be the loss function.
# The ELBO (evidence lower bound) is defined as 
# Elbo(theta, phi) = -log p_theta(x) + DKL (q_phi(z)|| p_theta(z|x))
# In the absence of any latent variables, the Kullback Leibler Divergence term
# vanishes and only the evidence term - log p_theta(x) remains. Since there are
# no learnable parameters theta, -log p_theta(x) = -log p(x) where x is the data
# and p is just the normal distribution obs_dist, i.e. p(x) = prod Normal(x_i). 


# ii) Perform svi

# Normally, in the training loop pyro and pytorch would compute the gradients 
# of the ELBO w.r.t. the parameters. This involves deterministic differentiation
# but also some Monte Carlo sampling. Here the training loop is vacuous, though.
# We just compute the loss, change nothing, print the loss. In the next models,
# already some meaningful computations happen in this step.
data = (T_true, T_meas)
loss_sequence = []
for step in range(100):
    loss = svi.step(*data)
    loss_sequence.append(loss)
    if step %10 == 0:
        print(f'epoch: {step} ; loss : {loss}')
   
# We will also compute the log evidence by hand to show that the ELBO has the 
# very specific meaning of being the - log evidence in this example.
# For this, we take the obs_dist and pass to it each measurement separately, then
# multiply the probabilities / add the log probabilities.
obs_dist = pyro.distributions.Normal(loc = T_true, scale = sigma_T_meas)
log_prob = torch.sum(obs_dist.log_prob(T_meas))

# We can sample our model to produce some new data and evaluate how good our 
# model is also visually (not only computationally via the ELBO). If we had 
# a good model, the ELBO would be low and the data produced by our model would
# be hard to distinguish from our measurement data. Both is not the case (and
# we will improve our model in the next few notebooks).
model_data = model(T_true)



"""
    4. Interpretations and illustrations
"""

# i) Plot and print ELBO loss

# The ELBO is simply the - log evidence = - log probability of the data for
# this simple model. As computed above by the product of Gaussian probability
# density values.
print(' ELBO loss : {} ,\n - log prob of data : {}'.format(loss, -log_prob))

# The training does not adjust the model since the model has no adjustable parameters
plt.figure(1, dpi = 300)
plt.plot(loss_sequence)
plt.yscale("log")
plt.title('ELBO loss during training (log scale)')
plt.xlabel('Epoch nr')
plt.ylabel('value')


# ii) Compare model output and data

# Create the figure and 1x5 subplot grid
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=False, sharey=False)
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

# Second plot: data produced by model
for i in range(n_device):
    axes[1].scatter(T_true[i,:], model_data[i,:], s=10, label=f"T_meas: sensor[{i}]")
axes[1].set_title("Data sampled from model")
axes[1].set_xlabel("T_true")
axes[1].set_ylabel("T_meas")
axes[1].set_ylim(y_min, y_max)
axes[1].legend()
    
    
    