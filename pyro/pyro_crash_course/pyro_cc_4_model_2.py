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
    

This script will build a first simple model featuring latent variables - that
means the model contains random quantities that are not directly observed. The
advantage over unknown but deterministic parameters lies in the possibility to
include prior information and to infer posterior distributions. These summarize
the state of knowledge of the variables after incorporating all observations.
We thereby do not expand the previous model_1 but swap parameters for latent
variables which makes the whole approach more bayesian. This model_2 is still
simple and in terms of performance will not produce any great improvements. But
it will allow us to do more than can be done with simple least squares.

# Our model assumes the thermistors generate measurements T_meas which are made
made up from scaled T_true and some offset (and noise).  We will assume that the
scale and offset are unknown random variables. In contrast to model_1 where they 
were seen as unknown constants, in this model_2 they will be known prior to any
observation but only up to some uncertainty. Think of e.g. scale and offset being
roughly quantified in the sensors specsheets as 0 +-0.01 or them being measured
in a previous campaign or something similar. Since we will assume the standard
deviation of the noise to be fixed, the model has again no trainable parameters.

However, the guide as a model of the posterior distribution will now be nontrivial
and itself have some parameters indicating mean and variance of the posterior
distribution of offset and scale. These parameters will have to be trained. The
consequence of this model formulation is that after training we will have to our
disposal a posterior distribution that we can sample from. This enables us to
not only determine the most probable values for offset, scale but also how much
to trust them since we can interpret the spread of the posterior distribution as
a measure of our own uncertainty.

In the end, this model_2 showcases the basics of building a (model, guide) pair
featuring latent random variables and how the associated posterior distributions
are inferred by pyro's inference machinery.

For this, do the following:
    1. Imports and definitions
    2. Build model and guide
    3. Perform inference
    4. Interpretations and illustrations
    
In this script we keep explanations of previously introduced building blocks short;
for explanations on pyro.sample, pyro.plate, pyro.param and other basic commands,
see the previous scripts.
    
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
# with some uncertainty. 
# We might write this as: T_meas = a_0 + a_1*T_true + noise.
# Or similarly as the probabilistic model 
# a_0 ~ N(0, 0.1)
# a_1 ~ N(1, 0.1)
# mu_T_meas = a_0 + a_1*T_true
# T_meas ~ N(mu_T_meas, sigma_T_meas)

# Define priors
mu_alpha_0 = 0
mu_alpha_1 = 1
sigma_alpha_0 = 0.1
sigma_alpha_1 = 0.1


# Build the model in pyro
def model(input_vars = T_true, observations = None):
    # We dont need pyro.param anymore but instead we sample offset and scale from
    # the prior distributions. This will declare them as random and require us
    # to specify a variational distribution later in the guide.
    alpha_0_prior = pyro.distributions.Normal(mu_alpha_0, sigma_alpha_0)
    alpha_0 = pyro.sample('alpha_0', alpha_0_prior)
    alpha_1_prior = pyro.distributions.Normal(mu_alpha_1, sigma_alpha_1)
    alpha_1 = pyro.sample('alpha_1', alpha_1_prior)
    
    # Build the observation distribution using the sampled alpha_0 and alpha_1.
    obs_dist = pyro.distributions.Normal(loc = alpha_0 + alpha_1 * T_true,
                                         scale = sigma_T_meas)
    
    # Sample from this distribution and declare the samples independent in the
    # first two dims. Independence here means that sampling the noise is done 
    # independently of device or T_true. It does not mean that there are independent
    # alpha_0, alpha_1 for each device and measurement..
    with pyro.plate('device_plate', dim = -2):
        with pyro.plate('measure_plate', dim = -1):
            obs = pyro.sample('observations', obs_dist, obs = observations)
    
    return obs
                      

# ii) Build the guide
# The guide ( = variational distribution) is now nontrivial for the first time.
# Since we sampled alpha_0, alpha_1 in the model but never observed them, we 
# need to specify a family q_phi of distributions q parametrized by phi that
# approximate the posterior distributions of alpha_0, alpha_1. The parameters phi
# are then adjusted by training such that q_phi is actually close to the true
# posterior. This is done by the second ELBO term that played no role up until
# now since we never had any latents variables.

# Build the guide
def guide(input_vars = T_true, observations = None):
    # We will approximate the posterior distributions of alpha_0, alpha_1 by
    # Normal distributions with unknown means and standard deviations. These
    # are now our trainable parameters. Since standard deviations need to be
    # positive, this is added as a constraint.
    alpha_0_mu_post = pyro.param('alpha_0_mu_post', init_tensor = torch.zeros([]))
    alpha_1_mu_post = pyro.param('alpha_1_mu_post', init_tensor = torch.ones([]))
    
    alpha_0_sigma_post = pyro.param('alpha_0_sigma_post', init_tensor = torch.ones([]),
                                    constraint = pyro.distributions.constraints.positive)
    alpha_1_sigma_post = pyro.param('alpha_1_sigma_post', init_tensor = torch.ones([]),
                                    constraint = pyro.distributions.constraints.positive)

    # In the guide we need to sample the unobserved latents to declare the model
    # for their posteriors. So we first build the posterior distributions, then
    # sample from them using pyro.sample and the same names "alpha_0", "alpha_1"
    # as in the model.
    alpha_0_dist_post = pyro.distributions.Normal(loc = alpha_0_mu_post,
                                                  scale = alpha_0_sigma_post)
    alpha_1_dist_post = pyro.distributions.Normal(loc = alpha_1_mu_post,
                                                  scale = alpha_1_sigma_post)

    alpha_0 = pyro.sample('alpha_0', alpha_0_dist_post)
    alpha_1 = pyro.sample('alpha_1', alpha_1_dist_post)

    return alpha_0, alpha_1


# iii) Illustrate model and guide

# Now the illustration looks slightly different. Where alpha_0, alpha_1 where
# parameters before, now they are random variables illustrated with ellipses.
# At the same time the guide is now nontrivial and worth rendering, too. It
# features the unknown means and standard deviations of the posteriors.
graphical_model = pyro.render_model(model = model, model_args= (T_true,),
                                    render_distributions=True,
                                    render_params=True)
graphical_guide = pyro.render_model(model = guide, model_args= (T_true,),
                                    render_distributions=True,
                                    render_params=True)

graphical_model
graphical_guide


# iv) Record example outputs of model and guide prior to training
# In this model_2, randomness plays a much bigger role than in model_0 or model_1
# where randomness only entered via i.i.d. noise. Now the offset and scale of the
# model are chosen at random, so sampling the model multiple times actually leads
# to different lines (then sampled noisily). We sample multiple times the model
# (producing a full set of [n_device, n_measure] observations each time) and we
# also sample multiple times the guide (producing alpha_0, alpha_1 each time).
# We also sample the guide, then plug these samples into the model to emulate
# the posterior predictive simulation. This allows us to see how the training 
# changes model and guide.
# In total, we will evaluate three distributions:
#   prior predictive distribution : p_theta(x,z) = p_theta(x|z)*p_theta(z)
#       The distribution of data x when using the model() function. Samples latents
#       z from the prior, then runs the rest of the model. Is equivalent to just
#       calling the model() with our prior assumptions.
#   posterior distribution : q_phi(z) (approximation to p_theta(z|x))
#       The distribution q_phi we use as an approximation to the true posterior
#       p_theta(z|x) of the latents z given observed data x. Is equivalent to just
#       calling the guide().
#   posterior predictive distribution : p_theta(x'|x)
#       The distribution of new data x' given old data x. Shows how the model 
#       would generate new data x' given that it has observed and performed inference
#       with observed data x. Learning from x means that posterior distributions
#       for the latents z are used and then plugged into the rest of the model.
#       Is equivalent to  calling guide() to produce latents z and then conditioning
#       the model() on these latents to sample new data x'.
#
#    
# We use pyro.infer.Predictive for that. This construction takes as input the model
# and/or the guide. If only one function (model or guide) is provided, that function
# is run forward to sample it. If both functions are provided, Predictive runs the
# guide and plugs sampled values of alpha_0, alpha_1 into the appropriate positions
# in the model function thereby conditioning model on the latent samples of the
# guide. 


n_model_samples = 30
n_guide_samples = 1000  

predictive = pyro.infer.Predictive
prior_predictive_pretrain = predictive(model, num_samples = n_model_samples)()['observations']
posterior_pretrain_dict = predictive(guide, num_samples = n_guide_samples)()
posterior_pretrain = torch.vstack((posterior_pretrain_dict['alpha_0'],
                                   posterior_pretrain_dict['alpha_1'])).T
posterior_predictive_pretrain = predictive(model, guide = guide, num_samples = n_model_samples)()['observations']

    

"""
    3. Perform inference
"""


# i) Set up inference

adam = pyro.optim.Adam({"lr": 1.0})
elbo = pyro.infer.Trace_ELBO(num_particles = 5)
svi = pyro.infer.SVI(model, guide, adam, elbo)


# ii) Perform svi
# Now some meaningful optimization happens. The gradients of the ELBO w.r.t. the
# parameters are computed and the params are adjusted to decrease the ELBO loss.

data = (T_true, T_meas)
loss_sequence = []
for step in range(1000):
    loss = svi.step(*data)
    loss_sequence.append(loss)
    if step %50 == 0:
        print(f'epoch: {step} ; loss : {loss}')
        
# What does the model/guide structure actually mean for the ELBO and what happens
# now during optimization?
# We know that ELBO = -log p_theta(x) + DKL(q_phi(z) || p_theta(z|x))
# Whereas before the DKL term vanished and p_theta(x) was just the simple to
# evaluate likelihood, now both terms exist and are a bit more complicated.
# -log p_theta(x) :
#   p_theta(x) in the presence of a latent z is actually \int p_theta(x,z) dz
#   which cannot really be evaluated analytically in general. But it can be easily
#   estimated by sampling some z and then evaluating p_theta(x,z). The overall
#   consequence is that you only get noisy estimates of this term, this is why
#   we set num_particles for the ELBO; to better estimate the average p_theta(x)
#   and decrease estimation variance.
# DKL(q || p) :
#   This term does not vanish anymore with the divergence between the model for
#   the posterior (q_phi(z)) and the true posterior (p_theta(z|x)) being relevant.
#   The model for the posterior is given by the guide and the guides parameters
#   phi are the parameters of the posterior distribution alpha_0_mu_post, ... etc.
#   This term is also not a deterministic quantity needs to be estimated by 
#   sampling z from the guide.
# Overall now, the ELBO is a stochastic quantity that has to be sampled instead
# of deterministically computed. That is  also the reason why the ELBO loss plots
# look so noisy. Furthermore, the DKL term is nonnegative, so the ELBO is now
# strictly bigger than -log p_theta(x). So we cannot directly compare the values
# of the ELBO for model_0, model_1 and this new model_2 as the ELBO for the current
# model_2 also includes the approximation error between guide and true posterior.
# In the case where we would have found the parameters that approximated the
# posterior perfectly (and would have used so many particles as to estimate the ELBO
# almost noiselessly), the DKL term would vanish and the log p term would be the
# same as in model_1. We see here the additional complexities brought in by latent
# variables that make ELBO computation noisy and approximation quality issues
# overlay our criteria of best model fit. That is the price we pay for bayesian
# inference for general models, though. The inference = optimization becomes
# something hard to solve with final performance of the estimator depending on
# optimization hyperparameters.
        

# iii) Record example outputs of model and guide after training

prior_predictive_posttrain = predictive(model, num_samples = n_model_samples)()['observations']
posterior_posttrain_dict = predictive(guide, num_samples = n_guide_samples)()
posterior_posttrain = torch.vstack((posterior_posttrain_dict['alpha_0'],
                                   posterior_posttrain_dict['alpha_1'])).T
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
   

# Note: the log_prob values are also accessible by inspection of the model trace.
# But now they do not exactly correspond to the ELBO.
model_trace.compute_log_prob()
log_prob_from_trace = torch.sum(model_trace.nodes["observations"]["log_prob"])
    
# What can we expect in terms of results?
# Now we are doing proper bayesian inference with latent variables and therefore
# our result is not simply the maximum likelihood estimator anymore. The model
# is still linear in the latents though and we cannot expect this model to perform
# any better than the previous model_1. Actually, we could expect it to perform
# less reliable even since we have some latent variables that get reinitialized
# for each model call. So qualitywise there is no improvement expected. In terms
# of insight, what we get from the inference is some additional knowledge of
# uncertainty of the model parameters. The ELBO is sampled during each optimization
# step and is therefore expected to be noisy with the optimization leading to
# a downward trend but nothing at all monotonous.


# The actual closed form solution can be computed for this simple problem. We
# compare it to the posterior distribution that pyro came up with to show they
# match. This actually requires a bit of work and using the conditional distribution
# of a multivariate gaussian of x = [alpha, measurements] with mu_x, Sigma_xx
# derived from the properties of alpha and measurements.
# we arrive at 
#   y = measurements
#   a = alpha = params
#   mu_y = T_true
#   Sigma_xx = [Sigma_aa, Sigma_ay]
#            = [Sigma_ya, Sigma_yy]
#   Sigma_ay = Sigma_aa A^T
#   Sigma_yy = A Sigma_aa A^T + Sigma_noise
#   A = design matrix = [1, T_true]
#   mu_bar = mu_alpha + Sigma_a A^T (A Sigma_alpha A^T + Sigma_noise)^-1(y - mu_y)
#   Sigma_bar =
# where mu_bar, Sigma_bar are the posterior mean and covariance matrix, i.e
# alpha|measurements ~ N(mu_bar, Sigma_bar). Note that Sigma_bar is actually not
# a diagonal matrix so by just declaring alpha_0, alpha_1 to be independently 
# distributed in our guide, we made a modelling mistake. To rectify that, we could
# introduce for alpha the multivariate normal distribution and declare a mean vector
# and covariance matrix as unknown parameters in the guide.

A = torch.vstack((torch.ones([n_device*n_measure]), T_true.flatten())).T
mu_alpha = torch.tensor([mu_alpha_0, mu_alpha_1], dtype = torch.float64)
Sigma_alpha = torch.tensor([[sigma_alpha_0**2, 0], [0, sigma_alpha_1**2]], dtype = torch.float64)
Sigma_noise = (sigma_T_meas**2)*torch.eye(n_device*n_measure)
Sigma_y = A@Sigma_alpha@A.T + Sigma_noise

mu_alpha_cf = mu_alpha + Sigma_alpha @ A.T@torch.linalg.pinv((A@Sigma_alpha@A.T
                        +  Sigma_noise)) @(T_meas.flatten() - T_true.flatten())
Sigma_alpha_cf = Sigma_alpha - Sigma_alpha @ A.T@torch.linalg.pinv((A@Sigma_alpha@A.T
                        + Sigma_noise)) @A@Sigma_alpha

print('The closed form solution is \n alpha_0_mu_post = {}, alpha_0_sigma_post = {};\n'
      'alpha_1_mu_post = {}, alpha_1_sigma_post = {};\n'
      .format(mu_alpha_cf[0], torch.sqrt(Sigma_alpha_cf[0,0]), 
              mu_alpha_cf[1], torch.sqrt(Sigma_alpha_cf[1,1])))

# When we compare inferred and true posterior distribution parameters, we find 
# that the means match well but the standard deviations / variances are underestimated
# by svi-based inference. This is a known phenomenon for wrongly declared covariance
# models, see e.g. Variational Inference: A Review for Statisticians by Blei et
# al. page 9.



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


# ii) Compare model output and data

# Create the figure and 1x5 subplot grid
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
axes[0,3].hist2d(posterior_pretrain[:,0].numpy(), posterior_pretrain[:,1].numpy(),
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
axes[1,3].hist2d(posterior_posttrain[:,0].numpy(), posterior_posttrain[:,1].numpy(),
                 bins=10, cmap='viridis')
axes[1,3].set_title("2D Histogram of parameters post-train")
axes[1,3].set_xlabel("alpha_0")
axes[1,3].set_ylabel("alpha_1")

plt.tight_layout()
plt.show()


# Note that here we illustrate multiple realizations of the model. We notice that
# after training the band of values considered possible has been shrinked down
# and especially the scale is known with much more certainty as assumed before.
    



    