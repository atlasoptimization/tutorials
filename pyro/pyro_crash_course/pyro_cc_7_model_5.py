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
    

This script will expand on the previous model_4. It features the per-device random
variabes alpha, the per-device latent discrete random variable is_faulty and now
additionally some shared nonlinear trend among all devices. That trend is modelled
as a Neural Network with unknown parameters. These parameters are later adjusted
during inference to increase the evidence of the data. With this model component,
our crash course comes to an end: With latent continuous and discrete random variables,
nonlinearities, and learnable parameters, we have the main ingredients we need to
already build a wide range of practically relevant models. The resulting model
should also show the most performant behavior yet with different devices, outliers,
nonlinear trends, and noise in the measurement data all modelled reasonably well.

Our model assumes the thermistors generate measurements T_meas which are made up
from the same process as in model_4. In addition though, we will have an artificial
neural net (ann) represent some nonlinear tendencies. This means, our final model
looks something like this:
    T_meas = offset + scale * T_true + ann(T_true) + noise
where T_true is the true temperature, T_meas, the measured temperature, offset,
and scale are per device_random variables impacted by outliers, ann represents
the drift and noise is i.i.d.

Having an ANN as a regular function with trainable parameters in our model is 
demonstrating the flexibility that pyro and svi offer. Doing bayesian inference 
on a mix of discrete, continuous variables being passed through an ANN is not
something that can be done by hand or with classical least squares (without a lot
of hassle). 

For this, do the following:
    1. Imports and definitions
    2. Build model and guide
    3. Perform inference
    4. Interpretations and illustrations
    
I hope this final example can be a model that you return to when you encounter a
specific task in the future. Then you can kick out a few unwarranted components 
and adapt the rest of the model for specifics of the task. Maybe you even try out
the full model chain going from parameters to latents to hierarchical with some
nonlinearities on whatever future tasks you will encounter. 

Good luck and have fun with pyro,
Jemil
    
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



# i) Define the ANN)
# Building a neural network in pyro uses the pyro.nn.PyroModule class. This is 
# very similar to a pytorch neural net but a few bookkeeping operations are handled
# automatically. Overall, building a neural network involves creating a new class
# (=a blueprint for an object) by specifying the layers and nonlinearities and then
# instantiating that class to create a specific object with specific values adhering
# to that blueprint.
class ANN(torch.nn.Module):
    # Just plug together a simple neural net T_meas - > Drift that takes each entry
    # of T_meas separately and maps it nonlinearly to an associated drift value
    # Since we map 1 value of T_true to 1 value of T_meas [1 -> 1], most dims are 1.
    def __init__(self):
        # Initialize instance using init method from base class
        super().__init__()
        
        # Create a few linear transforms with different parameters so we have a
        # chain of nonlinear x linear to combine input data in a nontrivial way.
        self.lin_1 = torch.nn.Linear(1,8)
        self.lin_2 = torch.nn.Linear(8,8)
        self.lin_3 = torch.nn.Linear(8,1)
        # nonlinear transforms
        self.nonlinear = torch.nn.Tanh()
    def forward(self, t):
        # Define forward computation on the input data T_true.
        # Shape the minibatch so that batch_dims are on left, argument_dims on right
        t = t.reshape([-1, 1])
        
        # Then compute hidden units and output of nonlinear pass
        hidden_units_1 = self.nonlinear(self.lin_1(t))
        hidden_units_2 = self.nonlinear(self.lin_2(hidden_units_1))
        nonlinear_drift = self.lin_3(hidden_units_2)
        
        nonlinear_drift = nonlinear_drift.reshape([n_device, n_measure])
        return nonlinear_drift

# ii) Define the model
# We implement the same model as in the previous notebook but add an additional
# nonlinear drift that is superimposed on the data.
# This may be written as the probabilistic model:
# is_faulty ~ Bernoulli(p_faulty)           coin flip to decide production fault
# Sigma = Sigma_normal or Sigma_faulty      dependent on is_faulty
# a_dist = N([0,1], Sigma)
# a ~ N([0,1], 0.1 * I)
# mu_T_meas = a[:,0] + a[:,1] * T_true + ann(T_true)    <- This is new
# T_meas ~ N(mu_T_meas, sigma_T_meas)
# This addition to the model is overall really straightforward with few conceptual
# issues arising from the addition of the nonlinearity - apart from the obvious
# consequence of much harder inference and the absence of any closed form solutions.
# Compared to notebook 3 featuring unknown parameters and their gradient based
# inference, the addition of a neural net looks more complicated but is conceptually
# very similar: A function with a lot of unknown parameters to be adjusted by
# taking steps in a direction that decreases the loss function.

# Define priors and fixed params
mu_alpha = torch.tensor([0.0, 1.0]).expand(n_device, -1)
Sigma_alpha = torch.tensor([[0.1**2,0], [0, 0.1**2]])
p_faulty = 0.05
Sigma_faulty = 100*Sigma_alpha

# Now instantiate the neural net ann; set the params to double to match the data
ann = ANN().double()

# Build the model in pyro; just mix in ann() like any normal torch function
@config_enumerate
def model(input_vars = T_true, observations = None):
    # Mark the parameters inside of the ann for optimization
    pyro.module("ann", ann)
    
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
            
    # Independence in first dim of alpha_dist, sample n_device times. We call
    # ann(T_true) and add it to the mean, otherwise same code as in model_4.
    with device_plate:
        alpha = pyro.sample("alpha", alpha_dist)
    mean_obs = alpha[:, 0].unsqueeze(1) + alpha[:, 1].unsqueeze(1) * T_true + ann(input_vars)
    obs_dist = pyro.distributions.Normal(loc=mean_obs, scale=sigma_T_meas)

    # Sample from this distribution and declare the samples independent in the
    with pyro.plate('device_plate', dim=-2):
        with pyro.plate('measure_plate', dim=-1):
            pyro.sample("observations", obs_dist, obs=observations)


# ii) Build the guide
# The guide is the same as in model_4 since we only added more parameters (they
# come from the ann() function) but no more latent variables, whose posterior we
# would need to approximate.

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


# iii) Illustrate model and guide

# Now the illustration of the model shows additional parameters representing the
# biases and weights of the linear layers in the neural network. The guide looks
# the same as before.
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
elbo = pyro.infer.TraceEnum_ELBO(num_particles = 10,
                                 max_plate_nesting = 2)
svi = pyro.infer.SVI(model, guide, adam, elbo)


# ii) Perform svi

data = (T_true, T_meas)
loss_sequence = []
for step in range(3000):
    loss = svi.step(*data)
    loss_sequence.append(loss)
    if step %50 == 0:
        print(f'epoch: {step} ; loss : {loss}')
          
# We run this inference for a longer time than for the other models. Learning 
# the nonlinear drift turns out to be a bit difficult after all! The training
# procedure is prone to just learning model_4 and ignoring the nonlinear trend
# by setting it to 0 or some basically linear unexpressive function. To actually
# recover parts of the sinusoidal trend, we increase the randomness of the inference
# by increading the number of steps and decreasing the number of particles. That
# allows better exploration of the model space and taking us out of local minima
# As a downside, running this cell might take 20 mins this is the right point in
# time to already skip ahead with reading the summary and condensing the whole
# of the crash course for yourself.

        
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


# ii) Showcase the nonlinear drift

# SVI adjusted the parameters in the neural net such that the ELBO was reduced.
# We now want to show how the function ann() looks like after training.
plt.figure(2, dpi = 300)
plt.plot(ann(T_true).detach().T)
plt.title('Learned drift function')
plt.xlabel('T_true')
plt.ylabel('Drift value')


# iii) Compare model output and data

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

# The results show that we end up with a lower ELBO loss in total than all the
# other models; each component contributes something towards modelling the data.

# Maybe this is also a good point in time to briefly recapitulate our journey
# through this calibration task.
# model_0: No trainable params Final ELBO: ~ 7.2 k
# model_1: Trainable deterministic alpha. Final ELBO: 6.4 k
# model_2: Latent random variable alpha. Final ELBO: 6.4 k
# model_3: Latent random per-device alpha. Final ELBO: 2.4 k
# model_4: Above + failure model. Final ELBO: 2.0 k
# model_5: Above + nonlinear trend. Final ELBO: 0.8 k

# Each structural upgrade enables the model to explain variance that was previously
# treated as noise. The ELBO tells us how much explanatory power each idea brought.

    