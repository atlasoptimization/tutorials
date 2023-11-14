#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script aims to illustrate basic pyro functionality by creating a stochastic
model of a levelling process that is contaminated with the occasional gross error.
Gross errors occur at random according to a bernoulli distribution and - after
inclusion into the stochastic model - are automatically identified and corrected
by pyro. In comparison to the LS approach, no specific handholding is necessary;
we just have to let pyro know that sometimes gross errors happen.
For this, do the following:
    1. Imports and definitions
    2. Simulate some data
    3. Build stochastic model 
    4. Statistical inference
    5. Plots and ilustrations
"""



"""
    1. Imports and definitions ------------------------------------------------
"""


# i) Imports

import numpy as np
import torch
import pyro
from pyro.infer import SVI



# ii) Definitions

h_A = 0     # True height of the fixed points is 0m  above sea level
h_B = 0
h_1 = 1     # True height of the unknown points is 1m and -1m (only used for getting data & verification)
h_2 = -1

L_A1 = 1    # Distances between the locations, i.e. L_A1 is the distance between fixpoint A and unknown point 1
L_12 = 2
L_2B = 1

n_observations = 6
pyro.clear_param_store()



"""
    2. Simulate some data ----------------------------------------------------
"""


# i) Means and variances

# True mean is sequence of 
dh_true = np.vstack((h_1 - h_A, h_2 - h_1, h_B- h_2, h_2- h_B, h_1 - h_2, h_A - h_1))
dh_true = torch.tensor(dh_true)

sigma_I = 2e-3         # = 2 mm / sqrt(km)
sigma_true = (sigma_I**2)*np.diag([L_A1, L_12, L_2B, L_2B, L_12, L_A1])
sigma_true = torch.tensor(sigma_true)


# ii) Draw from a Gaussian 

observations_numpy = np.random.multivariate_normal(dh_true.squeeze(), sigma_true)
observations_tensor = torch.tensor(observations_numpy)



"""
    3. Build stochastic model -------------------------------------------------
"""


# i) Function taking as inputs observations and linking them to distributions

A = torch.tensor(np.array([[1, 0],[-1, 1], [0, -1], [0, 1], [1,-1], [-1,0]])).double()
def model_gross_error_levelling(observations = None):
    theta = pyro.param("theta", torch.ones([2]).double())    # theta = estimator for the heights
    # Use theta to define distribution
    multivariate_gaussian_distribution = pyro.distributions.MultivariateNormal(A@theta,sigma_true) 
    # observations defined via probability distribution and theta
    obs = pyro.sample("obs", multivariate_gaussian_distribution, obs = observations)
    # observations contaminated by gross errors
    gross_errors = model_gross_errors()
    obs_gross_errors = obs + gross_errors
    return obs_gross_errors

# ii) Define functions creating gross errors
def model_gross_errors():
    # Flip a coin for each observation
    bernoulli_distribution = pyro.distributions.Bernoulli(0.1*torch.ones([n_observations]))
    gross_error_yesno = pyro.sample("gross_error_yesno", bernoulli_distribution)
    # If coin shows a 1, generate a gross error
    uniform_distribution = pyro.distributions.Uniform(-(gross_error_yesno*3+1e-9), gross_error_yesno*3 + 1e-9)
    gross_error_size = pyro.sample("gross_error_size", uniform_distribution)
    
    return gross_error_size


    

"""
    4. Statistical inference -------------------------------------------------
"""



# i) Create the guide for stochastic variational inference (SVI)

model_guide = pyro.infer.autoguide.AutoNormal(pyro.poutine.block(model_gross_error_levelling, hide=['gross_error_yesno']))


# ii) Run the optimization for SVI

adam = pyro.optim.Adam({"lr": 0.02})
elbo = pyro.infer.TraceEnum_ELBO()
svi = pyro.infer.SVI(model_gross_error_levelling, model_guide, adam, elbo)

losses = []
for step in range(1000):  
    loss = svi.step(observations_tensor)
    losses.append(loss)
    if step % 100 == 0:
        print("Elbo loss: {}".format(loss))



"""
    5. Plots and ilustrations -------------------------------------------------
"""



# i) Calculate LS solution

print('The true heights are {}'.format([h_1, h_2]))

A = np.array([[1, 0],[-1, 1], [0, -1], [0, 1], [1,-1], [-1,0]])
P = np.linalg.pinv(sigma_true)
h_est_LS = np.linalg.pinv(A.T@P@A)@A.T@P@observations_numpy
print('The LS solution for the heights is {}'.format(h_est_LS))



# ii) Print out result from svi

for name, value in pyro.get_param_store().items():
    print(" The pyro solution for the parameter {} is {} ".format(name, pyro.param(name).data.cpu().numpy()))

