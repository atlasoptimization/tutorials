#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is part of a series of pyro tutorials. During these tutorials, we
will tackle a sequence of problems of increasing complexity that will take us
from understanding simple pyro concepts to fitting models via svi and finally
fitting a model involving hidden variables and neural networks.
The tutorials consist in the following:
    - key bayesian concepts                 (tutorial_0) 
    - first contact with pyro               (tutorial_1)
    - forward model and diagnosis           (tutorial_2)
    - data generation and analysis          (tutorial_3)
    - fitting simple model                  (tutorial_4)
    - fitting multivariate model            (tutorial_5)
    - fitting model with hidden variables   (tutorial_6)
    - fitting model with neural networks    (tutorial_7)

This script will create a simple forward model in pyro and explore the different
aspects and relationships in that model using pyro's diagnosis tools. We will 
learn to incoporate information about conditional independence into models 
using the to_event() and pyro.plate() commands and see how pyro keeps track of
model dependencies with pyro.poutine

For this, do the following:
    1. Imports and definitions
    2. Build model producing a single batch of data
    3. Perform model inspection with pyro.poutine
    4. Build and inspect model producing multiple independent batches
    5. Notes on model dimensions
    
The script is meant solely for educational and illustrative purposes. Written by
Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.com.
"""



"""
    1. Imports and definitions
"""


# i) Imports

import torch
import pyro
import copy
import matplotlib.pyplot as plt

from pprint import pprint


# ii) Definitions




"""
    2. Build model producing a single batch of data
"""


# i) Set up model shapes
# Model shapes should tell us the amount of independent data points (batch_shape)
# and the shape of a single datapoint (event_shape). # Both event_shape and 
# batch_shape together give to pyro an unmistakeable info on how any later data
# is sliced to learn from it.Please note the difference here between these two
# things. In a simple model containing 10 independent measurements of some object's
# length, the event shape would be 1 (just length measured) and the batch_shape
# would be 10 (10 independent datapoints). You could also consider the length
# measurements of the above example to be autocorrelated due to some systematic
# effect persisting over time thereby leading to a batch_shape of 1 and an 
# event_shape of 10.
#
# Therefore, both of these quantities aren't given by the nature of the problem;  
# they are a modelling choice. The more the whole dataset is modelled holistically
# as an outcome of the interpedent real world, the bigger the event_shape and
# the smaller the batch_shape. If instead you decide to go for the most simple
# model that ignores interdependencies, then you will have a bigger batch_shape
# and in the extreme case of your model consisting of pure noise all your data
# is considered independent which implies batch_shape = n_data, event_shape = 1. 
# You will see that in tutorial_3 where we build a more complicated model after
# we have figured out how to build models at all in this tutorial.

n_batch_1 = 1       # dim independent data = 1 single realization
n_event_1 = 2       # dim single realization =  2 values from 2D multivariate Gaussian


# ii) Define model

# In pyro anything can be a stochastic model as long as it combines the typical
# pyro and pytorch primitives in a valid way that allows a sample statement to 
# be executed. The sammpling statement (and later the inference) require that
# for any sample outcome the batch_shape and event_shape are defined. As a basic
# rule this needs to be done with pyro.plate (to designate the batch dimension
# of independent sampling) and distribution.to_event (to designate the dependent
# event dimensions). 
# 
# The standard way to define a stochastic model is as a function that generates
# as output some samples. For this, some parameters are declared, then piped 
# into some distribution and this distribution is then sampled. Some optional
# input argument "observations" can be provided in case some conditioning and 
# inference is supposed to happen later on. It is also possible to have the 
# statements defining a stochastic model inside of classes and their methods
# and attributes. For now we will proceed simple and define a function that 
# generates a single sample.

# Define simple model that generates one realization of a 2D datapoint.
def model_1(observations = None):
    # The model will declare parameters, then create a distribution and sample
    # from it.
    
    # First, declare a parameter mu_1 that is the mean vector of a multivariate 
    # Gaussian. Parameters are tensors that pyro will adjustto increase model
    # fit if trained on data. In comparison, we will just declare the covariance
    # matrix sigma_1 to be some fixed matrix thereby not giving pyro any permission
    # to change it.
    mu_1 = pyro.param(name = 'mu_1', init_tensor = torch.zeros([1,2]))
    sigma_1 = torch.tensor([[1,0.5],[0.5,1]])
    
    # Define a distribution from which to sample. This distribution takes as
    # input the parameter mu_1 and the fixed tensor sigma_1 we defined before.
    dist_1 = pyro.distributions.MultivariateNormal(loc = mu_1, covariance_matrix = sigma_1)
    
    # Sample from the distribution and give a unique name to that sample. As an
    # optional keyword we declare the outcome of that sample statement to be 
    # observed. If we have any data, we could call the model with the argument
    # observations = data to condition sample_1 on the data and let pyro adjust
    # the parameter mu_1.
    sample_1 = pyro.sample('sample_1', dist_1, obs = observations)
    
    return sample_1
    

# iii) Illustrate model in graphical notation

# If you have the appropriate packages installed, you can print out a graphical
# visualization of the model in the console. This allows you to parse the model
# architecture quickly and see which parameters, random variables and independence
# statements the model includes. Random quantities are surrounded by a circle,
# independence statements indicating repeated sampling are symbolized by a
# rectangle ( = the 'plate') and parameters are annotated with arrows to show
# where they exert influence.
pyro.render_model(model_1, model_args=(), render_distributions=True, render_params=True)



# iii) Run model forward

# Run the model once. This produces one sample of shape [1,2] where batch_shape
# is 1 and event_shape is 2.
single_sample_1 = model_1()
print(' A single sample of model_1 produces e.g. {}.\n The shape of the sample is {}'
      .format(single_sample_1, single_sample_1.shape))

# Run the model forward 100 times. This produces 100 samples of shape [1,2] which
# we will manually assemble to shape [100,2] and plot. This works for data creation
# but if we would like to train on 100 datapoints, we would have to adjust the 
# model to produce data of shape [100,2], e.g. by creating a distribution of that
# shape.
many_sample_1 = torch.zeros([100,2])
for k in range(100):
    many_sample_1[k,:] = model_1()
plt.scatter(many_sample_1[:,0].detach(), many_sample_1[:,1].detach())
plt.title('Scatterplot of 100 realizations')

# Run the model forward while providing the observations argument. The outcome
# of that is just the observations argument - the sample outcome has been fixed
# and pyro now could compute gradients of e.g. the likelihood of the data w.r.t.
# the parameter mu_1 and optimize over it. Pyro will complain that this conditioning
# action is only allowed during inference.
conditioned_sample_1 = model_1(observations = torch.tensor([[0,0]]))
print(' The conditioned sample is {}, which is exactly the observations'
      .format(conditioned_sample_1))



"""
    3. Perform model inspection with pyro.poutine
"""


# Since we had a very simple model that was callable without any errors and it
# delivered the expected results, we have been confident that our model was
# specified correctly. Typically, practically relevant models are some complicated
# that their correctness is hard to ascertain by hand. To make the model diagnosis
# easier, pyro offers the poutine submodule which provides tools for displaying
# pyros very formal interpretation of the model.

# i) Basic pyro.poutine functionality
# With pyro.poutine you have access to the "trace" functionality that allows to
# run a model and record what happens inside of the model in form of a model
# trace - this is a structured object that contains methods to display and 
# manipulate conditional dependences and batch_shapes, event_shapes.

# Run model_1 with input "None", build the model_trace and look at some shapes.
# Please confirm that all of these numbers are as you expect them.
model_1_trace = pyro.poutine.trace(model_1).get_trace(None)
print('These are the shapes of the involved objects : \n{} \nFormat: batch_shape,'\
      ' event_shape'.format(model_1_trace.format_shapes()))

# You can also compute the log_prob, i.e the (log of the) probability of the
# sample
model_1_trace.compute_log_prob()

# ii) Analysis full execution trace
# Even more insight than the shapes are recorded in the trace.nodes object. This
# is a dictionary containing further contextual information.

# Construct model nodes and print them
model_1_nodes = model_1_trace.nodes
print(' model_1_nodes contains entries on {} \n'.format(model_1_nodes.keys()))
print(' This is the full representation of the execution trace:')
pprint(model_1_nodes)
# Note how the  entries for mu_1 specifies the name and value of the parameter
# among with several other diagnostics such as its constraints, gradients, etc.
# The entries for sample_1 specify the name and random value (for this specific
# execution) as well as diagnostics such as the distribution, the log_prob, if
# the sample was observed, and its independence relations. sigma_1 is not listed
# in the execution trace since it is neither parameter nor sample - just a fixed
# quantity.


# iii) Function to analyze models

# We would like to produce the diagnostics as under ii) in a more compact way.
# The following function can be called to print model diagnostics such as the
# involved shapes and the model nodes.
def analyze_model(model, trace_input):
    model_trace = pyro.poutine.trace(model).get_trace(trace_input)
    model_trace.compute_log_prob()
    pprint(model_trace.nodes)
    print(model_trace.format_shapes())
    return model_trace


"""
    4. Build and inspect model producing multiple independent batches
"""


# The first model was simple because there was only one nontrivial dimension,
# the one containing the multivariate realizations. When building a model that
# produces multiple independent realizations, we need to declare to pyro which
# dimensions contain independent realizations via the pyro.plate context. In
# addition we notify pyro via the distribution.to_event() function about the
# dimension containing dependent variables. As a rule of thumb, declare all
# dimensions either independent via pyro.plate or dependent via to_event() if 
# the shape of the distribution itself does not do it for you.


# i) Build stochastic model producing multiple realizations

# Define model that generates 10 realizations of a 2D datapoint.
def model_2(observations = None):
    # The model will declare parameters, then create a distribution and sample
    # from it.
    
    # Declare mean and covariance as before
    mu_2 = pyro.param(name = 'mu_2', init_tensor = torch.zeros([1,2]))
    sigma_2 = torch.tensor([[1,0.5],[0.5,1]])
    
    # Define a distribution from which to sample. We expand the distribution
    # in dim 1 so it represents 10 samples. 
    dist_2 = pyro.distributions.MultivariateNormal(loc = mu_2, covariance_matrix = sigma_2).expand([10])
    
    # dist_2 has now batch_shape 10 and event_shape 2. The function to_event(n_dims)
    # declares the rightmost n_dims of the batch dims as event_dims. We do not
    # need to do that here since dim 2 (the rightmost) is already considered to
    # be the event_dim due to usage of the multivariate Gaussian distribution.
    # If you would call dist_2 it would display the following:
    # MultivariateNormal(loc: torch.Size([10, 2]), covariance_matrix: torch.Size([10, 2, 2]))
    
    # Sample from the distribution. Do this inside the pyro plate context to 
    # declare the rightmost of the batch dimensions (dim = -1) as independent.
    with pyro.plate('batch_plate', size = 10, dim = -1):
        sample_2 = pyro.sample('sample_2', dist_2, obs = observations)
    
    return sample_2


# ii) Run model and analyze

# If we run the model now, it produces output of shape [10,2] where the last dim
# indexes the two interdependent events and the first dim indexes independent
# data points.
sample_2 = model_2()
print('The sample from model two has the appropriate shape = {}'.format(sample_2.shape))

model_2_trace = analyze_model(model= model_2, trace_input = None)
# Analyze the output carefully to see if the recorded trace mathes what you expect.

# Illustrate model by plotting symbolic representation
pyro.render_model(model_2, model_args=(), render_distributions=True, render_params=True)

# The log_prob and log_prob sum have shapes [10] and [] respectively reflecting
# the fact that log_prob assesses the probability of each independent sample and
# log_prob_sum sums them into a single scalar suitable for backprop.
log_prob = model_2_trace.nodes['sample_2']['log_prob']
log_prob_sum = model_2_trace.nodes['sample_2']['log_prob_sum']

# Using log_prob_sum, we could easily do gradient descent on the parameter mu_2
# to fit it to the dataset. In practice, we usually call svi for inference.
sample = copy.copy(sample_2.detach())
mu_2 = pyro.get_param_store()['mu_2']
optimizer = torch.optim.Adam([mu_2], lr=0.1)      
for step in range(100):
    # Set the gradients to zero
    optimizer.zero_grad()  
    
    # compute the loss function
    trace = pyro.poutine.trace(model_2).get_trace(sample)
    trace.compute_log_prob()
    loss = -1 * trace.nodes['sample_2']['log_prob_sum']
    
    # compute the gradients, update t_mu
    loss.backward()
    optimizer.step()
    
    if step % 10 == 0:
        print('Step {}, Loss {:.3f}, mu_2 [{:.3f} {:.3f}]'
              .format(step+1, loss.item(), mu_2[0][0].item(), mu_2[0][1].item()))
print(" Optimization results mu_2 = [{:.3f} {:.3f}] \n"\
      " Analytical ML estimate mu_2 = [{:.3f}, {:.3f}]".
      format(mu_2[0][0].item(), mu_2[0][1].item(), 
             torch.mean(sample,0)[0], torch.mean(sample,0)[1]))



"""
    5. Notes on model dimensions
"""





























