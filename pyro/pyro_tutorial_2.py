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
    4. Build model producing multiple independent batches
    5. Perform model inspection with pyro.poutine
    6. Notes on model dimensions
    
The script is meant solely for educational and illustrative purposes. Written by
Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.com.
"""



"""
    1. Imports and definitions
"""


# i) Imports

import torch
import pyro
import matplotlib.pyplot as plt



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

def model_1(observations = None):
    


# iii) Run model forward





"""
    3. Perform model inspection with pyro.poutine
"""


# i) Illustrate model outcome

# ii) Basic pyro.poutine functionality

# iii) Analysis full execution trace

# iv) Function to analyze models



"""
    4. Build model producing multiple independent batches
"""


"""
    5. Perform model inspection with pyro.poutine
"""


"""
    6. Notes on model dimensions
"""





























