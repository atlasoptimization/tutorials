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


"""
    2. Build model producing a single batch of data
"""


"""
    3. Perform model inspection with pyro.poutine
"""


"""
    4. Build model producing multiple independent batches
"""


"""
    5. Perform model inspection with pyro.poutine
"""


"""
    6. Notes on model dimensions
"""