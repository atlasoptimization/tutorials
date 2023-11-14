#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script aims to illustrate standard least squares procedure to fit heights
to levelling data.
For this, do the following:
    1. Imports and definitions
    2. Simulate some data
    3. Least squares
    4. Plots and ilustrations
    
Author: Dr. Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.ch.
Copyright: 2021-2023, Atlas optimization GmbH, Zurich, Switzerland. All rights reserved.
"""



"""
    1. Imports and definitions ------------------------------------------------
"""


# i) Imports

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)


# ii) Definitions

h_A = 0     # True height of the fixed points is 0m  above sea level
h_B = 0
h_1 = 0     # True height of the unknown points is 0m and 0m (only used for getting data & verification)
h_2 = 0

L_A1 = 1    # Distances between the locations, i.e. L_A1 is the distance between fixpoint A and unknown point 1
L_12 = 2
L_2B = 1



"""
    2. Simulate some data ----------------------------------------------------
"""


# i) Means and variances

# True mean is sequence of 
dh_true = np.vstack((h_1 - h_A, h_2 - h_1, h_B- h_2, h_2- h_B, h_1 - h_2, h_A - h_1)).squeeze()

sigma_I = 2e-3         # = 2 mm / sqrt(km)
sigma_true = (sigma_I**2)*np.diag([L_A1, L_12, L_2B, L_2B, L_12, L_A1])


# ii) Draw from a Gaussian 

observations = np.random.multivariate_normal(dh_true, sigma_true)



"""
    3. Least squares -------------------------------------------------
"""


A = np.array([[1, 0],[-1, 1], [0, -1], [0, 1], [1,-1], [-1,0]])
P = np.linalg.pinv(sigma_true)
h_est_LS = np.linalg.pinv(A.T@P@A)@A.T@P@observations


"""
    4. Plots and ilustrations -------------------------------------------------
"""



# i) Plot LS solution

print('The true heights are {}'.format([h_1, h_2]))
print('The LS solution for the heights is {}'.format(h_est_LS))

fig_0 = plt.figure(0, figsize = [6,3], dpi = 300)
x_tickpostions = np.linspace(0,6,6)
x_ticks = ["dh_A1" , "dh_12", " dh_2B", "dh_B2", "dh_21", "dh_1A"]
plt.xticks(x_tickpostions, x_ticks)
plt.plot(x_tickpostions, 1e+3*(observations), label = 'observed')
plt.plot(x_tickpostions, 1e+3*(dh_true), label = 'true')
plt.xlabel('height difference at subsection')
plt.ylabel('value of height difference [mm]')
plt.title('Observed height differences vs true height differences')
plt.legend()

fig_1 = plt.figure(1, figsize = [6,3], dpi = 300)
x_tickpostions = np.linspace(0,6,6)
x_ticks = ["dh_A1" , "dh_12", " dh_2B", "dh_B2", "dh_21", "dh_1A"]
plt.xticks(x_tickpostions, x_ticks)
plt.plot(x_tickpostions, 1e+3*(A@h_est_LS), label = 'estimated')
plt.plot(x_tickpostions, 1e+3*(dh_true), label = 'true')
plt.xlabel('height difference at subsection')
plt.ylabel('value of height difference [mm]')
plt.title('Estimated height differences vs true height differences')
plt.legend()

fig_2 = plt.figure(2, figsize = [6,3], dpi = 300)
x_tickpostions = np.linspace(0,6,6)
x_ticks = ["dh_A1" , "dh_12", " dh_2B", "dh_B2", "dh_21", "dh_1A"]
plt.xticks(x_tickpostions, x_ticks)
plt.bar(x_tickpostions, 1e+3*(A@h_est_LS) - dh_true, label = 'errors')
plt.xlabel('height difference at subsection')
plt.ylabel('value of height difference')
plt.title('Errors of height differences [mm]')
plt.legend()

fig_3 = plt.figure(3, figsize = [6,3], dpi = 300)
x_tickpostions = np.linspace(0,6,6)
x_ticks = ["dh_A1" , "dh_12", " dh_2B", "dh_B2", "dh_21", "dh_1A"]
plt.xticks(x_tickpostions, x_ticks)
plt.bar(x_tickpostions, 1e+3*(A@h_est_LS - observations), label = 'residuals')
plt.xlabel('height difference at subsection')
plt.ylabel('value of height difference')
plt.title('Residuals of height differences[mm]')
plt.legend()





















