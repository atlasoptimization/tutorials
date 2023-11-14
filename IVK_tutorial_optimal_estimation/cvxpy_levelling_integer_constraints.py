#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script aims to illustrate using cvxpy to solve a sequence of estimation
problems that showcase the power of integer constraints.
For this, do the following:
    1. Imports and definitions
    2. Simulate some data
    3. Levelling deformation model - boolean constraint
    4. Levelling deformation model - finite set constraint
    5. LS for levelling - logical constraint
    6. LS for levelling - nonlinear objective
"""



"""
    1. Imports and definitions ------------------------------------------------
"""


# i) Imports

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

np.random.seed(0)


# ii) Definitions - levelling network

h_A = 0     # True height of the fixed points is 0m  above sea level
h_B = 0
h_1 = 0     # True height of the unknown points is 0m and 0m (only used for getting data & verification)
h_2 = 0

L_A1 = 1    # Distances between the locations, i.e. L_A1 is the distance between fixpoint A and unknown point 1
L_12 = 2
L_2B = 1



# iii) Definitions - deformation timeseries

n_time = 100
time = np.linspace(0,5,n_time)


# iv) Plotting function

def plot_results_of_levelling_optimization(theta_est):
    fig_1 = plt.figure(1, figsize = [6,3], dpi = 300)
    x_tickpostions = np.linspace(0,6,6)
    x_ticks = ["dh_A1" , "dh_12", " dh_2B", "dh_B2", "dh_21", "dh_1A"]
    plt.xticks(x_tickpostions, x_ticks)
    plt.plot(x_tickpostions, A@theta_est, label = 'estimated')
    plt.plot(x_tickpostions, dh_true, label = 'true')
    plt.xlabel('height difference at subsection')
    plt.ylabel('value of height difference')
    plt.title('Estimated height differences vs true height differences')
    plt.legend()
    plt.show()
    
    fig_2 = plt.figure(2, figsize = [6,3], dpi = 300)
    x_tickpostions = np.linspace(0,6,6)
    x_ticks = ["dh_A1" , "dh_12", " dh_2B", "dh_B2", "dh_21", "dh_1A"]
    plt.xticks(x_tickpostions, x_ticks)
    plt.bar(x_tickpostions, A@theta_est - dh_true, label = 'errors')
    plt.xlabel('height difference at subsection')
    plt.ylabel('value of height difference')
    plt.title('Errors of height differences')
    plt.legend()
    plt.show()
    
    fig_3 = plt.figure(3, figsize = [6,3], dpi = 300)
    x_tickpostions = np.linspace(0,6,6)
    x_ticks = ["dh_A1" , "dh_12", " dh_2B", "dh_B2", "dh_21", "dh_1A"]
    plt.xticks(x_tickpostions, x_ticks)
    plt.bar(x_tickpostions, A@theta_est - observations, label = 'residuals')
    plt.xlabel('height difference at subsection')
    plt.ylabel('value of height difference')
    plt.title('Residuals of height differences')
    plt.legend()
    plt.show()

def plot_results_of_fitting_deformation(theta_est):
    fig_1 = plt.figure(1, figsize = [6,3], dpi = 300)
    plt.plot(time, deformation_timeseries_measured, label = 'measured deformation')
    plt.plot(time, M@theta_est, label = 'deformation model')
    plt.xlabel('Time in y')
    plt.ylabel('measured deformation in mm')
    plt.title('Fit of deformation model')
    plt.legend()
    plt.show()
    

    

"""
    2. Simulate some data ----------------------------------------------------
"""


# i) Simulate data for levelling network

# True mean is sequence of 
dh_true = np.vstack((h_1 - h_A, h_2 - h_1, h_B- h_2, h_2- h_B, h_1 - h_2, h_A - h_1)).squeeze()

sigma_I = 2e-3         # = 2 mm / sqrt(km)
sigma_true = (sigma_I**2)*np.diag([L_A1, L_12, L_2B, L_2B, L_12, L_A1])

A = np.array([[1, 0],[-1, 1], [0, -1], [0, 1], [1,-1], [-1,0]])
Sigma_sqrt_inv = np.sqrt(np.linalg.pinv(sigma_true))

# Draw from a Gaussian 
observations = np.random.multivariate_normal(dh_true, sigma_true)


# ii) Simulate data for deformation timeseries
# Out model is a combination of a linear trend and a sine with yearly period

noise = np.random.normal(0,1e-3,n_time)
deformation_timeseries_true = -(time/1000)+ 0.002*np.sin(2*np.pi*time)
deformation_timeseries_measured = 1e3*(deformation_timeseries_true + noise)  # conversion to mm



"""
   3. Levelling deformation model - boolean constraint -----------------------
"""

# We have 4 different deformation models and we want to fit the best combination of two
# ---> theta are boolean, their sum <=2

# we have linear, quadratic, yearly, monthly trend model
model_1 = time
model_2 = time**2
model_3 = np.sin(2*np.pi*time)
model_4 = np.sin(24*np.pi*time)
M = np.vstack((model_1,model_2,model_3,model_4)).T


# ii) invoke variables, define and solve problem

theta = cp.Variable(4)                 # Declare coefficients
z = cp.Variable(4,boolean = True)      # Declare boolean
objective = cp.norm(M@theta-deformation_timeseries_measured,p=1)  
# constraints: upper bounds on the coefficients
ub = 10
lb = -10
# constraints:  only two boolean variables nonzero 
cons = [cp.sum(z) <=2]
# constraints: if boolean variable zero, then corresponding theta also zero
cons = cons + [theta >= z*lb]
cons = cons + [theta <= z*ub]
problem = cp.Problem(cp.Minimize(objective), constraints = cons)     
problem.solve()

theta_val_boolean = theta.value  
z_val_boolean = z.value    



# iii) Visualize results

print(" The estimations lead to boolean variables z = {} and coefficients theta = {}"
      .format(z_val_boolean, theta_val_boolean))
plot_results_of_fitting_deformation(theta_val_boolean)



"""
    4. Levelling deformation model - finite set constraint -------------------
"""

# Additional information has left us with the knowledge that the first parameter
# can only take the values -0.001, -0.0015, -0.002

# i) invoke variables, define objective, constraints and solve problem

theta = cp.Variable(4)                 # Declare coefficients
z = cp.Variable(3,boolean = True)      # Declare boolean
objective = cp.norm(M@theta-deformation_timeseries_measured,p=1)  
# constraints:vector of choices for the coefficient
vec_choice = np.array([-1, -1.5, -2])
# constraints:  only one boolean variable nonzero 
cons = [cp.sum(z) <= 1]
# constraints: if boolean variable zero, then corresponding theta also zero
cons = cons + [theta[0] == z.T@vec_choice]
problem = cp.Problem(cp.Minimize(objective), constraints = cons)     
problem.solve()

theta_val_finite_set = theta.value  
z_val_finite_set = z.value    



# iii) Visualize results

print(" The estimations lead to boolean variables z = {} and coefficients theta = {}"
      .format(z_val_finite_set, theta_val_finite_set))
plot_results_of_fitting_deformation(theta_val_finite_set)



"""
    5. LS for levelling - logical constraint ----------------------------------
"""

# Additional information has left us with the (inclusive) OR condition that 
# h_1 <=0.001 or h2<=-0.001

# i) invoke variables, define objective, constraints and solve problem

theta = cp.Variable(2)     
z = cp.Variable(2, boolean = True) 
objective = cp.norm(Sigma_sqrt_inv@(A@theta-observations),p=1)      
# z[0] and z[1] are switches for activating the constraints
cons = [cp.sum(z) ==1]
# When z = 0, the constraint is active. # The big M method - if the number 
# is subtracted, constraint is fulfilled automatically
bignumber =1            
cons = cons + [theta[0] - z[0]*bignumber <=0.001]
cons = cons + [theta[1] - z[1]*bignumber <=-0.001]
problem = cp.Problem(cp.Minimize(objective), constraints = cons)      
problem.solve() 

z_val_logic = z.value
theta_val_logic = theta.value     
dh_logic = A@theta_val_logic           
error_dh_logic = dh_logic - dh_true  
residuals_logic = dh_logic - observations  


# ii) Visualize results

print(" The heights estimated with constraints (h_1<=1 OR h_2<=-1)) are {}".format(theta_val_logic))
plot_results_of_levelling_optimization(theta_val_logic)



"""
    6. Least squares with nonlinear objective --------------------------------
"""






