#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script aims to illustrate using cvxpy to solve a sequence of estimation
problems that augment classical LS with linear equality and inequality constraints.
For this, do the following:
    1. Imports and definitions
    2. Simulate some data
    3. LS for levelling
    4. LS for levelling with linear equality 1
    5. LS for levelling with linear equality 2
    6. LS for levelling with linear inequality 1
    7. LS for levelling with linear inequality 2
    
Author: Dr. Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.ch.
Copyright: 2021-2023, Atlas optimization GmbH, Zurich, Switzerland. All rights reserved.
"""



"""
    1. Imports and definitions ------------------------------------------------
"""


# i) Imports

import numpy as np
import cvxpy as cp
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


# iii) Plotting function

def plot_results_of_levelling_optimization(theta_est):
    fig_1 = plt.figure(1, figsize = [6,3], dpi = 300)
    x_tickpostions = np.linspace(0,6,6)
    x_ticks = ["dh_A1" , "dh_12", " dh_2B", "dh_B2", "dh_21", "dh_1A"]
    plt.xticks(x_tickpostions, x_ticks)
    plt.plot(x_tickpostions, 1e+3*(A@theta_est), label = 'estimated')
    plt.plot(x_tickpostions, 1e+3*(dh_true), label = 'true')
    plt.xlabel('height difference at subsection')
    plt.ylabel('value of height difference [mm]')
    plt.title('Estimated height differences vs true height differences')
    plt.legend()
    plt.show()
    
    fig_2 = plt.figure(2, figsize = [6,3], dpi = 300)
    x_tickpostions = np.linspace(0,6,6)
    x_ticks = ["dh_A1" , "dh_12", " dh_2B", "dh_B2", "dh_21", "dh_1A"]
    plt.xticks(x_tickpostions, x_ticks)
    plt.bar(x_tickpostions, 1e+3*(A@theta_est - dh_true), label = 'errors')
    plt.xlabel('height difference at subsection')
    plt.ylabel('value of height difference [mm]')
    plt.title('Errors of height differences')
    plt.legend()
    plt.show()
    
    fig_3 = plt.figure(3, figsize = [6,3], dpi = 300)
    x_tickpostions = np.linspace(0,6,6)
    x_ticks = ["dh_A1" , "dh_12", " dh_2B", "dh_B2", "dh_21", "dh_1A"]
    plt.xticks(x_tickpostions, x_ticks)
    plt.bar(x_tickpostions, 1e+3*(A@theta_est - observations), label = 'residuals')
    plt.xlabel('height difference at subsection')
    plt.ylabel('value of height difference [mm]')
    plt.title('Residuals of height differences')
    plt.legend()
    plt.show()



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
    3. LS for levelling -------------------------------------------------------
"""


# i) Invoke basic quantities

A = np.array([[1, 0],[-1, 1], [0, -1], [0, 1], [1,-1], [-1,0]])
Sigma_sqrt_inv = np.sqrt(np.linalg.pinv(sigma_true))


# ii) invoke variables, define and solve problem

theta = cp.Variable(2)      # Define variable theta with dimension 2
objective = cp.norm(Sigma_sqrt_inv@(A@theta-observations),p=2)       # Define objective function, the 2-norm of residuals
problem = cp.Problem(cp.Minimize(objective))        # Define problem as Minimization of objective function
problem.solve(verbose = True)         # Solve the problem

theta_val_LS = theta.value          # Access the optimal value of the variables
dh_LS = A@theta_val_LS              # Calculate best guesses for height differences
error_dh_LS = dh_LS - dh_true       # Calculate the error of height differences
residuals_LS = dh_LS - observations     # Calculate the residuals of height differences


# iii) Visualize results

print(" The heights estimated with unconstrained LS are {}".format(theta_val_LS))
plot_results_of_levelling_optimization(theta_val_LS)



"""
    4. LS for levelling with linear equality 1 ----------------------------------
"""

# Additional information has left us with the equality h_1 = 0m

# i) invoke variables, define objective, constraints and solve problem

theta = cp.Variable(2)      
objective = cp.norm(Sigma_sqrt_inv@(A@theta-observations),p=2)       
cons = [theta[0] == 0]                               # Define constraint as h_1 = 0
problem = cp.Problem(cp.Minimize(objective), constraints = cons)
problem.solve()         

theta_val_LS_eq_1 = theta.value          
dh_LS_eq_1 = A@theta_val_LS_eq_1             
error_dh_LS_eq_1 = dh_LS_eq_1 - dh_true       
residuals_LS_eq_1 = dh_LS_eq_1 - observations


# ii) Visualize results

print(" The heights estimated with constrained (h_1=0) LS are {}".format(theta_val_LS_eq_1))
plot_results_of_levelling_optimization(theta_val_LS_eq_1)



"""
    5. LS for levelling with linear equality 2 ----------------------------------
"""

# Additional information has left us with the equality dh_A1 = dh_12
# ---> (h_1-h_A) = (h_2-h_1)

# i) invoke variables, define objective, constraints and solve problem

theta = cp.Variable(2)      
objective = cp.norm(Sigma_sqrt_inv@(A@theta-observations),p=2)      
cons = [(theta[0] - h_A) == (theta[1] - theta[0])]          # Define constraint as dh_A1 = dh_12               
problem = cp.Problem(cp.Minimize(objective), constraints = cons)      
problem.solve() 

theta_val_LS_eq_2 = theta.value     
dh_LS_eq_2 = A@theta_val_LS_eq_2           
error_dh_LS_eq_2 = dh_LS_eq_2 - dh_true  
residuals_LS_eq_2 = dh_LS_eq_2 - observations  


# ii) Visualize results

print(" The heights estimated with constrained (dh_A1 = dh_12) LS are {}".format(theta_val_LS_eq_2))
print(" The value of dh_A1 and  dh_12 are {} and {} respecitvely"
      .format((theta_val_LS_eq_2[0] - h_A), (theta_val_LS_eq_2[1] - theta_val_LS_eq_2[0])))
plot_results_of_levelling_optimization(theta_val_LS_eq_2)



"""
    6. LS for levelling with linear inequality 1 --------------------------------
"""


# Additional information has left us with the knowledge that the average height is <=0
# ---> 0.5(theta[0] + theta[1]) <=0

# i) invoke variables, define objective, constraints and solve problem

theta = cp.Variable(2)      
objective = cp.norm(Sigma_sqrt_inv@(A@theta-observations),p=2)      
cons = [theta[0] + theta[1] <=0]          # Define constraint as 0.5(h_1+h_2) <=0              
problem = cp.Problem(cp.Minimize(objective), constraints = cons)      
problem.solve() 

theta_val_LS_ineq_1 = theta.value     
dh_LS_ineq_1 = A@theta_val_LS_ineq_1           
error_dh_LS_ineq_1 = dh_LS_ineq_1 - dh_true  
residuals_LS_ineq_1 = dh_LS_ineq_1 - observations  


# ii) Visualize results

print(" The heights estimated with constrained (h_1+h_2 <=0) LS are {}".format(theta_val_LS_ineq_1))
print(" The value of h_1+h_2 is {}".format(theta_val_LS_ineq_1[0]+theta_val_LS_ineq_1[1]))
plot_results_of_levelling_optimization(theta_val_LS_ineq_1)






"""
    7. LS for levelling with linear inequality 2 -----------------------------
"""


# We want that some of our residuals are positive so that we are consistently
# overestimating the height differences there ---> A@theta - observations >=0
# It might be that the problem is infeasible - to avoid this, we only demand
# positivity for the first two residuals.

# i) invoke variables, define objective, constraints and solve problem

theta = cp.Variable(2)      
objective = cp.norm(Sigma_sqrt_inv@(A@theta-observations),p=2)      
cons = [A[0:2,:]@theta - observations[0:2] >= 0]          # Define constraint A@theta - y >=0              
problem = cp.Problem(cp.Minimize(objective), constraints = cons)      
problem.solve() 

theta_val_LS_ineq_2 = theta.value     
dh_LS_ineq_2 = A@theta_val_LS_ineq_2           
error_dh_LS_ineq_2 = dh_LS_ineq_2 - dh_true  
residuals_LS_ineq_2 = dh_LS_ineq_2 - observations  


# ii) Visualize results

print(" The heights estimated with constrained (resid>=0) LS are {}".format(theta_val_LS_ineq_2))
print(" The value of the residuals is {}".format(A@theta_val_LS_ineq_2 - observations))
plot_results_of_levelling_optimization(theta_val_LS_ineq_2)





