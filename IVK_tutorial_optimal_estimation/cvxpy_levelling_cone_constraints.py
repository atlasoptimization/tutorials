#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script aims to illustrate using cvxpy to solve a sequence of estimation
problems that augment classical LS with conic constraints.
For this, do the following:
    1. Imports and definitions
    2. Simulate some data
    3. Levelling deformation model - random constraints
    4. Levelling deformation model - random objective
    5. LS for levelling - unknown probability distribution 
    6. LS for levelling - do variance components analysis
    7. LS for levelling - maximum correlation
    8. LS for levelling - derive probabilistic bounds
    
    
Author: Dr. Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.ch.
Copyright: 2021-2023, Atlas optimization GmbH, Zurich, Switzerland. All rights reserved.
"""



"""
    1. Imports and definitions ------------------------------------------------
"""


# i) Imports

import numpy as np
import cvxpy as cp
from scipy.stats import norm
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

n_time = 100
time = np.linspace(0,5,n_time)

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
    
    
def plot_results_of_fitting_deformation(theta_est):
    fig_1 = plt.figure(1, figsize = [6,3], dpi = 300)
    plt.plot(time, deformation_timeseries_measured, label = 'measured deformation')
    plt.plot(time, M@theta_est, label = 'deformation model')
    plt.xlabel('Time in y')
    plt.ylabel('measured deformation in m')
    plt.title('Fit of deformation model')
    plt.legend()
    plt.show()
    
    
    
"""
    2. Simulate some data -----------------------------------------------------
"""


# i) Means and variances

# True mean is sequence of 
dh_true = np.vstack((h_1 - h_A, h_2 - h_1, h_B- h_2, h_2- h_B, h_1 - h_2, h_A - h_1)).squeeze()

sigma_I = 2e-3         # = 2 mm / sqrt(km)
sigma_true = (sigma_I**2)*np.diag([L_A1, L_12, L_2B, L_2B, L_12, L_A1])

A = np.array([[1, 0],[-1, 1], [0, -1], [0, 1], [1,-1], [-1,0]])
Sigma_sqrt_inv = np.sqrt(np.linalg.pinv(sigma_true))

observations = np.random.multivariate_normal(dh_true, sigma_true)


# ii) Simulate data for deformation timeseries
# Out model is a combination of a linear trend and a sine with yearly period

noise = np.random.normal(0,1e-3,n_time)
deformation_timeseries_true = -(time/1000)+ 0.002*np.sin(2*np.pi*time)
deformation_timeseries_measured = deformation_timeseries_true + noise



"""
    3. Levelling deformation model - random constraints -----------------------
"""

# Additional constraint is that g_1 h_1 + g_2h_2 <=-0.01 where g_1,g_2 are random
# The probability of constraint violation shall be smaller than 5%

# i) Define mean and covariance and draw a g randomly
mu_g = np.ones([2])
sigma_g = 0.1*np.eye(2)
sigma_g_sqrt = np.sqrt(sigma_g)   # Warning - only works for diagonal matrices

g = np.random.multivariate_normal(mu_g, sigma_g)


# ii) invoke variables, define objective, constraints and solve problem

theta = cp.Variable(2)      
objective = cp.norm(Sigma_sqrt_inv@(A@theta-observations),p=2)
# constraint: prob(g^Ttheta<=0) >=95%
#             => mu_g^Ttheta + phi^{-1}(0.95) cp.norm(sigma_g_sqrt @ theta) <=0       
cons = [mu_g.T@theta + norm.ppf(0.95) * cp.norm(sigma_g_sqrt @ theta, p=2) <=-0.01]
problem = cp.Problem(cp.Minimize(objective), constraints = cons)
problem.solve()         

theta_val_LS_probcons = theta.value          
dh_LS_probcons = A@theta_val_LS_probcons             
error_dh_LS_probcons = dh_LS_probcons - dh_true       
residuals_LS_probcons = dh_LS_probcons - observations


# iii) Visualize results using plots and a simulation run to validate 
# adherence to probabilistic constraint empirically

print(" The heights estimated with constrained (p(g_1h_1 + g_2h_2 <=-0.01) >=95%) LS are {}".format(theta_val_LS_probcons))
plot_results_of_levelling_optimization(theta_val_LS_probcons)

n_simu = 1000
g_mat = np.zeros([2,n_simu])
for k in range(n_simu):
    g_mat[:,k] = np.random.multivariate_normal(mu_g, sigma_g)
constraint_value = g_mat.T@theta_val_LS_probcons

success_percentage = np.sum(constraint_value<=-0.01)/n_simu
failure_percentage = 1 - success_percentage
print("Empirically, the constraint is violated in {}% of cases based on 1000 simulations."
      .format(100*failure_percentage))



"""
    4. Levelling deformation model - random objective -------------------------
"""


# Suppose that the design matrix is random and unknown before the actual parameters
# theta have to be fixed. Then the theta need to be chosen in such a way that
# the expected value of the objective is minimized.

# i) Define basic model parameters
model_1 = time
model_2 = time**2
model_3 = np.sin(2*np.pi*time)
model_4 = np.sin(24*np.pi*time)
M = np.vstack((model_1,model_2,model_3,model_4)).T

# M is now random
M_mu = M
sigma_M = 1    # add noise with sigma_M to each entry of M_mu
P = n_time*sigma_M**2*np.eye(4) #= E U^TU


# ii) invoke variables, define and solve problem

theta = cp.Variable(4)                 # Declare coefficients
objective = cp.sum_squares(M_mu@theta-deformation_timeseries_measured)  + cp.quad_form(theta,P)
problem = cp.Problem(cp.Minimize(objective))     
problem.solve()

theta_val_random_obj = theta.value  
# Compare to what happens when just the mean of A is taken in the optimization problem
theta_val_no_stoch = np.array([-1.29257087e-03,  8.09332016e-05,  2.23754226e-03,  7.88765353e-05])
M_no_stoch = M_mu@theta_val_no_stoch

# Compare this to just operating on the mean.
n_simu = 100
error_norm = np.zeros(n_simu)
error_norm_no_stoch = np.zeros(n_simu)
M_rand = np.zeros([M.shape[0], M.shape[1],n_simu])
for k in range(n_simu):
    M_temp = M_mu + np.random.normal(0,sigma_M, M.shape)
    M_rand[:,:,k] = M_temp
    error_norm[k] = np.linalg.norm(M_temp@theta_val_random_obj - deformation_timeseries_measured)
    error_norm_no_stoch[k] = np.linalg.norm(M_temp@theta_val_no_stoch - deformation_timeseries_measured)

print(("The average residual norm of the best fit taking into account the stochasticity "
      " of the problem amounts to {}. Without stochasticity, it is {}")
      .format(np.mean(error_norm), np.mean(error_norm_no_stoch)))
print("The fit minimizing expected value of the objective function:")
plot_results_of_fitting_deformation(theta_val_random_obj)
print("The fit based on the mean of all involved random quantities:")
plot_results_of_fitting_deformation(theta_val_no_stoch)



"""
    5. LS for levelling - unknown probability distribution --------------------
"""

# Suppose we do not know the probability distribution of dh_{12} but we have
# observed it quite a few times and have some theoretical information that allow us
# to formulate some bounds


# i) Invoke variables & define objective

n_dim_pvec = 100
p_vec = cp.Variable(n_dim_pvec) 
# constraints on expected value, maximally achievable dh_12 and negativity
# 0.4mm <= E[dh_12] <=0.5mm
# -1mm <= dh_12 <= 1mm
# p(dh_12 <= 0) <= 20%  
x_vals = np.linspace(-1.5,1.5,n_dim_pvec)
x_neg = x_vals <=0
x_less_m1 = x_vals <=-1
x_more_p1 = x_vals >=1


 # ii) Define constraints and solve problem
cons = [p_vec>=0]
cons = cons + [np.ones(n_dim_pvec).T@p_vec == 1]
cons = cons +[x_vals.T@p_vec >=0.4]
cons = cons +[x_vals.T@p_vec <=0.5]
cons = cons +[p_vec[x_less_m1] ==0]
cons = cons +[p_vec[x_more_p1] ==0]
cons = cons +[x_neg.T@p_vec <=0.2]

# Rest of the optimization problem
objective = cp.sum(cp.entr(theta))
problem = cp.Problem(cp.Maximize(objective), constraints = cons)
problem.solve(solver = 'SCS', verbose = True)         
p_vec_entropy = p_vec.value          


# iii) Visualize the maximum entropy solution

fig_1 = plt.figure(1, figsize = [6,3], dpi = 300)
plt.plot(x_vals, p_vec_entropy, label = 'Max ent')
plt.xlabel('X values')
plt.ylabel('Probability of occurrence')
plt.title('Maximum entropy distribution')
plt.legend()
plt.show()

print(("The sum of all probabilities is {}, the probability mass for negative values is"
      " {}, and the expected value is {}").format(p_vec_entropy.T@np.ones(n_dim_pvec), 
                                p_vec_entropy.T@x_neg, p_vec_entropy.T@x_vals))



"""
    6. LS for levelling - do variance components analysis ----------------------
"""


# We want to build a very simple model for a covariance matrix and do inference
# on the components. We do this using information aggregated into an empirical
# covariance matrix

# i) Construct empirical covariance matrix

n_simu = 10
observation_matrix = np.zeros([6,n_simu])
for k in range(n_simu):
    observation_matrix[:,k] = np.random.multivariate_normal(dh_true, sigma_true) - dh_true
emp_cov = (1/n_simu)*observation_matrix@observation_matrix.T


# ii) invoke variables, define objective, constraints and solve problem

theta = cp.Variable([1])
C_base = np.diag([L_A1, L_12, L_2B, L_2B, L_12, L_A1])
objective = cp.norm(emp_cov - theta*C_base)
cons = [theta*C_base>>0]        # >> 0 means positive semidefinite

problem = cp.Problem(cp.Minimize(objective), constraints = cons)
problem.solve()
sigma_I_est = np.sqrt(theta.value)
sigma_est = sigma_I_est*C_base


# iii) Visualize results using plots and a simulation run to validate 
# adherence to probabilistic constraint empirically

fig_1 = plt.figure(1,figsize = [5,5], dpi = 300)
plt.title('Empirical covariance matrix')
tickpositions = np.linspace(0,6,6)
ticks = ["dh_A1" , "dh_12", " dh_2B", "dh_B2", "dh_21", "dh_1A"]
plt.xticks(tickpositions, ticks)
plt.yticks(tickpositions, ticks)
plt.imshow(emp_cov)

fig_2 = plt.figure(1,figsize = [5,5], dpi = 300)
plt.title('Fitted covariance matrix')
tickpositions = np.linspace(0,6,6)
ticks = ["dh_A1" , "dh_12", " dh_2B", "dh_B2", "dh_21", "dh_1A"]
plt.xticks(tickpositions, ticks)
plt.yticks(tickpositions, ticks)
plt.imshow(sigma_est)

print(("This is actually not so impressive. The positive semidefiniteness constraint"
       " does very little here, since sigma_I would have been estimated to be positive"
       " anyway. The whole thing becomes way more interesting thorugh, if we do "
       " proper optimization through the convex cone of psd matrices. BTW: "
       "Right now it is estimated to be {} while the true value is {}.").format(sigma_I_est, sigma_I))



"""
    7. LS for levelling - maximum correlation ---------------------------------
"""


# We want to find out the biggest correlations we could possibly have between
# the two variables dh_A1 and dh_2B given some model for the covariance matrix.
# The model contains a way to construct a covariance matrix + some bounds on entries.


# i) invoke variables, define objective, constraints and solve problem

theta = cp.Variable([4])
C_0 = np.diag([L_A1, L_12, L_2B])
C_1 = np.array([[0,1,0], [1,0,0], [0,0,0]])
C_2 = np.array([[0,0,0], [0,0,1], [0,1,0]])
C_3 = np.array([[0,0,1], [0,0,0], [1,0,0]])
C_mat_full = theta[0]*C_0+ theta[1]*C_1 + theta[2]*C_2 + theta[3]*C_3

objective = C_mat_full[0,2]
cons = [C_mat_full >> 0]        # >> 0 means positive semidefinite
cons = cons + [theta[0] == 1]
cons = cons + [theta[1] >=0.3]
cons = cons + [theta[1] <=0.5]
cons = cons + [theta[1] + theta[2] >=0.1]
cons = cons + [theta[1] + theta[2] <=0.5]
cons = cons + [theta[2] + theta[3] >=0.1]
cons = cons + [theta[2] + theta[3] <=0.5]

problem = cp.Problem(cp.Maximize(objective), constraints = cons)
problem.solve(verbose = True)
Full_mat_optimized = C_mat_full.value
max_cov = Full_mat_optimized[0,2]


# iii) Visualize results using plots and a simulation run to validate 
# adherence to probabilistic constraint empirically

fig_1 = plt.figure(1,figsize = [5,5], dpi = 300)
plt.title('Covariance matrix/sigma_I optimized for correlations')
tickpositions = np.linspace(0,2,3)
ticks = ["dh_A1" , "dh_12", " dh_2B"]
plt.xticks(tickpositions, ticks)
plt.yticks(tickpositions, ticks)
plt.imshow(Full_mat_optimized)



print(("The maximum covariance between dh_A1 and dh_2B is {}.").format(max_cov))





"""
    8. LS for levelling - derive probabilistic bounds ------------------------
"""




# We fand to find out the probability that [h_1, h_2] lie in [-1,1]^2. Suppose
# we have mean and variance of h_1 and h_2 given the observations.
# Then we can estimate P([h_1,h_2] in [-1,1]^2) by optimizing over all probability
# distributions having this specific mean and variance and bound that probability
# from below.

# i) Set up basic quantities
mm_unit = 1
mu_h = np.zeros([2])
std_h = 1*mm_unit
sigma_hh = std_h**2*np.eye(2)

# a_i, b_i for determining the polyhedron [-1,1]^2. P = set s.t. a_k^Tz<=b
n_cons = 4
a_1 = np.array([1,0])
a_2 = np.array([-1,0])
a_3 = np.array([0,1])
a_4 = np.array([0,-1])
A_c = np.vstack((a_1,a_2,a_3,a_4))

b = 2*mm_unit


# ii) Invoke variables, constraints, and solve the problem

P = cp.Variable(shape = [2,2], PSD=True)
q = cp.Variable(shape = 2)
r = cp.Variable(shape = 1, nonneg = True)
tau = cp.Variable(shape = 4, nonneg = True)

objective = cp.trace(sigma_hh@P) + 2*q.T@mu_h +r

# Original semidefiniteness constraint
cons= [cp.bmat([[P,cp.reshape(q,[2,1])],
                [cp.reshape(q,[1,2]),cp.reshape(r,[1,1])]])>>0]
# Additional bounding constraints
for k in range(n_cons):
    LMI_mat_1=cp.bmat([[P,cp.reshape(q,[2,1])],[cp.reshape(q,[1,2]),cp.reshape(r-1,[1,1])]])
    LMI_mat_2=tau[k]*cp.bmat([[np.zeros([2,2]), 1/2*A_c[k,:].reshape([2,1])],[1/2*A_c[k,:].reshape([1,2]), -np.reshape(b,[1,1])]])    
    cons=cons+[LMI_mat_1>>LMI_mat_2]

problem = cp.Problem(cp.Minimize(objective), constraints = cons)
problem.solve(solver = 'SCS')


# iii) Interpret, visualize, and check

P_opt = P.value
q_opt = q.value
r_opt = r.value

value_of_problem = np.trace(sigma_hh@P_opt) + 2*q_opt.T@mu_h +r_opt
probability = 1-value_of_problem

n_simu = 1000
h_simu = np.zeros([2,n_simu])
for k in range(n_simu):
    h_simu[:,k] = np.random.multivariate_normal(mu_h,sigma_hh)

h_1_violation = (h_simu[0,:]>=b).astype(int) +( h_simu[0,:]<=-b).astype(int)
h_2_violation = (h_simu[1,:]>=b).astype(int) +( h_simu[1,:]<=-b).astype(int)

total_violation = np.sum((h_1_violation + h_2_violation) >=1)
total_valid = n_simu - total_violation

print(("The probability that [h_1,h_2] lie in the box [-1,1]^2 mm is at least"
      " {} %. The empirical probability of lying in the box amouts to {}% based on 1000"
      " gaussian simulations").format(probability*100, 100*total_valid/n_simu))













    


