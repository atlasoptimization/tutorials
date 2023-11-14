"""
The goal of this script is to simulate measurements of a linear relationship. 
The data, however are corrupted by an outlier. Then l1, l2, linfty norm optimal
fitting is applied.
For this, do the following:
    1. Definitions and imports
    2. Simulate different sets of observations
    3. Fit a function through these observations
    4. Plots and Illustrations
    
The script is meant solely for educational and illustrative purposes. Written by
Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.ch.

"""


"""
    1. Definitions and imports -----------------------------------------------
"""


# i) Imports

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp


# ii) Definitions

n=100
t=np.linspace(-0.25,1.25,n)

n_sample=6
t_sample=np.linspace(0,1,n_sample)

np.random.seed(2)


"""
    2. Simulate different sets of observations -------------------------------
"""


# i) Data : Outliers but no noise

alpha_0=1
alpha_1=1
data_true=alpha_0+alpha_1*t_sample
data=data_true-np.eye(n_sample)[:,4]



"""
    3. Fit a function through these observations -----------------------------
"""


# i) Fit by minimizing l1 norm

A=np.vstack((np.ones([n_sample]),t_sample)).T
A_full=np.vstack((np.ones([n]),t)).T

alpha_l1=cp.Variable(2)
objective_l1=cp.Minimize(cp.norm(A@alpha_l1-data,p=1))

problem_l1=cp.Problem(objective_l1)
problem_l1.solve()


# ii) Fit by minimizing l2 norm

alpha_l2=cp.Variable(2)
objective_l2=cp.Minimize(cp.norm(A@alpha_l2-data,p=2))

problem_l2=cp.Problem(objective_l2)
problem_l2.solve()


# i) Fit by minimizing linfty norm

alpha_li=cp.Variable(2)
objective_li=cp.Minimize(cp.norm(A@alpha_li-data,p='inf'))

problem_li=cp.Problem(objective_li)
problem_li.solve()




"""
    4. Plots and Illustrations -----------------------------------------------
"""


# i) Plot the L1 norm fit

plt.figure(1,dpi=300)
plt.scatter(t_sample,data,color='k')
plt.plot(t, A_full@alpha_l1.value, color='k')
plt.ylim([0,2.5])
plt.xlim([-0.25,1.25])
plt.title('Line fit $L_1$')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.xticks([])
plt.yticks([])


# ii) Plot the L2 norm fit

plt.figure(2,dpi=300)
plt.scatter(t_sample,data,color='k')
plt.plot(t, A_full@alpha_l2.value, color='k')
plt.ylim([0,2.5])
plt.xlim([-0.25,1.25])
plt.title('Line fit $L_2$')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.xticks([])
plt.yticks([])


# ii) Plot the Linfty norm fit

plt.figure(3,dpi=300)
plt.scatter(t_sample,data,color='k')
plt.plot(t, A_full@alpha_li.value, color='k')
plt.ylim([0,2.5])
plt.xlim([-0.25,1.25])
plt.title('Line fit $L_{\infty}$')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.xticks([])
plt.yticks([])











































