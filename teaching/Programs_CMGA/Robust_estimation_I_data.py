"""
The goal of this script is to simulate measurements of a linear relationship. 
The data, however are corrupted by an outlier. 
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


# i) Data 1: no outliers, no noise = nonn

alpha_0=1
alpha_1=1
data_nonn=alpha_0+alpha_1*t_sample


# ii) Data 2: yes outliers, no noise = yonn

data_yonn=data_nonn-np.eye(n_sample)[:,4]


# iii) Data 3: yes outliers, yes noise = yoyn

noise=np.random.normal(0,0.1,[n_sample])
data_yoyn=data_yonn+noise


# iv) Data 4: no outliers, yes noise = noyn

data_noyn=data_nonn+4*noise


"""
    3. Fit a function through these observations -----------------------------
"""


# i) Fit to yonn

A=np.vstack((np.ones([n_sample]),t_sample)).T
A_full=np.vstack((np.ones([n]),t)).T

alpha_yonn=cp.Variable(2)
objective_yonn=cp.Minimize(cp.norm(A@alpha_yonn-data_yonn,p=2))

problem_yonn=cp.Problem(objective_yonn)
problem_yonn.solve()




"""
    4. Plots and Illustrations -----------------------------------------------
"""


# i) Plot of yonn data with fit

plt.figure(1,dpi=300)
plt.scatter(t_sample,data_yonn,color='k')
plt.plot(t, A_full@alpha_yonn.value, color='k')
plt.ylim([0,2.5])
plt.xlim([-0.25,1.25])
plt.title('Line fit $L_2$')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.xticks([])
plt.yticks([])


# ii) Plot of yoyn data

plt.figure(2,dpi=300)
plt.scatter(t_sample,data_yoyn,color='k')
plt.ylim([0,2.5])
plt.xlim([-0.25,1.25])
plt.title('Line fit $L_2$')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.xticks([])
plt.yticks([])


# iii) Plot of yonn data

plt.figure(3,dpi=300)
plt.scatter(t_sample,data_yonn,color='k')
plt.ylim([0,2.5])
plt.xlim([-0.25,1.25])
plt.title('Dataset 1')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.xticks([])
plt.yticks([])


# iv) Plot of noyn dataset

plt.figure(4,dpi=300)
plt.scatter(t_sample,data_noyn,color='k')
plt.ylim([0,2.5])
plt.xlim([-0.25,1.25])
plt.title('Dataset 2')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.xticks([])
plt.yticks([])


# v) Plot of yonn dataset v2

plt.figure(5,dpi=300)
plt.scatter(np.delete(t_sample[0:-1],3),np.delete(data_yonn[0:-1],3),color='k')
plt.ylim([0,2.5])
plt.xlim([-0.25,1.25])
plt.title('Dataset 3')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.xticks([])
plt.yticks([])
















































