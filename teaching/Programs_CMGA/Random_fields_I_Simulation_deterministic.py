"""
The goal of this script is to simulate some radially random and deterministic fields 
for visualization purposes. 
For this, do the following:
    1. Definitions and imports
    2. Simulation
    3. Plots and Illustrations
    
The script is meant solely for educational and illustrative purposes. Written by
Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.ch.

"""

"""
    1. Definitions and imports -----------------------------------------------
"""


# i) Imports

import numpy as np
import matplotlib.pyplot as plt



# ii) Definitions

n_r = 300
n_x = 40
n_y = 40

r = np.linspace(0,1.5,n_r)
x = np.linspace(-1,1,n_x)
y = np.linspace(-1,1,n_y)

xx,yy = np.meshgrid(x,y)

n_total = n_x*n_y

n_simu=5

np.random.seed(1)


"""
    2. Simulation ------------------------------------------------------------
"""

# i) Covariance matrix
d=0.3
cov_fun = lambda x,y: np.exp(-(np.linalg.norm(x-y)/d)**2)

Cov_mat = np.zeros([n_r,n_r])

for k in range(n_r):
    for l in range(n_r):
        Cov_mat[k,l] = cov_fun(r[k],r[l])


# ii) Simulation radially symmetric

f_simu = np.zeros([n_simu, n_r])

for k in range(n_simu):
    f_simu[k,:] = np.random.multivariate_normal(np.zeros(n_r), Cov_mat)


# iii) deterministic function

fun_det = lambda x,y: -(0.5*x*x -(x*y)**2+0.5*y*y)
f_deterministic = np.zeros([n_y,n_x])
for k in range(n_y):
    for l in range(n_x):
        f_deterministic[k,l] = fun_det(y[k],x[l])


# iv) create radially symmetric image

rr = np.sqrt(xx**2+yy**2)
f_radial = np.zeros([n_y,n_x])

for k in range(n_y):
    for l in range(n_x):
        temp_range = rr[k,l]
        index = np.abs(temp_range-r).argmin()
        f_radial[k,l] = f_simu[0,index]



"""
    3. Plots and Illustrations -----------------------------------------------
"""



# i) Plot some simple images

plt.figure(1,dpi=300)
plt.imshow(f_deterministic)
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.xticks([])
plt.yticks([])


# ii) radial function

plt.figure(2,dpi=300)
plt.imshow(f_radial)
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.xticks([])
plt.yticks([])







