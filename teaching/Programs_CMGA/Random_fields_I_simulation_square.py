"""
The goal of this script is to simulate some random and deterministic fields 
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
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits import mplot3d


# ii) Definitions

n_x = 40
n_y = 40
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

Cov_mat = np.zeros([n_total,n_total])

for k in range(n_total):
    for l in range(n_total):
        loc_1_temp = np.array([xx.flatten()[k], yy.flatten()[k]])
        loc_2_temp = np.array([xx.flatten()[l], yy.flatten()[l]])
        Cov_mat[k,l] = cov_fun(loc_1_temp, loc_2_temp)


# ii) Simulation

f_simu = np.zeros([n_simu, n_total])

for k in range(n_simu):
    f_simu[k,:] = np.random.multivariate_normal(np.zeros(n_total), Cov_mat)

f_simu_reshaped = np.reshape(f_simu,[n_simu, n_y,n_x], order ='C')



"""
    3. Plots and Illustrations -----------------------------------------------
"""


# i) Plot surface plot

fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, dpi=300, figsize=(6,6))

surf = ax.plot_surface(xx, yy, f_simu_reshaped[0,:,:], cmap=cm.viridis,
                       linewidth=0, antialiased=False)


plt.xlabel('x axis')
plt.ylabel('y axis')
plt.xticks([])
plt.yticks([])
ax.set_zticks([])
ax.view_init(-120, -30)
plt.show()


# ii) Plot some simple images

plt.figure(1,dpi=300)
plt.imshow(f_simu_reshaped[0,:,:])
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.xticks([])
plt.yticks([])


# iii) Means and covariances

true_mean = np.zeros([n_y,n_x])
true_cov = Cov_mat

empirical_mean = np.mean(f_simu_reshaped,0)
empirical_cov = np.zeros([n_total,n_total])

for k in range(n_simu):
    empirical_cov = empirical_cov + np.reshape(f_simu_reshaped[k,:,:].flatten(),[n_total,1])@np.reshape(f_simu_reshaped[k,:,:].flatten(),[n_total,1]).T
    
empirical_cov = (1/n_simu)*empirical_cov



plt.figure(2,dpi=300)
plt.imshow(true_mean)
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.clim(-2,2)
plt.xticks([])
plt.yticks([])


plt.figure(3,dpi=300)
plt.imshow(true_cov)
plt.xlabel('point nr')
plt.ylabel('point nr')
plt.xticks([])
plt.yticks([])



plt.figure(4,dpi=300)
plt.imshow(empirical_mean)
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.clim(-2,2)
plt.xticks([])
plt.yticks([])


plt.figure(5,dpi=300)
plt.imshow(empirical_cov)
plt.xlabel('point nr')
plt.ylabel('point nr')
plt.xticks([])
plt.yticks([])












