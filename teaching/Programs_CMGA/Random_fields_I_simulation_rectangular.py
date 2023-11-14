"""
The goal of this script is to simulate some random fields for visualization purposes. 
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



# ii) Definitions

n_x = 50
n_y = 20
x = np.linspace(-1,1,n_x)
y = np.linspace(-1,1,n_y)

xx,yy = np.meshgrid(x,y)

n_total = n_x*n_y

n_simu=5



"""
    2. Simulation ------------------------------------------------------------
"""

# i) Covariance matrix
d=1
cov_fun = lambda x,y: np.exp(-(np.linalg.norm(x-y)/d)**1)

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

f_simu = np.reshape(f_simu,[n_simu, n_y,n_x], order ='C')



"""
    3. Plots and Illustrations -----------------------------------------------
"""


# i) Plot surface plot

fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, dpi=300, figsize=(3,6))

surf = ax.plot_surface(xx, yy, f_simu[0,:,:], cmap=cm.viridis,
                       linewidth=0, antialiased=False)


plt.xlabel('x axis')
plt.ylabel('y axis')
plt.xticks([])
plt.yticks([])
ax.set_zticks([])
ax.view_init(-100, -5)
plt.show()


# ii) Plot some simple images

plt.figure(1,dpi=300)
plt.imshow(f_simu[0,:,:])
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.xticks([])
plt.yticks([])
























