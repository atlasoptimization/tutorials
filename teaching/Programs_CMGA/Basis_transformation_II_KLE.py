"""
The goal of this script is to show what PCA does on a high dim example. This is
supposed to illustrate, how the components of the svd correspond to directions
of maximum variance. We also use KLE to simulate some correlated process in
the first place.
For this, do the following:
    1. Definitions and imports
    2. Generate data via KLE
    3. Do a KLE on simulated data
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



# ii) Definitions

n=100
n_simu=1

t=np.linspace(0,1,n)



"""
    2. Generate data via KLE --------------------------------------------------
"""


# i) Create covariance function

d=0.2                                       # correlation length
cov_fun=lambda s,t: np.exp(-(s-t)**2/d**2)


# ii) Fill covariance matrix

Cov_mat=np.zeros([n,n])
for k in range(n):
    for l in range(n):
        Cov_mat[k,l]=cov_fun(t[k],t[l])
        

# ii) Randomly simulate using KLE

[U,S,V]=np.linalg.svd(Cov_mat)

z=np.random.normal(0,1,n)
random_function=np.zeros([n])
for k in range(n):
    random_function=random_function+z[k]*np.sqrt(S[k])*U[:,k]
    


"""
    3.  Do a KLE on simulated data --------------------------------------------
"""


# i) Represent full function in terms of few basis functions

f_rec_1=U[:,0]*(U[:,0].T@random_function)
f_rec_3=U[:,0:3]@(U[:,0:3].T@random_function)


"""
    4. Plots and Illustrations -----------------------------------------------
"""


# i) Plot random function, basis functions, and reconstructions

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 8}

plt.rc('font', **font)

(w,h)=plt.figaspect(0.5)
f1=plt.figure(1,dpi=300,figsize=(w,h))
gs=f1.add_gridspec(2,3)

f1_ax1=f1.add_subplot(gs[0,0])
f1_ax1.plot(random_function,color='k')
plt.title('random function')
plt.xticks([])
plt.yticks([])

f1_ax2=f1.add_subplot(gs[0,1])
f1_ax2.plot(U[:,0],color='k')
plt.title('Basis function 1')
plt.xticks([])
plt.yticks([])

f1_ax3=f1.add_subplot(gs[0,2])
f1_ax3.plot(U[:,1],color='k')
plt.title('Basis function 2')
plt.xticks([])
plt.yticks([])

f1_ax4=f1.add_subplot(gs[1,0])
f1_ax4.imshow(Cov_mat)
plt.title('Cov mat')
plt.xticks([])
plt.yticks([])

f1_ax5=f1.add_subplot(gs[1,1])
f1_ax5.plot(f_rec_1,color='k')
plt.title('Reconstruction (1 element)')
plt.xticks([])
plt.yticks([])

f1_ax6=f1.add_subplot(gs[1,2])
f1_ax6.plot(f_rec_3,color='k')
plt.title('Reconstruction (3 elements)')
plt.xticks([])
plt.yticks([])
























