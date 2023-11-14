"""
The goal of this script is to show what PCA does on a low dim example. This is
supposed to illustrate, how the components of the svd correspond to directions
of maximum variance.
For this, do the following:
    1. Definitions and imports
    2. Generate data
    3. Do PCA
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
n_dim=2



"""
    2. Generate data ---------------------------------------------------------
"""


# i) Create covariance matrix of data

Cov_mat=np.array([[1.1, 0.9],[0.9,1.1]])



# ii) Generate data randomly

#np.random.seed(0)
data=np.zeros([2,n])
for k in range(n):
    data[:,k]=np.random.multivariate_normal(np.zeros([2]),Cov_mat)




"""
    3. Do PCA -----------------------------------------------------------------
"""


# i) Do PCA and contract

Emp_cov=(1/n)*data@data.T

[U,S,V]=np.linalg.svd(Emp_cov,hermitian=True)





"""
    4. Plots and Illustrations -----------------------------------------------
"""


# i) Plot the data and svd

(w,h)=plt.figaspect(0.5)
f1=plt.figure(1,dpi=300,figsize=(w,h))
gs=f1.add_gridspec(1,2)

f1_ax1=f1.add_subplot(gs[0,0])
f1_ax1.scatter(data[0,:],data[1,:],color='k')
plt.title('Datapoints')
plt.xticks([])
plt.yticks([])

f1_ax2=f1.add_subplot(gs[0,1])
f1_ax2.imshow(Emp_cov,vmin=0,vmax=1)
plt.title('Empirical Cov mat')
plt.xticks([])
plt.yticks([])

print('The eigenvectors are')
print(U[:,0],U[:,1])

print('The eigenvalues are')
print(S[:])

















