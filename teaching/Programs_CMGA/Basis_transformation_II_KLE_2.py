"""
The goal of this script is to show what PCA does on a high dim example. This is
supposed to illustrate, how the KLE can be used for interpolation, compression, 
and simulation.
For this, do the following:
    1. Definitions and imports
    2. Simulate data
    3. Do KLE on simulated data
    4. Interpolation, compression, simulation
    5. Plots and Illustrations
    
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
n_simu=100

t=np.linspace(0,1,n)



"""
    2. Simulate data ----------------------------------------------------------
"""


# i) Create covariance function

d=0.2                                       # random walk
cov_fun=lambda s,t: np.min([s,t])


# ii) Fill covariance matrix

Cov_mat=np.zeros([n,n])
for k in range(n):
    for l in range(n):
        Cov_mat[k,l]=cov_fun(t[k],t[l])
        

# ii) Simulate
    
data=np.zeros([n,n_simu])
for k in range(n_simu):
    data[:,k]=np.random.multivariate_normal(np.zeros([n]),Cov_mat)


"""
    3.  Do KLE on simulated data -----------------------------------------------
"""


# i) Decompose empirical covariance matrix

Emp_cov=(1/n)*data@data.T
[U,S,V]=np.linalg.svd(Emp_cov)





"""
    4. Interpolation, compression, simulation -------------------------------
"""

# i) Interpolate

ind_sample=np.round(np.linspace(10,n-10,5)).astype(int)
f_sample=data[ind_sample,0]
t_sample=t[ind_sample]

alpha_int=np.linalg.pinv(U[ind_sample,0:5])@f_sample
interpolation=U[:,0:5]@alpha_int

# ii) Compress

coeffs=U.T@data[:,0]


# iii) Simulate

z=np.random.normal(0,1,n)
random_function=np.zeros([n])
for k in range(n):
    random_function=random_function+z[k]*np.sqrt(S[k])*U[:,k]


"""
    5. Plots and Illustrations -----------------------------------------------
"""


# i) Plot random function, basis functions, and reconstructions

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 8}

plt.rc('font', **font)

(w,h)=plt.figaspect(0.5)
f1=plt.figure(1,dpi=300,figsize=(w,h))
gs=f1.add_gridspec(2,4)

f1_ax1=f1.add_subplot(gs[0,0])
f1_ax1.plot(data,color='k')
plt.title('Observations')
plt.xticks([])
plt.yticks([])

f1_ax2=f1.add_subplot(gs[0,1])
f1_ax2.imshow(Emp_cov)
plt.title('Empirical covariance')
plt.xticks([])
plt.yticks([])

f1_ax3=f1.add_subplot(gs[0,2])
f1_ax3.plot(U[:,0],color='k')
plt.title('Basis function 1')
plt.xticks([])
plt.yticks([])

f1_ax4=f1.add_subplot(gs[0,3])
f1_ax4.plot(U[:,1],color='k')
plt.title('Basis function 2')
plt.xticks([])
plt.yticks([])


f1_ax5=f1.add_subplot(gs[1,1])
f1_ax5.scatter(t_sample,f_sample,color='k')
f1_ax5.plot(t,interpolation,color='k')
plt.title('Interpolation')
plt.xticks([])
plt.yticks([])

f1_ax6=f1.add_subplot(gs[1,2])
markerline, stemlines, baseline=plt.stem(coeffs,linefmt='k')
markerline.set_markerfacecolor('k')
plt.title('Compression')
plt.xticks([])
plt.yticks([])

f1_ax7=f1.add_subplot(gs[1,3])
f1_ax7.plot(random_function,color='k')
plt.title('Simulation')
plt.xticks([])
plt.yticks([])
























