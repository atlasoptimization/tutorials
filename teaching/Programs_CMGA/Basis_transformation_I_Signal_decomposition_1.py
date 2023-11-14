"""
The goal of this script is to simulate a signal and decompose it into its
components. This is supported by a Fourier analysis.
For this, do the following:
    1. Definitions and imports
    2. Simulate signal
    3. Fourier Analysis
    4. Decomposition, reconstruction, interpolation
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
t=np.linspace(0,1,n)

np.random.seed(2)




"""
    2. Simulate signal -------------------------------------------------------
"""


# i) Covariance function and matrix

d= 0.1                                          # correlation length of signal
cov_fun=lambda s,t: np.exp(-((s-t)/d)**2)

cov_mat=np.zeros([n,n])
for k in range(n):
    for l in range(n):
        cov_mat[k,l]=cov_fun(t[k],t[l])


# ii) Simulate signal

sigma_noise=0.2

f_true=np.random.multivariate_normal(np.zeros([n]), cov_mat)
noise=np.random.normal(0,sigma_noise,[n])

f=f_true+noise



"""
    3. Fourier Analysis ------------------------------------------------------
"""


# i) Build fourier basis

def basis_functions(k,t):
    if k==0:
        value=1
    elif (k%2==0):
        value=np.sqrt(2)*np.cos(np.pi*k*t)
    else:
        value=np.sqrt(2)*np.sin(np.pi*(k+1)*t)
        
    return value/np.sqrt(n)


# ii) Build basis transformation matrix

F=np.zeros([n,n])
for k in range(n):
    for l in range(n):
        F[k,l]=basis_functions(k,t[l])


# iii) Representation in Fourier basis

alpha=F@f




"""
    4. Decomposition, reconstruction, interpolation --------------------------
"""


# i) Decompose by bandpass filtering

thres_1=np.round(n/10).astype(int)
thres_2=np.round(n/2).astype(int)
thres_3=n

F_1=F[0:thres_1,:]
f_1=F_1.T@F_1@f

F_2=F[thres_1:thres_2,:]
f_2=F_2.T@F_2@f

F_3=F[thres_2:thres_3,:]
f_3=F_3.T@F_3@f


# ii) One by one reconstruction

component_1=F[0:1,:].T@F[0:1,:]@f
component_2=F[1:2,:].T@F[1:2,:]@f
component_3=F[2:3,:].T@F[2:3,:]@f
component_4=F[3:4,:].T@F[3:4,:]@f
component_5=F[4:5,:].T@F[4:5,:]@f
component_6=F[5:6,:].T@F[5:6,:]@f

component_10=F[9:10,:].T@F[9:10,:]@f


# iii) Interpolation

ind_sample=np.round(np.linspace(10,n-10,5)).astype(int)
f_sample=f[ind_sample]
t_sample=t[ind_sample]

alpha_int=np.linalg.pinv(F[0:10,ind_sample]).T@f_sample
interpolation=F[0:10,:].T@alpha_int



"""
    5. Plots and Illustrations -----------------------------------------------
"""


# i) The signal

plt.figure(1,dpi=300)
plt.plot(t,f,color='k')
plt.title('The signal')
plt.xlabel('Time t')
plt.ylabel('Function value')
plt.ylim([-3,3])
plt.xticks([])
plt.yticks([])


# ii) The decomposition

(w,h)=plt.figaspect(0.3)
f1=plt.figure(2,dpi=300,figsize=(w,h))
gs1=f1.add_gridspec(1,3)

f1_ax1=f1.add_subplot(gs1[0,0])
f1_ax1.plot(t,f_1,color='k')
plt.title('Component 1')
plt.xlabel('Time t')
plt.ylim([-3,3])
plt.xticks([])
plt.yticks([])

f1_ax2=f1.add_subplot(gs1[0,1])
f1_ax2.plot(t,f_2,color='k')
plt.title('Component 2')
plt.xlabel('Time t')
plt.ylim([-3,3])
plt.xticks([])
plt.yticks([])

f1_ax3=f1.add_subplot(gs1[0,2])
f1_ax3.plot(t,f_3,color='k')
plt.title('Component 3')
plt.xlabel('Time t')
plt.ylim([-3,3])
plt.xticks([])
plt.yticks([])


# iii) The reconstruction

plt.figure(3,dpi=300)
plt.plot(t,component_1,color='k')
plt.ylim([-3,3])
plt.xticks([])
plt.yticks([])

plt.figure(4,dpi=300)
plt.plot(t,component_3,color='k')
plt.ylim([-3,3])
plt.xticks([])
plt.yticks([])

plt.figure(5,dpi=300)
plt.plot(t,component_4,color='k')
plt.ylim([-3,3])
plt.xticks([])
plt.yticks([])

plt.figure(6,dpi=300)
plt.plot(t,component_5,color='k')
plt.ylim([-3,3])
plt.xticks([])
plt.yticks([])

plt.figure(7,dpi=300)
plt.plot(t,component_10,color='k')
plt.ylim([-3,3])
plt.xticks([])
plt.yticks([])


# iv) Some covariance matrices

plt.figure(8,dpi=300)
plt.imshow(cov_mat)
plt.xticks([])
plt.yticks([])

index_choice=np.array([1,10,50])
index_mesh=np.ix_(index_choice,index_choice)
plt.figure(9,dpi=300)
plt.imshow(cov_mat[index_mesh])
plt.xticks([])
plt.yticks([])



# v) The interpolation

plt.figure(10,dpi=300)
plt.scatter(t_sample,f_sample)
plt.plot(t,interpolation,color='k')
plt.title('Interpolation of data')
plt.xlabel('Time t')
plt.ylabel('Function value')
plt.ylim([-3,3])
plt.xticks([])
plt.yticks([])




















