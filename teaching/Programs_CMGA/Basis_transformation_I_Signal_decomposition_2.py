"""
The goal of this script is to simulate a messy signal and decompose it into its
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

n=300
t=np.linspace(0,1,n)

np.random.seed(2)




"""
    2. Simulate signal -------------------------------------------------------
"""


# i) Covariance function and matrix

d_1= 0.2                                          # correlation length of signal 1
d_2= 0.05                                          # correlation length of signal 2

cov_fun_smooth_1=lambda s,t: np.exp(-((s-t)/d_1)**2)
cov_fun_smooth_2=lambda s,t: np.exp(-((s-t)/d_2)**2)
cov_fun_ragged=lambda s,t: np.exp(-(np.abs(s-t)/0.2))

cov_mat_smooth_1=np.zeros([n,n])
for k in range(n):
    for l in range(n):
        cov_mat_smooth_1[k,l]=cov_fun_smooth_1(t[k],t[l])
        
cov_mat_smooth_2=np.zeros([n,n])
for k in range(n):
    for l in range(n):
        cov_mat_smooth_2[k,l]=cov_fun_smooth_2(t[k],t[l])
        
cov_mat_ragged=np.zeros([n,n])
for k in range(n):
    for l in range(n):
        cov_mat_ragged[k,l]=cov_fun_ragged(t[k],t[l])



# ii) Simulate signal

sigma_noise=0.3

f_smooth_1=np.random.multivariate_normal(np.zeros([n]), cov_mat_smooth_1)
f_smooth_2=np.random.multivariate_normal(np.zeros([n]), cov_mat_smooth_2)
f_ragged=np.random.multivariate_normal(np.zeros([n]), cov_mat_ragged)
noise=np.random.normal(0,sigma_noise,[n])

f=f_smooth_1+f_smooth_2+f_ragged+noise



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


# ii) Successive addition

f_rec_1=F[0:1,:].T@F[0:1,:]@f
f_rec_2=F[0:3,:].T@F[0:3,:]@f
f_rec_3=F[0:5,:].T@F[0:5,:]@f
f_rec_4=F[0:20,:].T@F[0:20,:]@f
f_rec_5=F[20:,:].T@F[20:,:]@f





"""
    5. Plots and Illustrations -----------------------------------------------
"""


ylims=[-4,4]

# i) The signal

plt.figure(1,dpi=300)
plt.plot(t,f,color='k')
plt.title('The signal')
plt.xlabel('Time t')
plt.ylabel('Function value')
plt.ylim(ylims)
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
plt.ylim(ylims)
plt.xticks([])
plt.yticks([])

f1_ax3=f1.add_subplot(gs1[0,2])
f1_ax3.plot(t,f_3,color='k')
plt.title('Component 3')
plt.xlabel('Time t')
plt.ylim(ylims)
plt.xticks([])
plt.yticks([])


# iii) The reconstruction

plt.figure(3,dpi=300)
plt.plot(t,f_rec_1,color='k')
plt.ylim(ylims)
plt.xticks([])
plt.yticks([])

plt.figure(4,dpi=300)
plt.plot(t,f_rec_2,color='k')
plt.ylim(ylims)
plt.xticks([])
plt.yticks([])

plt.figure(5,dpi=300)
plt.plot(t,f_rec_3,color='k')
plt.ylim(ylims)
plt.xticks([])
plt.yticks([])

plt.figure(6,dpi=300)
plt.plot(t,f_rec_4,color='k')
plt.ylim(ylims)
plt.xticks([])
plt.yticks([])

plt.figure(7,dpi=300)
plt.plot(t,f_rec_5,color='k')
plt.ylim(ylims)
plt.xticks([])
plt.yticks([])

plt.figure(8,dpi=300)
markerline, stemlines, baseline=plt.stem(F@f,linefmt='k')
markerline.set_markerfacecolor('k')
#plt.ylim(ylims)
plt.title('Expansion coefficients = spectrum')
plt.xlabel('Coefficient nr')
plt.ylabel('Size coefficient')
plt.xticks([])
plt.yticks([])




# iv) Some covariance matrices

cov_mat_full=cov_mat_smooth_1+cov_mat_smooth_2+sigma_noise**2*np.eye(n)

plt.figure(9,dpi=300)
plt.imshow(cov_mat_full)
plt.xticks([])
plt.yticks([])

index_choice=np.array([1,10,50])
index_mesh=np.ix_(index_choice,index_choice)

plt.figure(10,dpi=300)
plt.imshow(cov_mat_full[index_mesh])
plt.xticks([])
plt.yticks([])
























