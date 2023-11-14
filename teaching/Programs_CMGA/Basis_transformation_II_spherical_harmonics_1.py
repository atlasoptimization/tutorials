"""
The goal of this script is to plot spherical harmonics and use them for fitting
a function defined on the sphere.
For this, do the following:
    1. Definitions and imports
    2. Generate data
    3. Invoke and fit spherical harmonics
    4. Plots and Illustrations
    
The script is meant solely for educational and illustrative purposes. Written by
Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.ch.
"""


"""
    1. Definitions and imports -----------------------------------------------
"""


# i) Imports

import numpy as np
from scipy.special import sph_harm
from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D


# ii) Definitions

n=100
n_sample=10
n_data=n_sample**2

phi=np.linspace(0,np.pi,n)
phi_sample=np.linspace(0,np.pi,n_sample)
theta=np.linspace(0,2*np.pi,n)
theta_sample=np.linspace(0,2*np.pi,n_sample)

pp, tt=np.meshgrid(phi,theta)
pp_sample, tt_sample=np.meshgrid(phi_sample,theta_sample)

n_exp=20
n_all=np.round(n_exp*(n_exp+1)/2).astype(int)



"""
    2. Generate data ---------------------------------------------------------
"""


# i) Generate data randomly

np.random.seed(10)
#f_sample=np.random.normal(0,1,n_data)
f_sample_square=gaussian_filter(np.random.normal(0,1,[n_sample,n_sample]),sigma=2)
f_sample=f_sample_square.flatten()
sample_coords=np.vstack((pp_sample.flatten(),tt_sample.flatten())).T

# ii) Cartesian coordinates

x = np.sin(pp) * np.cos(tt)
y = np.sin(pp) * np.sin(tt)
z = np.cos(pp)



"""
    3. Invoke and fit spherical harmonics ------------------------------------
"""


# i) Fill a matrix with spherical harmonics

m_sph=1
n_sph=4
sph_mn=np.zeros([n,n],dtype=complex)


for k in range(n):
    for l in range(n):
        sph_mn[k,l]=sph_harm(m_sph,n_sph,theta[k],phi[l])


# ii) Generate harmonical basis

sph=np.zeros([n,n,n_all],dtype=complex)

q=-1
for nn in range(n_exp):
    for mm in range(nn):
        q=q+1
        for k in range(n):
            for l in range(n):
                sph[k,l,q]=sph_harm(mm,nn,theta[k],phi[l])
        
sph_basis=np.hstack((np.real(np.reshape(sph,[n**2,n_all])),np.imag(np.reshape(sph,[n**2,n_all]))))


# iii) Fit harmonics to data

index_sample=np.round(np.linspace(0,n-1,n_sample)).astype(int)
ii,jj=np.meshgrid(index_sample,index_sample)

index_vector=np.vstack((ii.flatten(),jj.flatten())).T

linear_index=np.zeros([n_data])
for k in range(n_data):
    linear_index[k]=np.ravel_multi_index((index_vector[k,0],index_vector[k,1]), [n,n])

linear_index=np.round(linear_index).astype(int)

# Fit the model
sph_basis_at_sample=sph_basis[linear_index,:]
alpha=np.linalg.pinv(sph_basis_at_sample)@f_sample
f_hat=np.reshape(sph_basis@alpha,[n,n])



"""
    4. Plots and Illustrations -----------------------------------------------
"""


# i) Plot real and imaginary part

fcolors_real=np.real(sph_mn)
fmax, fmin = fcolors_real.max(), fcolors_real.min()
fcolors_real = (fcolors_real - fmin)/(fmax - fmin)

fcolors_imag=np.imag(sph_mn)
fmax, fmin = fcolors_imag.max(), fcolors_imag.min()
fcolors_imag = (fcolors_imag - fmin)/(fmax - fmin)

f1 = plt.figure(figsize=plt.figaspect(1.),dpi=300)
ax = f1.add_subplot(111,projection='3d')
ax.plot_surface(x, y, z,  rstride=1, cstride=1, facecolors=cm.coolwarm(fcolors_real))
ax.set_axis_off()
plt.title('Real part of spherical harmonic %d %d' %(m_sph , n_sph))
plt.show()

f2 = plt.figure(figsize=plt.figaspect(1.),dpi=300)
ax = f2.add_subplot(111,projection='3d')
ax.plot_surface(x, y, z,  rstride=1, cstride=1, facecolors=cm.coolwarm(fcolors_imag))
ax.set_axis_off()
plt.title('Imaginary part of spherical harmonic %d %d' %(m_sph , n_sph))
plt.show()


# ii) Plot data and fit

plt.figure(3,dpi=300)
plt.scatter(sample_coords[:,0],sample_coords[:,1],c=f_sample)
plt.title('Data')
plt.xticks([])
plt.yticks([])

plt.figure(4,dpi=300)
plt.imshow(f_hat)
plt.title('Fit')
plt.xticks([])
plt.yticks([])


# iii) Plot fit onto sphere

f_data=np.zeros([n**2])
f_data[linear_index]=f_sample
f_data[np.mod(linear_index+1,n**2)]=f_sample
f_data[np.mod(linear_index-1,n**2)]=f_sample
f_data[np.mod(linear_index+n,n**2)]=f_sample
f_data[np.mod(linear_index-n,n**2)]=f_sample
f_data=f_data.reshape([n,n])

fcolors_real=f_data
fmax, fmin = fcolors_real.max(), fcolors_real.min()
fcolors_real = (fcolors_real - fmin)/(fmax - fmin)

f5 = plt.figure(figsize=plt.figaspect(1.),dpi=300)
ax = f5.add_subplot(111,projection='3d')
ax.plot_surface(x, y, z,  rstride=1, cstride=1, facecolors=cm.coolwarm(fcolors_real))
ax.set_axis_off()
plt.title('Data to be interpolated')
plt.show()


fcolors_real=f_hat
fmax, fmin = fcolors_real.max(), fcolors_real.min()
fcolors_real = (fcolors_real - fmin)/(fmax - fmin)

f6 = plt.figure(figsize=plt.figaspect(1.),dpi=300)
ax = f6.add_subplot(111,projection='3d')
ax.plot_surface(x, y, z,  rstride=1, cstride=1, facecolors=cm.coolwarm(fcolors_real))
ax.set_axis_off()
plt.title('Interpolation of data using sph')
plt.show()













