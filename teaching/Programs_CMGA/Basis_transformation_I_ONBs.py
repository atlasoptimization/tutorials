"""
The goal of this script is to plot sevaral different orthonormal basis systems.
For this, do the following:
    1. Definitions and imports
    2. Generate functions and matrices
    3. Plots and Illustrations
    
The script is meant solely for educational and illustrative purposes. Written by
Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.ch.

"""


"""
    1. Definitions and imports
"""


# i) Imports

import numpy as np
import matplotlib.pyplot as plt

from scipy.special import legendre
from scipy.special import hermite
from scipy.special import laguerre


# ii) Definitions

n=100
t=np.linspace(0,1,n)
t_l=np.linspace(-1,1,n)
t_h=np.linspace(-3,3,n)
t_lag=np.linspace(0,5,n)



"""
    2. Generate functions and matrices
"""


# i) Fourier basis

def basis_functions(k,t):
    if k==0:
        value=1
    elif (k%2==0):
        value=np.sqrt(2)*np.cos(np.pi*k*t)
    else:
        value=np.sqrt(2)*np.sin(np.pi*(k+1)*t)
        
    return value

Fourier=np.zeros([n,n])
for k in range(n):
    for l in range(n):
        Fourier[k,l]=basis_functions(k,t[l])


# ii) Legendre polynomials

Legendre=np.zeros([n,n])
for k in range(10):
    for l in range(n):
        polynomial=legendre(k)
        Legendre[k,l]=polynomial(t_l[l])
        
        
# iii) Laguerre polynomials

Laguerre=np.zeros([n,n])
for k in range(10):
    for l in range(n):
        polynomial=laguerre(k)
        Laguerre[k,l]=polynomial(t_lag[l])


# iv) Hermite polynomials

Hermite=np.zeros([n,n])
for k in range(10):
    for l in range(n):
        polynomial=hermite(k)
        Hermite[k,l]=polynomial(t_h[l])




"""
    3. Plots and Illustrations
"""


# i) Plot Fourier basis

plt.figure(1,dpi=300)
plt.plot(t,Fourier[0:5,:].T)
plt.title('Fourier basis')
#plt.ylim([-1.5,1.5])
plt.xticks([])
plt.yticks([])


# ii) Plot Legendre basis

plt.figure(2,dpi=300)
plt.plot(t_l,Legendre[0:5,:].T)
plt.title('Legendre basis')
#plt.ylim([-1.5,1.5])
plt.xticks([])
plt.yticks([])


# iii) Plot Laguerre basis

plt.figure(3,dpi=300)
plt.plot(t_l,Laguerre[0:5,:].T)
plt.title('Laguerre basis')
#plt.ylim([-1.5,1.5])
plt.xticks([])
plt.yticks([])


# iv) Plot hermite basis

plt.figure(4,dpi=300)
plt.plot(t_h,Hermite[0:5,:].T)
plt.title('Hermite basis')
plt.ylim([-50,50])
plt.xticks([])
plt.yticks([])

























