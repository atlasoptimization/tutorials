"""
The goal of this script is to load a 2D signal and decompose it into its
components. This is supported by a Fourier analysis.
For this, do the following:
    1. Definitions and imports
    2. Fourier Analysis
    3. Decomposition, reconstruction
    4. Plots and Illustrations
    
The script is meant solely for educational and illustrative purposes. Written by
Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.ch.

"""


"""
    1. Definitions and imports
"""


# i) Imports

import numpy as np
import matplotlib.pyplot as plt

cameraman_image=np.load('/home/jemil/Desktop/Programming/Python/Teaching/Programs_CMGA/cameraman_image.npy')


# ii) Definitions - simpler due to quadratic image

n=cameraman_image.shape[1]

t=np.linspace(0,1,n)



"""
    2. Fourier Analysis
"""


# i) Build 1D fourier basis

def basis_functions(k,t):
    if k==0:
        value=1
    elif (k%2==0):
        value=np.sqrt(2)*np.cos(np.pi*k*t)
    else:
        value=np.sqrt(2)*np.sin(np.pi*(k+1)*t)
        
    return value/np.sqrt(n)


# ii) Build basis transformation matrix 1D

F=np.zeros([n,n])
for k in range(n):
    for l in range(n):
        F[k,l]=basis_functions(k,t[l])
        
F_small=F[0:40,:]


# iii) Build basis for 2D 

# First index denotes nr basis element, the rest is a vectorized basis image
F_tensor=np.kron(F_small,F_small)



"""
    3. Decomposition, reconstruction
"""


# i) Reconstruct using 10, 100, 1000, basis elements

image_vectorized=np.reshape(cameraman_image,[n**2])

image_rec_1=F_tensor[0:10,:].T@(F_tensor[0:10,:]@image_vectorized)
image_rec_1=np.reshape(image_rec_1,[n,n])

image_rec_2=F_tensor[0:100,:].T@(F_tensor[0:100,:]@image_vectorized)
image_rec_2=np.reshape(image_rec_2,[n,n])

image_rec_3=F_tensor[0:1000,:].T@(F_tensor[0:1000,:]@image_vectorized)
image_rec_3=np.reshape(image_rec_3,[n,n])



"""
    4. Plots and Illustrations
"""


# i) The image and some basis element

plt.figure(1, dpi=300)
plt.imshow(cameraman_image)
plt.title('Original image')
plt.xticks([])
plt.yticks([])

plt.figure(2, dpi=300)
plt.imshow(np.reshape(F_tensor[1,:],[n,n]))
plt.title('Basis element 1')
plt.xticks([])
plt.yticks([])

plt.figure(3, dpi=300)
plt.imshow(np.reshape(F_tensor[55,:],[n,n]))
plt.title('Basis element 55')
plt.xticks([])
plt.yticks([])


# ii) Successive reconstructions

plt.figure(4,dpi=300)
plt.imshow(image_rec_1)
plt.title('Reconstruction with 10 basis elements')
plt.xticks([])
plt.yticks([])

plt.figure(5,dpi=300)
plt.imshow(image_rec_2)
plt.title('Reconstruction with 100 basis elements')
plt.xticks([])
plt.yticks([])

plt.figure(6,dpi=300)
plt.imshow(image_rec_3)
plt.title('Reconstruction with 1000 basis elements')
plt.xticks([])
plt.yticks([])



















