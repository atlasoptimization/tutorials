"""
The goal of this script is to load a 2D signal and perform a wavelet decomposition.
For this, do the following:
    1. Definitions and imports
    2. Decomposition, reconstruction
    3. Plots and Illustrations
    
The script is meant solely for educational and illustrative purposes. Written by
Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.ch.

"""


"""
    1. Definitions and imports ------------------------------------------------
"""


# i) Imports

import numpy as np
import matplotlib.pyplot as plt

import copy
import pywt

cameraman_image=np.load('/home/jemil/Desktop/Programming/Python/Teaching/Programs_CMGA/cameraman_image.npy')


# ii) Definitions - simpler due to quadratic image

n=cameraman_image.shape[1]

t=np.linspace(0,1,n)
wavelet_name='bior1.5'


"""
    2. Decomposition, reconstruction ------------------------------------------
"""


# i) Coefficients

alpha=pywt.dwt2(cameraman_image,wavelet_name)

Approximation=alpha[0]
Horizontal_detail=alpha[1][0]
Vertical_detail=alpha[1][1]
Diagonal_detail=alpha[1][2]



# ii) Reconstruct using inverse  dwt

H_10pc=copy.copy(alpha[1][0])
H_10pc[H_10pc<=10]=0
V_10pc=copy.copy(alpha[1][1])
V_10pc[V_10pc<=10]=0
D_10pc=copy.copy(alpha[1][2])
D_10pc[D_10pc<=6]=0

alpha_10percent=(Approximation,(H_10pc,V_10pc,D_10pc))

H_1pc=copy.copy(alpha[1][0])
H_1pc[H_1pc<=50]=0
V_1pc=copy.copy(alpha[1][1])
V_1pc[V_1pc<=100]=0
D_1pc=copy.copy(alpha[1][2])
D_1pc[D_1pc<=38]=0

alpha_1percent=(Approximation,(H_1pc,V_1pc,D_1pc))


rec_full_cameraman_image=pywt.idwt2(alpha, wavelet_name, mode='symmetric', axes=(-2, -1))
rec_10percent_cameraman_image=pywt.idwt2(alpha_10percent, wavelet_name, mode='symmetric', axes=(-2, -1))
rec_1percent_cameraman_image=pywt.idwt2(alpha_1percent, wavelet_name, mode='symmetric', axes=(-2, -1))



"""
    3. Plots and Illustrations ------------------------------------------------
"""


# i) The image and some basis element

plt.figure(1, dpi=300)
plt.imshow(cameraman_image)
plt.title('Original image')
plt.xticks([])
plt.yticks([])

# plt.figure(2, dpi=300)
# plt.imshow(np.reshape(F_tensor[1,:],[n,n]))
# plt.title('Basis element 1')
# plt.xticks([])
# plt.yticks([])

# plt.figure(3, dpi=300)
# plt.imshow(np.reshape(F_tensor[55,:],[n,n]))
# plt.title('Basis element 55')
# plt.xticks([])
# plt.yticks([])


# # ii) Successive reconstructions

# plt.figure(4,dpi=300)
# plt.imshow(Approximation)
# plt.title('Reconstruction with 10 basis elements')
# plt.xticks([])
# plt.yticks([])

# plt.figure(5,dpi=300)
# plt.imshow(image_rec_2)
# plt.title('Reconstruction with 100 basis elements')
# plt.xticks([])
# plt.yticks([])

# plt.figure(6,dpi=300)
# plt.imshow(image_rec_3)
# plt.title('Reconstruction with 1000 basis elements')
# plt.xticks([])
# plt.yticks([])





# iii) Reconstruction details

plt.figure(4,dpi=300)
plt.imshow(Approximation)
plt.title('Approximation')
plt.xticks([])
plt.yticks([])

plt.figure(5,dpi=300)
plt.imshow(Horizontal_detail)
plt.title('Horizontal detail')
plt.xticks([])
plt.yticks([])

plt.figure(6,dpi=300)
plt.imshow(Vertical_detail)
plt.title('Vertical detail')
plt.xticks([])
plt.yticks([])

plt.figure(7,dpi=300)
plt.imshow(Diagonal_detail)
plt.title('Diagonal detail')
plt.xticks([])
plt.yticks([])

plt.figure(8,dpi=300)
plt.imshow(Horizontal_detail+Vertical_detail+Diagonal_detail)
plt.title('Details combined')
plt.xticks([])
plt.yticks([])














