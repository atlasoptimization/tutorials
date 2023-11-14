"""
The goal of this script is to plot L1, L2, Linfty related quantitites in 2D. 
For this, do the following:
    1. Definitions and imports
    2. Calculate the norms and probability distributions
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


# ii) Definitions

n=100
t=np.linspace(-2,2,n)
dt=(t[-1]-t[0])/n

xx,yy=np.meshgrid(t,t)

vv=np.stack((xx,yy),axis=2)


"""
    2. Calculate the norms and probability distributions
"""

# i) L1

norm_l1=np.zeros([n,n])
prob_l1=np.zeros([n,n])

norm_l1_fun=lambda v1: np.linalg.norm(v1,ord=1)

for k in range(n):
    for l in range(n):   
        norm_l1[k,l]=norm_l1_fun(vv[k,l])
        prob_l1=np.exp(-norm_l1)


# ii) L2

norm_l2=np.zeros([n,n])
prob_l2=np.zeros([n,n])

norm_l2_fun=lambda v1: np.linalg.norm(v1,ord=2)

for k in range(n):
    for l in range(n):   
        norm_l2[k,l]=norm_l2_fun(vv[k,l])
        prob_l2=np.exp(-norm_l2)


# iii) Linfty

norm_li=np.zeros([n,n])
prob_li=np.zeros([n,n])

norm_li_fun=lambda v1: np.linalg.norm(v1,ord=np.inf)

for k in range(n):
    for l in range(n):   
        norm_li[k,l]=norm_li_fun(vv[k,l])
        prob_li=np.exp(-norm_li)



"""
    3. Plots and Illustrations
"""

# i) Plot panel

w,h=plt.figaspect(0.8)
f1=plt.figure(1,dpi=300,figsize=(w,h))
gs1=f1.add_gridspec(2, 3)

f1_ax1 = f1.add_subplot(gs1[0,0])
f1_ax1.imshow(norm_l1)
plt.title(' $L_1$ norm')
plt.xticks([])
plt.yticks([])

f1_ax2 = f1.add_subplot(gs1[0,1])
f1_ax2.imshow(norm_l2)
plt.title(' $L_2$ norm')
plt.xticks([])
plt.yticks([])

f1_ax3 = f1.add_subplot(gs1[0,2])
f1_ax3.imshow(norm_li)
plt.title(' $L_{\infty}$ norm')
plt.xticks([])
plt.yticks([])

f1_ax4 = f1.add_subplot(gs1[1,0])
f1_ax4.imshow( prob_l1)
plt.title(' $L_1$ prob dist')
plt.xticks([])
plt.yticks([])

f1_ax5 = f1.add_subplot(gs1[1,1])
f1_ax5.imshow(prob_l2)
plt.title(' $L_2$ prob dist')
plt.xticks([])
plt.yticks([])

f1_ax6 = f1.add_subplot(gs1[1,2])
f1_ax6.imshow(prob_li)
plt.title(' $L_{\infty}$ prob dist')
plt.xticks([])
plt.yticks([])






















