"""
The goal of this script is to plot L1, L2, Linfty related quantitites. 
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
t=np.linspace(-3,3,n)
dt=(t[-1]-t[0])/n


"""
    2. Calculate the norms and probability distributions
"""

# i) L1

norm_l1_fun=lambda x: np.abs(x)

norm_l1=norm_l1_fun(t)
prob_l1_prenormalized=np.exp(-norm_l1)
prob_l1=prob_l1_prenormalized/(np.sum(prob_l1_prenormalized)*dt)


# ii) L2

norm_l2_fun=lambda x: np.abs(x)**2

norm_l2=norm_l2_fun(t)
prob_l2_prenormalized=np.exp(-norm_l2)
prob_l2=prob_l2_prenormalized/(np.sum(prob_l2_prenormalized)*dt)


# iii) Linfty

norm_li_fun=lambda x: np.abs(x)**20

norm_li=norm_li_fun(t)
prob_li_prenormalized=np.exp(-norm_li)
prob_li=prob_li_prenormalized/(np.sum(prob_li_prenormalized)*dt)



"""
    3. Plots and Illustrations
"""

# i) Plot panel

w,h=plt.figaspect(0.4)
f1=plt.figure(1,dpi=300,figsize=(w,h))
gs1=f1.add_gridspec(2, 3)

f1_ax1 = f1.add_subplot(gs1[0,0])
f1_ax1.plot(t, norm_l1, color='k')
plt.title(' $L_1$ norm')
plt.xticks([])
plt.yticks([])

f1_ax2 = f1.add_subplot(gs1[0,1])
f1_ax2.plot(t, norm_l2, color='k')
plt.title(' $L_2$ norm')
plt.xticks([])
plt.yticks([])

f1_ax3 = f1.add_subplot(gs1[0,2])
f1_ax3.plot(t, norm_li, color='k')
plt.title(' $L_{\infty}$ norm')
plt.ylim([0,10])
plt.xticks([])
plt.yticks([])

f1_ax4 = f1.add_subplot(gs1[1,0])
f1_ax4.plot(t, prob_l1, color='k')
plt.title(' $L_1$ prob dist')
plt.xticks([])
plt.yticks([])

f1_ax5 = f1.add_subplot(gs1[1,1])
f1_ax5.plot(t, prob_l2, color='k')
plt.title(' $L_2$ prob dist')
plt.xticks([])
plt.yticks([])

f1_ax6 = f1.add_subplot(gs1[1,2])
f1_ax6.plot(t, prob_li, color='k')
plt.title(' $L_{\infty}$ prob dist')
plt.xticks([])
plt.yticks([])



# ii) Compare l1 and l2

plt.figure(2,dpi=300)
plt.plot(t,prob_l1,color='k', label='l1')
plt.plot(t,prob_l2,color='k', label='l2')
plt.xlabel('Residual value')
plt.ylabel('Probability density')

plt.title(' Gaussian vs Laplacian distribution')
plt.xticks([])
plt.yticks([])

















