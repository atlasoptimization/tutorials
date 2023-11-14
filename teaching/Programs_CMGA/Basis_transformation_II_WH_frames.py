"""
The goal of this script is to generate the first few Weyl Heisenberg frame 
functions
For this, do the following:
    1. Definitions and imports
    2. Generate functions
    3. Plots and Illustrations
    
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

n_dim=100
t=np.linspace(0,1,n_dim)

T=0.5
F=0.5



"""
    2. Generate functions ----------------------------------------------------
"""


# i) Generate function handle

window_fun=lambda t: np.sqrt(1/T)*(T>=t and t>=0)
WH_fun=lambda m,n,t: np.exp(2*np.pi*n*1j*F*t)*window_fun(t-m*T)


# ii) Generate some concrete ones

m,n=np.meshgrid([0,1],[0,1])
m=m.flatten()
n=n.flatten()

wh_fun=np.zeros([n_dim,4],dtype=complex)
for k in range(len(m)):
    for l in range(n_dim):
        wh_fun[l,k]=WH_fun(m[k],n[k],t[l])



"""
    3. Plots and Illustrations -----------------------------------------------
"""


# i) The WH basis functions

ylims=[-1.5,1.5]

(w,h)=plt.figaspect(0.25)
f1=plt.figure(1,dpi=300,figsize=(w,h))
gs1=f1.add_gridspec(1,4)

f1_ax1=f1.add_subplot(gs1[0,0])
f1_ax1.plot(t,np.real(wh_fun[:,0]))
f1_ax1.plot(t,np.imag(wh_fun[:,0]))
plt.title('m=0,n=0')
plt.ylim(ylims)
plt.xticks([])
plt.yticks([])

f1_ax2=f1.add_subplot(gs1[0,1])
f1_ax2.plot(t,np.real(wh_fun[:,1]))
f1_ax2.plot(t,np.imag(wh_fun[:,1]))
plt.title('m=1,n=0')
plt.ylim(ylims)
plt.xticks([])
plt.yticks([])

f1_ax3=f1.add_subplot(gs1[0,2])
f1_ax3.plot(t,np.real(wh_fun[:,2]))
f1_ax3.plot(t,np.imag(wh_fun[:,2]))
plt.title('m=0,n=1')
plt.ylim(ylims)
plt.xticks([])
plt.yticks([])

f1_ax4=f1.add_subplot(gs1[0,3])
f1_ax4.plot(t,np.real(wh_fun[:,3]))
f1_ax4.plot(t,np.imag(wh_fun[:,3]))
plt.title('m=1,n=1')
plt.ylim(ylims)
plt.xticks([])
plt.yticks([])


f1.tight_layout()





