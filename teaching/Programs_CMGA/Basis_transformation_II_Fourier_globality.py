"""
The goal of this script is to simulate a very simple signal and decompose it 
into its frequencies. The impact of a single changed value of that signal is
investigated
For this, do the following:
    1. Definitions and imports
    2. Simulate signal
    3. Fourier Analysis
    4. Plots and Illustrations
    
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


# i) Very simple signal - a wave

f=np.sin(2*np.pi*t)
f_changed=np.sin(2*np.pi*t)+np.eye(n)[:,30]+np.eye(n)[:,60]



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
alpha_changed=F@f_changed



"""
    5. Plots and Illustrations -----------------------------------------------
"""


ylims=[-2,2]


# i) The decomposition

(w,h)=plt.figaspect(1)
f1=plt.figure(2,dpi=300,figsize=(w,h))
gs1=f1.add_gridspec(2,2)

f1_ax1=f1.add_subplot(gs1[0,0])
f1_ax1.plot(t,f,color='k')
plt.title('The signal')
plt.xlabel('Time t')
plt.ylabel('Function value')
plt.ylim(ylims)
plt.xticks([])
plt.yticks([])

f1_ax2=f1.add_subplot(gs1[0,1])
markerline, stemlines, baseline=plt.stem(F@f,linefmt='k')
markerline.set_markerfacecolor('k')
plt.title('Spectrum')
plt.xlabel('Coefficient nr')
plt.ylabel('Size coefficient')
plt.xticks([])
plt.yticks([])

f1_ax3=f1.add_subplot(gs1[1,0])
f1_ax3.plot(t,f_changed,color='k')
plt.title('The changed signal')
plt.xlabel('Time t')
plt.ylabel('Function value')
plt.ylim(ylims)
plt.xticks([])
plt.yticks([])

f1_ax4=f1.add_subplot(gs1[1,1])
markerline, stemlines, baseline=plt.stem(F@f_changed,linefmt='k')
markerline.set_markerfacecolor('k')
plt.title('Spectrum')
plt.xlabel('Coefficient nr')
plt.ylabel('Size coefficient')
plt.xticks([])
plt.yticks([])

f1.tight_layout()


















