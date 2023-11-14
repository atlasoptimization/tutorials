"""
The goal of this script is to do try out t-SNE on a multidimensional dataset.
For this, do the following:
    1. Definitions and imports
    2. Generate data
    3. Perform t-SNE
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

import sklearn as skl
from sklearn.manifold import TSNE


# ii) Definitions

n_dim=2
n_class=4
n_pt=100

t=np.linspace(0,1,n_pt)
class_label=np.round(np.linspace(0,n_class, n_pt)).astype(int)


"""
    2. Generate data
"""


# i) 2D Data for spiral

r=np.linspace(0,1,n_pt)
phi=np.linspace(0,4*np.pi,n_pt)

x=np.zeros(n_pt)
y=np.zeros(n_pt)

for k in range(n_pt):
    x[k]=r[k]*np.cos(phi[k])
    y[k]=r[k]*np.sin(phi[k])

coords=np.vstack((x,y)).T


"""
    3. Perform t-SNE
"""


# i) t-SNE to 2D space

coords_embedded_2D=TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(coords)


# ii) t-SNE to 1D space

coords_embedded_1D=TSNE(n_components=1, learning_rate='auto', init='random', perplexity=3).fit_transform(coords)



"""
    4. Plots and Illustrations
"""


# i) Plot unembedded data

plt.figure(1,dpi=300)
plt.scatter(x,y,c=class_label)
plt.title('Classes and original features')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.xticks([])
plt.yticks([])
plt.axis('equal')


# ii) Plot embedded data 2D

plt.figure(2,dpi=300)
plt.scatter(coords_embedded_2D[:,0],coords_embedded_2D[:,1],c=class_label)
plt.title('2D Embedding')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.xticks([])
plt.yticks([])
plt.axis('equal')


# iii) Plot embedded data 1D

plt.figure(3,dpi=300)
plt.scatter(coords_embedded_1D[:,0],np.ones(n_pt),c=class_label)
plt.title('1D Embedding')
plt.xlabel('Feature 1')
plt.xticks([])
plt.yticks([])
plt.axis('equal')

