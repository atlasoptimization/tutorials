
"""
Optimize price distribution for Petra
For this, do the following:
    1. Imports and definitions
    2. Set up problem
    3. Optimize parameters
    4. Plots and illustrations
"""

"""
    1. Imports and definitions
"""


# i) Imports
import numpy as np
import scipy.optimize as scopt
import matplotlib.pyplot as plt


# ii) Definitions
obs=np.array([[0.01, 100], [0.0625,270], [0.25, 1000],[1, 2300], [3, 6000]])
n=100
t=np.linspace(0,4,n)

"""
    2. Set up problem
"""

# i) Define loss function
f= lambda x1, x2, x3, s: np.sqrt(s + x1**2)*x2+x3
target_vec=obs[:,1]
obs_hat=lambda x1,x2,x3: np.array([f(x1,x2,x3,obs[0,0]),f(x1,x2,x3,obs[1,0]),f(x1,x2,x3,obs[2,0]),f(x1,x2,x3,obs[3,0]), f(x1,x2,x3,obs[4,0])])
loss_fun=lambda x: np.linalg.norm(target_vec.T-obs_hat(x[0],x[1],x[2]).T)


"""
    3. Optimize parameters
"""

opt=scopt.minimize(loss_fun,np.array([[0],[0],[0]]))
print(opt)
x_opt=opt.x



"""
    4. Plots and illustrations
"""

# i) Plot of optimized function

f_vals=np.zeros([n,1])
f_opt=lambda s: f(x_opt[0],x_opt[1],x_opt[2],s)
for k in range(n):
    f_vals[k]=f_opt(t[k])
    
ff=plt.figure(dpi=500)
plt.plot(t,f_vals)
plt.scatter(obs[:,0],obs[:,1])
plt.xlabel('Area in m2')
plt.ylabel('Price in CHF')
plt.title('Price vs painted area')

























