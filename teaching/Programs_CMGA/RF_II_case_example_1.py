"""
The goal of this script is to simulate a random field and then perform conditional
simulation to solve several questions regarding probable outcomes. 
For this, do the following:
    1. Definitions and imports
    2. Generate data
    3. Conditional simulation
    4. Assemble results
    5. Plots and Illustrations
    
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

# It is assumed that the whole region represents 1000m^2 independent of amount of
# pixels used
n_x = 20
n_y = 20
x = np.linspace(-1,1,n_x)
y = np.linspace(-1,1,n_y)

xx,yy = np.meshgrid(x,y)

n_total = n_x*n_y
n_simu_conditional = 100
n_sample = 10

np.random.seed(0)



"""
    2. Generate data ----------------------------------------------------------
"""

# i) Covariance matrix
d = 0.6
cov_fun = lambda x,y: 1*np.exp(-(np.linalg.norm(x-y)/d)**2)

Cov_mat = np.zeros([n_total,n_total])

for k in range(n_total):
    for l in range(n_total):
        loc_1_temp = np.array([xx.flatten()[k], yy.flatten()[k]])
        loc_2_temp = np.array([xx.flatten()[l], yy.flatten()[l]])
        Cov_mat[k,l] = cov_fun(loc_1_temp, loc_2_temp)


# ii) Simulation: f = log Cadmium concentration
# If you do e^f, the you get the cadmium concentration in g for each pixel

f_simu = np.reshape(np.random.multivariate_normal(0*np.ones(n_total), Cov_mat), [n_y,n_x],'C')


# iii) Generate samples

sample_indices = np.random.choice(np.linspace(0,n_total-1,n_total), size = n_sample, replace = False).astype(int)
sample_coords = np.vstack((xx.flatten()[sample_indices],yy.flatten()[sample_indices]))

data_sample = f_simu.flatten()[sample_indices]



"""
    3. Conditional simulation ----------------------------------------------------------
"""

# i) Create expected mean and covariance


# 1 = predicted, 2 = observed
Sigma_11 = Cov_mat
Sigma_22 = Cov_mat[np.ix_(sample_indices,sample_indices)]
Sigma_12 = Cov_mat[:,sample_indices]

mu_1 = np.zeros(n_total)
mu_2 = np.zeros(n_sample)

mu_bar =  mu_1 + Sigma_12@np.linalg.inv(Sigma_22)@(data_sample - mu_2)
Sigma_bar = Sigma_11 - Sigma_12@np.linalg.inv(Sigma_22)@(Sigma_12.T)

conditional_expectation = np.reshape(mu_bar,[n_y,n_x])
conditional_covmat = np.reshape(Sigma_bar,[n_total,n_total])


# ii) Conditional simulations

conditional_simus = np.zeros([n_simu_conditional, n_total])
for k in range(n_simu_conditional):
    conditional_simus[k,:] = np.random.multivariate_normal(mu_bar, Sigma_bar)

f_simu_conditional = np.reshape(conditional_simus,[n_simu_conditional, n_y,n_x], order ='C')



"""
    4. Assemble results ----------------------------------------------------------
"""

# i) Approximate conditional expectation

f_hat = np.mean(f_simu_conditional,0)


# ii) Approximate probability of having more than 1 kg of Cadmium

total_cd = lambda f: np.mean(np.exp(f))*1000 # average scaled up by area

more_than_1kg =np.zeros(n_simu_conditional)
for k in range(n_simu_conditional):
    more_than_1kg[k] = total_cd(f_simu_conditional[k,:,:])>=1000

prob_mt1kg = np.sum(more_than_1kg)/n_simu_conditional


# iii) Approximate probability of having less than 200g of Cadmium in certain area
# certain area = top left quarter

quarter_index_x = np.ceil(n_x /2).astype(int)
quarter_index_y = np.ceil(n_y /2).astype(int)
area_cd = lambda f: np.mean(np.exp(f[np.ix_(range(quarter_index_y),range(quarter_index_x))]))*250 # average scaled up by area

less_than_200g =np.zeros(n_simu_conditional)
for k in range(n_simu_conditional):
    less_than_200g[k] = area_cd(f_simu_conditional[k,:,:])<=200

prob_lt200g = np.sum(less_than_200g)/n_simu_conditional



"""
    5. Plots and Illustrations -----------------------------------------------
"""

vmin_plot =-2
vmax_plot = 2

# i) Plot data and analytical best guess

plt.figure(1,dpi=300)
plt.imshow(np.flipud(f_simu), extent = [-1,1,-1,1], vmin = vmin_plot, vmax =vmax_plot)
plt.scatter(sample_coords[0,:], sample_coords[1,:], c='r')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Ground truth and data')
plt.xticks([])
plt.yticks([])

plt.figure(2,dpi=300)
plt.imshow(np.flipud(conditional_expectation), extent = [-1,1,-1,1], vmin = vmin_plot, vmax = vmax_plot)
plt.scatter(sample_coords[0,:], sample_coords[1,:], c='r')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('True conditional expectation')
plt.xticks([])
plt.yticks([])

plt.figure(3,dpi=300)
plt.imshow(Sigma_bar)
plt.xlabel('nr point')
plt.ylabel('nr point')
plt.title('Conditional covariance')
plt.xticks([])
plt.yticks([])


# ii) Plot some conditional simulations

plt.figure(4,dpi=300)
plt.imshow(np.flipud(f_simu_conditional[0,:,:]), extent = [-1,1,-1,1], vmin = vmin_plot, vmax =vmax_plot)
plt.scatter(sample_coords[0,:], sample_coords[1,:], c='r')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Conditional simulation')
plt.xticks([])
plt.yticks([])

plt.figure(5,dpi=300)
plt.imshow(np.flipud(f_simu_conditional[1,:,:]), extent = [-1,1,-1,1], vmin = vmin_plot, vmax =vmax_plot)
plt.scatter(sample_coords[0,:], sample_coords[1,:], c='r')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Conditional simulation')
plt.xticks([])
plt.yticks([])

plt.figure(6,dpi=300)
plt.imshow(np.flipud(f_simu_conditional[2,:,:]), extent = [-1,1,-1,1], vmin = vmin_plot, vmax =vmax_plot)
plt.scatter(sample_coords[0,:], sample_coords[1,:], c='r')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Conditional simulation')
plt.xticks([])
plt.yticks([])

# iii) Plot some of the ingredients to estimation

plt.figure(7,dpi=300)
plt.imshow(Sigma_12)
plt.xlabel('points')
plt.ylabel('samples')
plt.title('Sigma 12')
plt.xticks([])
plt.yticks([])

plt.figure(8,dpi=300)
plt.imshow(Sigma_22)
plt.xlabel('sample')
plt.ylabel('samples')
plt.title('Sigma 22')
plt.xticks([])
plt.yticks([])

plt.figure(9,dpi=300)
plt.imshow(np.reshape(data_sample,[n_sample,1]))
plt.ylabel('samples')
plt.title('data')
plt.xticks([])
plt.yticks([])


# iv) answer questions

print(' We have simulated some conditional data')
print( (' Out best guess for the probability of there being more than 1kg of Cadmium in total' 
        ' is {}. We calculated this value by summing over conditional simulations').format(prob_mt1kg))
print( (' Out best guess for the probability of there being less than 200 g of Cadmium in area' 
        ' is {}. We calculated this value by summing over conditional simulations').format(prob_lt200g))






