"""
The goal of this script is to simulate some images and test them with non-normalized
probability distributions
For this, do the following:
    1. Definitions and imports
    2. Generate features
    3. Generate images
    4. Test with nn-prob dists
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

cameraman_image=np.load('/home/jemil/Desktop/Programming/Python/Teaching/Programs_CMGA/cameraman_image.npy')

# ii) Definitions


n_x = 30
n_y = 30
x = np.linspace(-1,1,n_x)
y = np.linspace(-1,1,n_y)

xx,yy = np.meshgrid(x,y)

n_total = n_x*n_y
np.random.seed(1)



"""
    2. Generate features -----------------------------------------------------
"""

# i) Covariance matrix

d=0.3
cov_fun = lambda x,y: np.exp(-(np.linalg.norm(x-y)/d)**2)

Cov_mat = np.zeros([n_total,n_total])

for k in range(n_total):
    for l in range(n_total):
        loc_1_temp = np.array([xx.flatten()[k], yy.flatten()[k]])
        loc_2_temp = np.array([xx.flatten()[l], yy.flatten()[l]])
        Cov_mat[k,l] = cov_fun(loc_1_temp, loc_2_temp)
        
        
pinv_C = np.linalg.pinv(Cov_mat)
cov_feature = lambda image: image.flatten().T @pinv_C@image.flatten()

nn_cov_log_prob = lambda image: -cov_feature(image)


# ii) Total variation

def tv_feature(image):
    image_dx = image[0:-1,1:]
    image_dy = image[1:,0:-1]
    image_base = image[0:-1,0:-1]
    
    tv = np.sum(np.sqrt(np.power(image_base - image_dx,2)+np.power(image_base - image_dy,2)))
    return tv

nn_tv_log_prob = lambda image: -tv_feature(image)


# ii) white noise prob

nn_wn_log_prob = lambda image: -image.flatten().T@image.flatten()


"""
    3. Generate images -------------------------------------------------------
"""


# i) white noise

white_noise_1 = np.random.normal(0,1,[n_y,n_x])
white_noise_2 = np.random.normal(0,1,[n_y,n_x])

# scale to 0,1
white_noise_1 = (white_noise_1 - np.min(white_noise_1))/(np.max(white_noise_1) - np.min(white_noise_1))
white_noise_2 = (white_noise_2 - np.min(white_noise_2))/(np.max(white_noise_2) - np.min(white_noise_2))


# ii) cameraman image

cameraman_1 = cameraman_image[np.ix_(range(50,50+n_y),range(50,50+n_x))]
cameraman_2 = cameraman_image[np.ix_(range(30,30+n_y),range(140,140+n_x))]

# scale to 0,1
cameraman_1 = (cameraman_1 - np.min(cameraman_1))/(np.max(cameraman_1) - np.min(cameraman_1))
cameraman_2 = (cameraman_2 - np.min(cameraman_2))/(np.max(cameraman_2) - np.min(cameraman_2))


# iii) Smooth stuff

smooth_1 = np.reshape(np.random.multivariate_normal(np.zeros(n_total), Cov_mat),[n_y,n_x])
smooth_2 = np.reshape(np.random.multivariate_normal(np.zeros(n_total), Cov_mat),[n_y,n_x])

# scale to 0,1
smooth_1 = (smooth_1 - np.min(smooth_1))/(np.max(smooth_1) - np.min(smooth_1))
smooth_2 = (smooth_2 - np.min(smooth_2))/(np.max(smooth_2) - np.min(smooth_2))



"""
    4. Test with nn-prob dists -----------------------------------------------
"""


# i) white noise

wn_wn1 = nn_wn_log_prob(white_noise_1)
wn_wn2 = nn_wn_log_prob(white_noise_2)

cov_wn1 = nn_cov_log_prob(white_noise_1)
cov_wn2 = nn_cov_log_prob(white_noise_2)

tv_wn1 = nn_tv_log_prob(white_noise_1)
tv_wn2 = nn_tv_log_prob(white_noise_2)


# ii) smooth

wn_sm1 = nn_wn_log_prob(smooth_1)
wn_sm2 = nn_wn_log_prob(smooth_2)

cov_sm1 = nn_cov_log_prob(smooth_1)
cov_sm2 = nn_cov_log_prob(smooth_2)

tv_sm1 = nn_tv_log_prob(smooth_1)
tv_sm2 = nn_tv_log_prob(smooth_2)


# iii) cameraman

wn_cm1 = nn_wn_log_prob(cameraman_1)
wn_cm2 = nn_wn_log_prob(cameraman_2)

cov_cm1 = nn_cov_log_prob(cameraman_1)
cov_cm2 = nn_cov_log_prob(cameraman_2)

tv_cm1 = nn_tv_log_prob(cameraman_1)
tv_cm2 = nn_tv_log_prob(cameraman_2)



"""
    5. Plots and Illustrations -----------------------------------------------
"""
    

# i) Plot images and log probs of white noise

plt.figure(1,dpi=300)
plt.imshow(white_noise_1)
plt.title('white noise_1')
plt.xticks([])
plt.yticks([])

plt.figure(2,dpi=300)
plt.imshow(white_noise_2)
plt.title('white noise_2')
plt.xticks([])
plt.yticks([])


# ii) Plot images and log probs of smooth stuff

plt.figure(3,dpi=300)
plt.imshow(smooth_1)
plt.title('smooth_1')
plt.xticks([])
plt.yticks([])

plt.figure(4,dpi=300)
plt.imshow(smooth_2)
plt.title('smooth_2')
plt.xticks([])
plt.yticks([])


# iii) Plot images and log probs of cameraman

plt.figure(5,dpi=300)
plt.imshow(cameraman_1)
plt.title('cameraman_1')
plt.xticks([])
plt.yticks([])

plt.figure(6,dpi=300)
plt.imshow(cameraman_2)
plt.title('cameraman_2')
plt.xticks([])
plt.yticks([])




print('White noise 1 has log probs wn_wn1 = {} , cov_wn1 = {}, tv_wn1 ={}'.format(wn_wn1,cov_wn1, tv_wn1))
print('White noise 2 hastlog probs wn_wn2 = {} , cov_wn2 = {}, tv_wn2 ={}'.format(wn_wn2,cov_wn2, tv_wn2))

print('Smooth 1 has log probs wn_sm1 = {} , cov_sm1 = {}, tv_sm1 ={}'.format(wn_sm1,cov_sm1, tv_sm1))
print('Smooth 2 has log probs wn_sm2 = {} , cov_sm2 = {}, tv_sm2 ={}'.format(wn_sm2,cov_sm2, tv_sm2))

print('Cameraman 1 has log probs wn_cm1 = {} , cov_cm1 = {}, tv_cm1 ={}'.format(wn_cm1,cov_cm1, tv_cm1))
print('Cameraman 2 has log probs wn_cm2 = {} , cov_cm2 = {}, tv_cm2 ={}'.format(wn_cm2,cov_cm2, tv_cm2))











