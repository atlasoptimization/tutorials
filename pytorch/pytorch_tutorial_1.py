#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is part of a series of pytorch tutorials. During these tutorials,
we will learn about tensors (part_1), use autograd and optimizers on a minimal
example (part_2), use torch for fitting a basic model via LS (part_3), learn
about pytorchs utility functions (part_4) and train a neural network to perform
a classification task (part_5). 

This script is to showcase the basic pytorch building block, the tensor. We will
figure out what a tensor is, how it compares to normal arrays and why tensors
are so useful in the context of optimization and ml.
For this, do the following:
    1. Imports and definitions
    2. Generate numpy array & check properties
    3. Generate torch tensor & check properties 
    4. Autograd functionality
    
The script is meant solely for educational and illustrative purposes. Written by
Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.com.
"""

"""
    1. Imports and definitions
"""


# i) Imports

import torch
import numpy as np



"""
    2. Generate numpy array & check properties
"""

# i) Generate numpy array

# numpy is a python package that provides a matlab-like functionality to manipulate
# scalars, vectors, matrices, and in general n-dimensional arrays of data. The
# basic building block is the class "array"

a = np.array(1)              # a scalar with value 0 
b = np.array([1,2])          # a vector with two entries
c = np.array([[1,2],[3,4]])  # a matrix with four entries

# print arrays to console
print('\n numpy array generation -----------------------------------------------')
print( ' a = \n {}\n b = \n {}\n c = \n {}'.format(a, b, c))


# ii) Methods of numpy arrays

# numpy arrays have a bunch of methods attached to them. As soon as an array
# has been created, you can : get the shape, max, or transpose or reshape the array.

shape_a = a.shape
shape_b = b.shape
shape_c = c.shape

max_of_c = c.max()
transpose_of_c = c.T
reshape_of_c = c.reshape([4,1])

# print results to console
print('\n numpy array methods --------------------------------------------------')
print( ' shape of a = {}, shape of b = {}, shape of c = {}'.format(shape_a, shape_b, shape_c))
print( ' maximum of c = \n {}\n transpose of c = \n {}\n reshape of c = \n {}\n'.format(max_of_c, transpose_of_c, reshape_of_c))


# iii) Functions in numpy library

# numpy also has a bunch of convenience functions to generate arrays and 
# manipulate arrays. numpy functions typically act on arrays to generate new arrays.

# Square function and derivative
np_x = np.linspace(0,1,11)         # x axis
np_y = np.square(np_x)              # y = x^2 
np_y_derivative = 2*np_x            # y' = 2*x   <- you have to do the derivative by hand


# print results to images
print('\n numpy array functions -------------------------------------------------')
print(' np_x = \n {} \n np_y = \n {} \n np_y_derivative = \n {} \n'.format(np_x, np_y, np_y_derivative))
print('\n --------------------- END OF NUMPY ----------------------------------- \n')



"""
    3. Generate torch tensor & check properties 
"""


# pytorch is a state-of-the-art python library that is used for machine learning.
# Simplifying a bit, pytorch can be seen as numpy but with support for automatic
# differentiation. The basic building block in pytorch is the tensor - tensors
# are objects like numpy arrays but they are delivered together with a history
# of which operations have created them. This allows calculating derivatives
# using the chain rule; a basic requirement for optimizing parameters in a model.
# Other than the tensor class, pytorch also provides a nice ecosystem for machine
# learning with many helpful constructions like specific datasets, optimization 
# algorithms, and typical neural nets already preimplemented.

# i) Generate torch tensors

# pytorch tensors are just like numpy arrays but they have some extra methods 
# associated to them. The basic building block is the class "tensor". Tensors
# are generated just like numpy arrays but with a different command.

t_a = torch.tensor(1.0)                    # a scalar with value 0 
t_b = torch.tensor([1.0,2.0])              # a vector with two entries
t_c = torch.tensor([[1.0,2.0],[3.0,4.0]])  # a matrix with four entries

# print arrays to console
print('\n torch tensor generation -----------------------------------------------')
print( ' t_a = \n {}\n t_b = \n {}\n t_c = \n {}'.format(t_a, t_b, t_c))



# ii) Methods of torch tensors

# pytorch tensors have the same methods as numpy arrays - and a few more that
# numpy arrays dont have. The results of the numpy-typical methods yield
# expected results. Lets look at them first.

# These are like in numpy
shape_t_a = t_a.shape
shape_t_b = t_b.shape
shape_t_c = t_c.shape

max_of_t_c = t_c.max()
transpose_of_t_c = t_c.T
reshape_of_t_c = t_c.reshape([4,1])

# print results to console
print('\n torch tensor methods I -----------------------------------------------')
print( ' shape of t_a = {}, shape of t_b = {}, shape of t_c = {}'.format(shape_t_a, shape_t_b, shape_t_c))
print( ' maximum of t_c = \n {}\n transpose of t_c = \n {}\n reshape of t_c = \n {}\n'.format(max_of_t_c, transpose_of_t_c, reshape_of_t_c))

# The following methods are new and have no analogy in numpy: 
#   requires_grad : track tensor
#   grad_fn  : gradient function of tensor
#   backward : give numerical value to gradient
#   detach  : untrack tensor
#
# They are all related to what is called the "computational graph". By declaring
# e.g. x.requires_grad = True, we declare that pytorch should track what happens
# to x so we know afterwards how x affected any end-result we might produce.
# grad_fn allows to access the origin and derivative (->gradient), backward() 
# takes any numerical values and uses them to calculate numerical derivatives.
# x.detach() removes x from the computational graph.
# Lets apply them in these methods to calculate derivatives automatically.

# track, manipulate, calculate gradients, and untrack a tensor
t_a.requires_grad = True                # add to computational graph
dependent_value = 2*t_a                 # new value is dependent on t_a
gradient_fn = dependent_value.grad_fn   # the gradient function w.r.t. t_a
dependent_value.backward()              # go from dependent value backwards to compute numerical value of gradients
gradient_num = t_a.grad                 # gradients are stored in attribute of tensors

# print results to console
print('\n torch tensor methods II ----------------------------------------------')
print(' t_a now requires that gradients are formed to measure its impact. \n t_a = {}'.format(t_a.__repr__()))
print(' dependent_value = \n {} \n gradient_fn = {} \n gradient_num = {}'.format(dependent_value.__repr__(), gradient_fn.__repr__(), gradient_num.__repr__()))



"""
    4. Autograd functionality
"""

# iii) Functions in pytorch library

# pytorch also has a bunch of convenience functions to generate tensors and 
# manipulate them. torch functions typically act on tensors to generate new tensors.
# Almost all of the functions in the pytorch library come together with a gradient
# function. So when you call e.g. y = torch.square(x), the gradient dy/dx is
# recorded as an abstract function (yes, even for things like reshaping and 
# taking the max).
# The functionality we need for forming gradients has actually already been presented.

# Square function and derivative
torch_x = torch.linspace(0,1,11)                # x axis
torch_x.requires_grad = True
torch_y = torch.sum(torch.square(torch_x))       # y = Sum x^2 (i.e. first squaring, then summing)
gradient_fn_y_1 = torch_y.grad_fn                # gradient y / last torch function (sum)
gradient_fn_y_2 = gradient_fn_y_1.next_functions # gradient last torch function / second to last torch function (square)
                                                 # <- now gradients come from pytorch
torch_y.backward ()
dy_dx_num = torch_x.grad                         # numerical value for chained gradient dy/dx  


# print results to console
print('\n torch functions and autograd ----------------------------------------------')
print((' torch_y is produced via summing and squaring torch_x. This is reflected in the\n'\
      ' gradients. The first gradient is given as gradient_fn_y_1 = {}. \n'\
      ' The second gradient is given as gradient_fn_y_2 = {}'.format(gradient_fn_y_1, gradient_fn_y_2)))
print(' The numerical value for the chained gradient is given as {}'.format(dy_dx_num))
print('--------------------- END OF TORCH ----------------------------------- \n')

# Please briefly discuss the relationship between the manual derivative we calculated
# in numpy, the gradient functions in torch and the final gradient value dy_dx_num.

