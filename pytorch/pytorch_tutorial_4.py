#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is part of a series of pytorch tutorials. During these tutorials,
we will learn about tensors (part_1), use autograd and optimizers on a minimal
example (part_2), use torch for fitting a basic model via LS (part_3), learn
about pytorchs utility functions (part_4) and train a neural network to perform
a classification task (part_5). 

This script illustrates some aspects of python and pytorch that are tied to 
object oriented programming and therefor a bit more difficult to understand.
Neural nets and unitility constructions like the dataloader are typically 
implemented in terms of classes : blueprints for objects. We will explore
this concept and the most important classes in pytorch and how to use them 
in practice.
For this, do the following:
    1. Imports and definitions
    2. Explore the concept of classes
    3. The DataLoader /Dataset classes (data organization)
    4. The Module class (neural nets)
    5. Plots and illustrations
    
The script is meant solely for educational and illustrative purposes. Written by
Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.com.
"""



"""
    1. Imports and definitions
"""


# i) Imports

import torch
import matplotlib.pyplot as plt


# ii) Generate some example data

# Sample from a two dimensional normal distribution creating 100 samples of 2d vectors.
n_samples = 100
dim_data = 2
mu = torch.zeros([dim_data])
sigma = torch.eye(dim_data)
data = torch.zeros([n_samples, dim_data])
for k in range(n_samples):
    data[k,:] = torch.distributions.MultivariateNormal(loc = torch.zeros([dim_data]),
                                                       covariance_matrix = torch.eye(dim_data)).sample()




"""
    2. Explore the concept of classes
"""


# Classes are at the heart of pythons bigger projects. While you can easily use
# python as a matlab style imperative programming language and just pass it a
# bunch of commands that it will execute in sequence, more complex constructions
# are often better handled in an object-oriented way. That means taking a bunch
# of variables and functions and packaging them into one neatly organized object
# where both interact nicely with each other. Since this packaging procedure 
# typically needs to be executed more than once and for different input variables,
# a blueprint for constructing an object from input variables is more useful than
# just the final object itself. These blueprints for constructing objects are
# called "classes".
# In this context of classes, the variables and static properties of a class are
# called the attributes and the functions are called the methods of the class.
# we will now build a class that incorporates the above dataset and provides a 
# methods for printing the mean and standard deviation and plotting the data.


# i) Build a class that stores and processes data

# Building a class requires some codified procedure. It can look confusing at 
# first but the essentials are simple.
#   1. Start by defining the class name like you would a function
#   2. Provide the rule for how the attributes of the class are initialized
#   3. Then write out any methods

# 1. Define class name
class CustomDataset():
    
    # 2. Initialize class
    # For this, define the __init__ function that takes as inputs some basic data
    # The "self" keyword represents the specific object after invoking it by 
    # calling the class.
    def __init__(self, data):
        self.data = data
        self.n_samples = data.shape[0]
        self.dim_data = data.shape[1]

        # We hereby defined the result of calling e.g. dataset = CustomDataset(data).
        # The "self" keyword is only used during class definition. Replace the 
        # "self" keyword above with your instance name (here: "dataset") to
        # illustrate to yourself the final behavior. After calling dataset = 
        # CustomDataset(data), dataset has the properties dataset.data, datasets.n_samples,
        # and dataset.dim_data.

    # 3. Define methods
    # You do this like you would define a normal function. This function is then
    # associated to the object constructed by this class. It has access to all
    # the information stored inside of the object; in this case these internal
    # functions have access to dataset.data, dataset.n_samples, and dataset.dim_data.
    def print_mean_and_std(self):
        # Print out the mean and the standard deviation of the data. Then integrate
        # them into the dataset
        self.mean = torch.mean(self.data, dim = 0)
        self.std = torch.std(self.data, dim = 0)
        print("Mean = {}, Standard deviation = {}".format(self.mean, self.std))
    
    def plot_data(self):
        # Plot the samples as a simple scatter plot using matplotlib
        plt.scatter(self.data[:,0], self.data[:,1])
        plt.title('Scatterplot of data')


# ii) Instantiate an object and investigate it

# Create an object "dataset" using the blueprint/class CustomDataset
dataset = CustomDataset(data)

# The object dataset has several different attributes now. We print some of them.
print("dataset.n_samples = {} , dataset.dim_data = {}".format(dataset.n_samples, dataset.dim_data))

# As of now, there dont exist entries dataset.mean or dataset.std. However, as
# soon as we call the method "print_mean_and_std", these entries are integrated
# into the object dataset.
dataset.print_mean_and_std()
print("dataset.mean = {}, dataset.std = {}".format(dataset.mean, dataset.std))

# We can also plot now the data by calling the "plot_data" function
dataset.plot_data()



"""
    3. The DataLoader / Dataset classes (data organization)
"""


# Fnctionality for dealing with data like we have introduced above is obviously
# useful. Therefore pytorch offers its own classes for handling data. There
# exist the torch.utils.data.Dataset class and the torch.utils.data.Doataloader
# class. The Dataset class offers some functionality
# Typicall how you use them is by creating a new class that inherits its attributes
# and methods from the pytorch classes and then adding your own customtailored
# methods.


# i) pytorch Dataset class

class MySpecificDataset(torch.utils.data.Dataset):
    # Passing the torch.utils.data.Dataset class as an argument means that the
    # new class MySpecificDataset that we define here will inherit all the attributes
    # and methods from the base class. This action of creating a new class from
    # a basic class is called "subclassing" and its relatively common. Often
    # pytorch classes are meant as "blueprints for blueprints" and you build
    # your own blueprint for an object based on pytorchs basic blueprint.
    
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        # Method for grabbing data based on a provided index
        return self.data[index, :]

    def __len__(self):
        # Method for calculating the number of separate elements in the dataset
        return self.data.shape[0]


# Instantiate and investigate
# The object after instantiation is not super interesting. You can access data
# using my_specific_dataset.data and can call a subset of data by providing an
# index tensor to the data (for this, we defined the __getitem__ method)
my_specific_dataset = MySpecificDataset(data)
indices_subset = torch.tensor([0,1,2])
my_specific_data_subset = my_specific_dataset[indices_subset]   # produces first three datapoints


# ii) pytorch DataLoader class

# We construct an object for loading and serving data by calling pytorchs 
# blueprint for this type of object. As input we provide our dataset and
# the size of the batches we hand during training and test.
my_specific_dataloader = torch.utils.data.DataLoader(my_specific_dataset, batch_size=16, shuffle=True, num_workers=0) 

# The object after instantiation is now a Dataloader that can serve data randomly
# and in batches, among other things. You serve your data to the model for training
# or testing by providing a DataLoader object. Lets check the functionality.
for batch_nr, data_points in enumerate(my_specific_dataloader):
    print("Data points in batch nr {} are {} ".format(batch_nr,data_points))



"""
    4. The Module class (neural nets)
"""


# i) Build an ANN class

class MySpecificANN(torch.nn.Module):
    # We build a class for calling and manipulating neural nets by subclassing pytorchs 
    # blueprint for this type of object.
    
    def __init__(self, dim_hidden):
        super().__init__()
        # Remember: The  __init__() method defines how a class is initialized.
        # Here we declare that the first step of class initialization consists
        # in initializing the class we are inheriting from (torch.nn.Module),
        # the archetypical class from which all ANN's are just specific instances.
        # This is what the super.__init__() statement means - > Initialize the
        # superclass of this subclass.

        # Lets build a transformation that maps 2d space into 2d space. Specify
        # two linear layers and a nonlinearity for that.
        # First, two fully connected linear layers via torch.nn.Linear
        self.fc1 = torch.nn.Linear(dim_data, dim_hidden)
        self.fc2 = torch.nn.Linear(dim_hidden, dim_data)
        
        # Second, setup the sigmoid function as non-linearity
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        # define the forward computation on the input data x
        # first, reshape to have batch dimension on the left
        x = x.reshape([-1,dim_data])
        # second, compute hidden layer
        hidden = self.sigmoid(self.fc1(x))
        # third, transform from hidden to output and return
        x_transformed = self.sigmoid(self.fc2(hidden))
        return x_transformed


# ii) Instantiate and investigate class

my_specific_ann = MySpecificANN(dim_hidden = 5)

# The result is an object of the MySpecificANN class. That is a subclass of the
# torch.nn.Module class which includess all the neural nets and provides us
# with functionalities like printing out the layers, the weights and biases, 
# saving, loading, and of course applying the ann or its layers to some data.

print(" The layers in the ANN 'my_specific_ann' are \n {}".format(my_specific_ann))
print(" The parameters in 'my_specific_ann' are \n {}".format(my_specific_ann.state_dict()))

# Lets apply the ANN to the input data to generate some output data.
# This is done by calling my_specific_ann(x) where x is some data. That is a 
# convenient shortcut for my_specific_ann.forward(x), the forward transform 
# defined above.
y = my_specific_ann(data)

# Note that y is a tensor and therefore all the actions that were taken to 
# transform data to y are stored in the computational graph. See below. This 
# gradient computation is actually what allows us to do the optimization of 
# parameters in a neural net.
grad_fn_1 = y.grad_fn
grad_fn_2 = grad_fn_1.next_functions
print(" Gradients of y are {} (last step), and {} (second to last step) , ... ".format(grad_fn_1, grad_fn_2))



"""
    5. Plots and illustrations
"""


# i) Plot y to illustrate the ANN's impact

my_specific_dataset_transformed = CustomDataset(y.detach())
my_specific_dataset_transformed.plot_data()















