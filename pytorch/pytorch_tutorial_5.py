#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is part of a series of pytorch tutorials. During these tutorials,
we will learn about tensors (part_1), use autograd and optimizers on a minimal
example (part_2), use torch for fitting a basic model via LS (part_3), learn
about pytorchs utility functions (part_4) and train a neural network to perform
a classification task (part_5). 

This script illustrates how pytorch can be used for a slghtly more complex task
that includes classifying timeseries. We will use the more object-oriented
approach to programming introduced in part_4 to create an object that integrates
data, neural networks and training routines into one consistent package.
For this, do the following:
    1. Imports and definitions
    2. Simulate some data and labels
    3. Define blueprint for a Classifier object
    4. Optimize parameters
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


# ii) Definitions

n_timeseries = 100
n_time = 24
n_classes = 2
t = torch.linspace(0,24,n_time)


"""
    2. Simulate some data and labels
"""


# We will generate two types of timeseries. The timeseries represent measurements
# of some quantity that is indicative of the health of some structural object
# - think of the deformations of a bridge over a day for example.
# The two types of timeseries are generated according to two different rules 
#   rule 1 : first and last element of timeseries are very similar
#               -> Bridges with this behavior are healthy
#   rule 2 : first and last element of timeseries are different
#               -> Bridges with this behavior are damaged
#
# This rule symbolizing the presence nonelastic irreversible deformations is 
# used to generate our data. It is not accessible to the classifier that we want
# to train. The classifier will only see the timeseries and a label 0 (healthy)
# or 1 (damaged) and will have to infer the underlying rule itself. This mimicks
# the real-world situation where only data is available but not the underlying 
# process that generated it.


# i) Define timeseries generation_process

sigma_noise = 0.3

def generate_timeseries(rule):
    # Generate a timeseries for healthy (rule = 0) or damaged (rule = 1) bridges.
    
    # Create the mean trajectory
    if rule == 0:
        mean = (1/50)*(-(t - 12)**2 - t + 12**2) + torch.normal(0,1, [1])
    elif rule == 1:
        mean = (1/50)*(-(t - 12)**2 + 12**2) + torch.normal(0,1, [1])
    else:
        raise Exception('Argument rule needs to be 0 or 1 but is {}'.format(rule))
    
    # Add some noise
    noisy_timeseries = mean + torch.normal(0, sigma_noise, [n_time])
    
    return noisy_timeseries


# ii) Generate timeseries

timeseries_data = torch.zeros([n_timeseries, n_time])                  # Note: left dim = batch_dim
generation_rules = torch.randint(0, 2, [n_timeseries])
for k in range(n_timeseries):
    timeseries_data[k,:] = generate_timeseries(generation_rules[k])
labels = generation_rules


# iii) Compile TimeseriesDataset

class ClassificationDataset(torch.utils.data.Dataset):
    # Build dataset class for the classification data  
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __getitem__(self, index):
        # Method for grabbing data and labels based on an index
        return self.data[index, :], self.labels[index]

    def __len__(self):
        # Method for calculating the number of separate elements in the dataset
        return self.data.shape[0]

classification_dataset = ClassificationDataset(timeseries_data, labels)


# iv) Integrate into data loader

# Split the dataset into training and testing sets
train_size = int(round(0.8 * len(classification_dataset)))  # 80% of the data for training
test_size = len(classification_dataset) - train_size  # 20% of the data for testing
train_dataset, test_dataset = torch.utils.data.random_split(classification_dataset, [train_size, test_size])

# Create DataLoaders for training and testing sets
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

# Illustrate train DataLoader and test DataLoader
print("Training Batches:")
for batch_idx, (data_batch, labels_batch) in enumerate(train_loader):
    print(f"Batch {batch_idx + 1} - Data Shape: {data_batch.shape}, Labels Shape: {labels_batch.shape}")

print("\nTesting Batches:")
for batch_idx, (data_batch, labels_batch) in enumerate(test_loader):
    print(f"Batch {batch_idx + 1} - Data Shape: {data_batch.shape}, Labels Shape: {labels_batch.shape}")



"""
    3. Define blueprint for a Classifier object
"""


# i) Define ANN constructor

class ANN(torch.nn.Module):
    def __init__(self, dim_hidden):
        # Initialize ANN class by initializing the superclass which inherits its
        # attributes and methods to the ANN class; this is the torch.nn.Module class
        # that contains attributes and methods for neural nets.
        super().__init__()
        self.dim_input = n_time
        self.dim_output = 2
        self.dim_hidden = dim_hidden
        
        # linear transforms
        self.fc_1 = torch.nn.Linear(n_time, dim_hidden)
        self.fc_2 = torch.nn.Linear(dim_hidden, dim_hidden)
        self.fc_3 = torch.nn.Linear(dim_hidden, n_classes)
        # nonlinear transforms - produces positive numbers
        self.nonlinear = torch.nn.Sigmoid()
        
    def forward(self, timeseries_data):
        # Define forward computation on the input timeseries_data
        # Shape the minibatch so that batch_dims are on left
        timeseries_data = timeseries_data.reshape([-1, n_time])
        
        # Then compute hidden units and output of nonlinear pass
        hidden_units_1 = self.nonlinear(self.fc_1(timeseries_data))
        hidden_units_2 = self.nonlinear(self.fc_2(hidden_units_1))
        class_energy = self.nonlinear(self.fc_3(hidden_units_2))
        # The output is for each timeseries a vector of dim n_classes  = 2 featuring
        # positive entries that could be interpreted as class energies = nonnormalized
        # class probabilities.
        return class_energy


# i) Define ClassifierObject class

class ClassifierObject():
    # For initialization, pass network architecture
    def __init__(self, dim_hidden):
        self.ann = ANN(dim_hidden)
        self.loss_fn = torch.nn.CrossEntropyLoss()
    
    # functionality for classification
    def classify(self, timeseries_data):
        unnormalized_class_probs = self.ann(timeseries_data)
        max_prob, index_max_prob = torch.max(unnormalized_class_probs, 1)
        return index_max_prob
    
    # functionality for calculating the loss
    def calculate_loss(self, timeseries_data, labels):
        # CrossEntropy loss function handles labels and unnormalized class probabilities
        unnormalized_class_probs = self.ann(timeseries_data)
        loss = self.loss_fn(unnormalized_class_probs, labels)
        return loss
    
    # functionalty for illustrating classification
    def illustrate(self, timeseries_data, true_labels, title):
        predictions = self.classify(timeseries_data)
        plt.figure(figsize = [15,5], dpi = 300)
        plt.scatter([k for k in range(n_timeseries)], predictions.detach() + 0.1*torch.ones(size = [n_timeseries]), label = 'predicted class')
        plt.scatter([k for k in range(n_timeseries)], true_labels.detach(), label = 'true class')
        plt.title(title)
        plt.legend()
        plt.show()

# Invoke object and illustrate initial performance
classifier_object = ClassifierObject(10)
classifier_object.illustrate(timeseries_data, labels, 'Untrained model prediction performance')




"""    
    4. Optimize parameters
"""


 # i) Set up optimizer
optimizer = torch.optim.Adam(classifier_object.ann.parameters(), lr=0.0001)      
loss_history = []


# ii) Optimize

for epoch in range(5000):
    # cycle through the train loader
    for batch_idx, (data_batch, labels_batch) in enumerate(train_loader):
        # Set the gradients to zero 
        optimizer.zero_grad()  
        
        # compute the loss function
        loss = classifier_object.calculate_loss(data_batch, labels_batch) 
        
        # compute the gradients
        loss.backward()
        # update the weights, record new parameters and loss
        optimizer.step()
        loss_history.append(loss.item())
        
    # print the loss value at specifix steps
    if epoch % 100 == 0:
        print("Epoch {}, Loss = {}".format(epoch+1, loss.item()))



"""
    5. Plots and illustrations
"""


# i) Illustrate the performance on testing dataset

# Iterate through the test loader and compute wrong labels
print("\nTesting Batches:")
labels_pred = []
labels_true = []
for batch_idx, (data_batch, labels_batch_true) in enumerate(test_loader):
    labels_true.append(labels_batch_true)
    labels_batch_pred = classifier_object.classify(data_batch)
    labels_pred.append(labels_batch_pred)
    
# Aggregate results, then print and plot
labels_true = torch.cat(labels_true,0)    
labels_pred = torch.cat(labels_pred,0) 
n_wrong_labels = torch.sum(torch.abs(labels_true - labels_pred))
print("Data in test set = {} timeseries, wrong labels = {}".format(test_size, n_wrong_labels))
    
classifier_object.illustrate(timeseries_data, labels, 'Trained model prediction performance')
    

# ii) Illustrate max impact features

# Figure out which input tensor is needed to maximally trigger the probabilities
# for label 0 or label 1. Do this by numerically optimizing an input tensor to
# maximize the class_energies predicted by classifier_object.ann
    
def maximum_impact_features(label = 0):
    # Calculate the timeseries triggering highest class_energy for input label.
    # Initialize timeseries and optimizer
    timeseries_max_impact = torch.zeros(size = [1,n_time], requires_grad = True)
    optimizer_max_impact = torch.optim.Adam([timeseries_max_impact], lr=0.01)
    loss_max_impact_history = []
    
    for step in range(5000):
        # Calculate loss and take optimization step
        optimizer_max_impact.zero_grad()
        loss_max_impact = -classifier_object.ann(timeseries_max_impact)[0,label]
        loss_max_impact.backward()
        optimizer_max_impact.step()
        loss_max_impact_history.append(loss_max_impact.item())
        
        # print the loss value at specifix steps
        if step % 100 == 0:
            print("Epoch {}, Loss = {}".format(step+1, loss_max_impact.item()))
    
    # Plot the timeseries
    plt.figure(num =2, figsize = (10,5), dpi = 300)
    plt.plot(timeseries_max_impact.detach().T)
    plt.title('Timeseries that triggers label {}'.format(label))
    plt.show()
    
maximum_impact_features(label = 0)   
    
    
    