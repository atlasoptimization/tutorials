"""
The goal of this set of functions is to provide auxiliary functions for Kriging
as needed for the course "Project parameter estimation" at ETH.
The following functions are provided:
    1. Make_experimental_semivariogram: Take a list, make out of it an experimental 
        semivariogram
    2. Fit_model_to_data: Fit a parametric model to an experimental semivariogram
    3. From_List_to_grid: Take a list and a NaN-filled matrix, embed the list
        entries into the matrix
    4. Write_list_in_grid: Write a list directly into a grid, similar to 
        From_List_to_grid but a slightly different use case
    5. From_grid_to_list: Take a grid and unravel the data back into a list
    6. Make_comparative_interpolation: Generate interpolation results with 
        alternative methods
    7. Make_Distance_matrix: Take a list of coordinates and create the pairwise
        distances

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as scopt
from scipy.interpolate import griddata


"""
1. Make_experimental_semivariogram -------------------------------------------
"""




def Make_experimental_semivariogram(Data):

    """
    
     This function takes as an input the (3,n_data) matrix Data containing the
     coordinates of the points and the values at those points.
     The output is the empirical variogram, meaning a (2,n_bin) matrix
     containing in the first row the lag bins and in the second row the
     corresponding estimated semivariances.
     
     The approach we follow here ist
     1a: Make the distance Matrix and 
     1b: distribute the distances into bins
     2: Calculate the differences delta between values of pairs of points
     3: Define a function that maps sets of delta's to a single semivariance
        estimation (deltas_to_semivariance)
     4: Calculate everything for accumarray and us it to map the differences
        to corresponding bins before applying deltas_to_semivariance to every 
        bin content.
     5: Reformat everything, such that empirical_variogram is in the form
        specified below.
    
     The formats are:
           Data                 = [X_p1 .......         X_pn_data]   (3,n_data)    
                                  [Y_p1 .......         Y_pn_data]
                                  [Z_p1 .......         Z_pn_data]
           empirical_variogram  = [Lag_bin1 .......  lag_binn_bin]   (2,n_bin)    
                                  [semi_bin1  ....  semi_binn_bin]
    """
    
     # 1a: ------------
    
     # Get the number of data points
    n_data=np.shape(Data)[1]
    
     # Make the distance matrix. For that use only the coordinate values.
    Dist_mat=Make_distance_matrix(Data[[0, 1],:],Data[[0,1],:])
    
     # 1b: ------------
    
     # Define bin_width; Use Rice rule for that and double the amount. the
     # doubling is done, because in the end we want to cut at half the maximum
     # distance but still want to have the right amount of bins.
     # If you are unsatisfied with the bin size and want something more dynamic/
     # a higher resolution you can modify this rule.
     # Different rules below:
     # n_bins=round(2*(n_data)^(1/3));    Rice Rule
     # n_bins=2*sqrt(n_data);             Sqrt Rule
     # n_bins=1+log2(n_data);             Sturges Rule
     # n_bins=n_data;
    n_bins=np.round(4*np.sqrt(n_data)).astype(int)
    bin_width= np.max(Dist_mat)/(n_bins)
    bin_vec=np.linspace(0,n_bins,n_bins+1)
    
     # Define function dist_to_bin that assigns a distance its corresponding bin
    dist_to_bin= lambda x: np.round(x/bin_width)
    bin_to_dist= lambda x: (x)*bin_width
    
     # Apply function to map distances to distance bins
    Dist_bins=dist_to_bin(Dist_mat)
    
     # 2: ------------
     # Calculate distance between Z values of all pairs of points
    Delta_mat=Make_distance_matrix(np.reshape(Data[2,:],[1,n_data]),np.reshape(Data[2,:],[1,n_data]))
    
     # 3: ------------
     # Define function mapping sets of deltas to semivariance estimations
    deltas_to_semivariance= lambda x: (np.linalg.pinv(np.array([[np.max([2*np.size(x),1])]])))*np.sum(x**2)
    
     # 4: ------------
     # Prepare everything for accumulation
    Dist_bins=Dist_bins.flatten()
    Delta_mat=Delta_mat.flatten()
    
    bin_indices = np.digitize(Dist_bins, bin_vec,right=True)
    Semivariances=np.zeros([n_bins])
    for k in range(1,n_bins):
        Semivariances[k]=deltas_to_semivariance(Delta_mat[bin_indices==k])
        
 
    
     # 5: ------------
     # Reformat everything
    lagvector=bin_to_dist(bin_vec[0:-1])
    experimental_semivariogram=np.vstack((lagvector[0:np.round(n_bins/2).astype(int)], Semivariances[0:np.round(n_bins/2).astype(int)]))

    return experimental_semivariogram
    
    
"""
2. Fit_model_to_data: --------------------------------------------------------
""" 


def Fit_model_to_data(experimental_semivariogram,var_type):

    """
      This function takes as input the experimental semivariogram consisting of
      the lags and associated semivariance estimates. It outputs a model for
      the semivariogram function. This model is found by assuming that the
      underlying true semivariogram is of type "exponential" and then
      by finding those model parameters that minimize an objective function
      similar to the squared residuals.
    
      The formats are:
                                   
      experimental_semivariogram  = [Lag_bin1 .......  lag_binn_bin]   (2,n_bin)
                                    [semi_bin1  ....  semi_binn_bin]
            variogram_type       =  Type of variogram function: either 
                                    'exponential' , 'squared_exponential' or 
                                      'spherical'
            semivar_function     =  function handle
    """
    
      # Take internal short name and delete empty bins
    exp_sem=experimental_semivariogram
    lag_exp=exp_sem[0,:]
    
      # First define the model to fit into our experimental semivariogram. This 
      # Function takes as input a row vector params =[range,sill] and the lag at
      # which to evaluate the function.
    
    if var_type=='exponential':
        semivar_fun_init = lambda params,lag: params[1]*(1-np.exp(-np.abs(lag)/params[0]))
    elif var_type=='squared_exponential':
        semivar_fun_init = lambda params,lag: params[1]*(1-np.exp(-(lag/params[0])**2))
    elif var_type=='spherical':
        semivar_fun_init = lambda params,lag: params[1]*((3*lag)/(2*params[0])-(lag**3)/(2*params[0]**3))*(lag<params[0])+params[1]*(lag>=params[0]);
    else:
        print('Unknown Convariance function. Choose either "exponential" or "squared exponential"')

    
      # Prepare everything for optimization step. This includes finding some 
      # initial starting parameters somewhat close to the solution and defining
      # the objective function to be minimized.
    
      # Initial value sill and range
    params_init=np.zeros([2])
    params_init[1]=np.max(exp_sem[1,:])                       # just take the maximum
    params_init[0]=np.where(exp_sem[1,:]>=0.5*params_init[1])[0][0]  # lag, where first hit 0.5 the maximum
    
                                      
      # Please note that there are better and stochastically justified of fitting
      # for semivariance functions than simple least squares. More professional
      # are the versions in the textbooks from Cressie or Chiles & Delfiner
    
    
    objectfun = lambda params: np.sum(((semivar_fun_init(params,lag_exp)-exp_sem[1,:])**2))
    
      # Use scipy.optimize to find those parameters [range,sill]=params_fin that
      # minimize the objective function. Display error messages.
    optimization_outcome = scopt.minimize(objectfun, params_init)
    print(optimization_outcome)
    
    params_fin=optimization_outcome.x
    
      # Define the model by using the optimizing parameters
    if var_type=='exponential':
        semivar_fun = lambda lag: params_fin[1]*(1-np.exp(-np.abs(lag)/params_fin[0]))
    elif var_type=='squared_exponential':
        semivar_fun = lambda lag: params_fin[1]*(1-np.exp(-(lag/params_fin[0])**2))
    elif var_type=='spherical':
        semivar_fun = lambda lag: params_fin[1]*((3*lag)/(2*params_fin[0])-(lag**3)/(2*params_fin[0]**3))*(lag<params_fin[0])+params_fin[1]*(lag>=params_fin[0]);
    else:
        print('Unknown Convariance function. Choose either "exponential" or "squared exponential"')

    
    return semivar_fun
    
    
    """
3. From_List_to_grid ----------------------------------------------------------
"""

def From_list_to_grid(Data,Field_values_init):

    
    """
      This function takes as Inputs two Matrices containing the data to be
      placed in a Grid (Data, (3,n_data)) and a Matrix "Field_values_init"
      (n,m) filled with NaN's serving as an initialization for the embedding
      of the list into a grid.
    
      Output 1 consists of a grid "Field_values". It is scaled, s.t. all the
      locations associated with the data lie inside and its entries are NaN,
      if there is no data for this location and Z(X,Y) otherwise. Scaling and
      finding the corresponding values is done with the help of a mapping
      function, which clears up the relation between grid points and
      coordinates.
      Output 2 "Grid_parameters" gives some hints on how this mapping function
      looks like by archiving which coordinate difference corresponds to which
      step size in the matrix.
      Output 3 "coord_to_index" is a function handle representing the function
      which maps coordinates onto indices.
      Output 4 "Data_to_estimate" is a matrix containing all the coordinates of
      those grid points, for which there is no data available. The coordinates
      of those grid_points will be use for interpolation later on.
    
      The formats are:
            Data_to_estimate     = [X_s1 .......        X_sn_est]    (2,n_est)
                                  [Y_s1 .......        Y_sn_est]
            Data                 = [X_p1 .......        X_pn_data]   (3,n_data)
                                  [Y_p1 .......        Y_pn_data]
                                  [Z_p1 .......        Z_pn_data]
            Field_values_init    = [NaN ....            ... NaN ]         (n,m)
                                  [NaN ....            ... NaN ]
            Field_values         = [NaN ....    Z(X_13) ... NaN ]         (n,m)
                                  [NaN ....       ... Z(X_n,m) ]
            Grid_parameters      = [min_X max_x delta_X]                  (2,3)
                                  [min_Y max_y delta_Y]
            coord_to_index       = function handle
    """
    
      # Get size of the Matrix, in which the list should be embedded
    n,m=np.shape(Field_values_init)
    
      # Extract maximum and minimum coordinates from data and plug them into
      # matrix minmax_mat = [x_min  x_max ]
      #                     [y_min  y_max ]
      #                     [z_min  z_max ]
    
    minmax_mat=np.zeros([3,2])
    minmax_mat[:,0]=np.min(Data,1)
    minmax_mat[:,1]=np.max(Data,1)
    
      # Make the mapping from coordinates to grid points coordinate_to_index and
      # its inverse index_to_coordinate
    
      # 1. Find the step sizes and archive everything in "Grid_parameters"
    delta_y=(minmax_mat[1,1]-minmax_mat[1,0])/(n-1)
    delta_x=(minmax_mat[0,1]-minmax_mat[0,0])/(m-1)
    Grid_parameters=np.array([[minmax_mat[0,0],minmax_mat[0,1], delta_x,], [minmax_mat[1,0],minmax_mat[1,1], delta_y]])
    
    
      # 2. Make index_to_coord. This is a function handle working with the syntax
      # [x,y]=index_to_coord(i,j)
    index_to_coord=lambda i,j : np.array([minmax_mat[0,0]+delta_x*(j),minmax_mat[1,0]+delta_y*(i)])
    
      # 3. Make coord_to_index. This is a function handle working with the syntax
      # [i,j]=coord_to_index(x,y)
    coord_to_index=lambda x,y: np.array([np.round((y-minmax_mat[1,0])/(delta_y)),np.round((x-minmax_mat[0,0])/(delta_x))])
    
      # Extract coordinates from data and use mapping to find corresponding
      # indices. Write the values from the data points into the grid at the
      # corresponding locations
    Field_values=Write_list_in_grid(Field_values_init,coord_to_index,Data,2)
    
      # Extract the indices of those Grid points, where there is no data.
    Nan_index=np.where(np.isnan(Field_values))
    n_nan=Nan_index[0].shape[0]
    
      # Calculate those coordinates and write them into the Matrix designating
      # them for later estimation
    Data_to_estimate=np.zeros([2,n_nan])
    for k in range(n_nan):
        [x,y]=index_to_coord(Nan_index[0][k], Nan_index[1][k])
        Data_to_estimate[:,k]=np.array([x,y]).T
    
    
    
    return Field_values, Grid_parameters, Data_to_estimate, coord_to_index,index_to_coord









"""
4. Write_list_in_grid --------------------------------------------------------
"""


def Write_list_in_grid(Grid,coord_to_index,Data_list,Row_data):
    
    """
      This function takes as input a list consisting of coordinates and values,
      a grid into which these values should be written as well as the necessary
      function (handle) that performs the mapping from coordinates to indices.
      This function is normally made by "From_list_to_grid".
      Row data is just an index specifying in which row of the Data_list the
      actual values lie that should be mapped on the grid.
    
      The output of this function is just a grid "Filled_grid" of exactly the
      same dimension as the Grid but now with additionally the values injected,
      that are specified in Data_list.
    
      The formats are:
            Grid                 = [NaN .......          Z_p95   ]        (n,m)    
                                  [NaN ... Z_p43             NaN]
            Data_list            = [X_s1 .......         X_sn_est]   (3,n_data)    
                                  [Y_s1 .......         Y_sn_est]
                                  [Z_s1 .......         Z_sn_est]
                                  [sigma_s1 .......  sigma_n_est]
            Filled_Grid          = [Z_s1 ....   Z_s12   ... Z_p95]        (n,m)
                                  [Z_s24 ....   Z_p43  Z_sn_est ]
            coordinate_to_index  = function handle
    """
    
    # Extract coordinates from Data_list and use mapping to find corresponding
    # indices.
    
    n_data=Data_list.shape[1]
    index_list=np.zeros([n_data,2])
    
    for k in range(n_data):
        [i,j]=coord_to_index(Data_list[0,k],Data_list[1,k])
        index_list[k,[0,1]]=[i,j]
    
    
    # Fill the grid by accessing the elements corresponding to data points and
    # filling them with the corresponding data
    Filled_grid=Grid.flatten()
    Filled_grid[np.ravel_multi_index(index_list.T.astype(int), Grid.shape)]=Data_list[Row_data,:]
    Filled_grid=np.reshape(Filled_grid,Grid.shape)    
    
    return Filled_grid


"""
5. From_grid_to_list ----------------------------------------------------------
"""

def From_grid_to_list(Grid_data):
    """
      This simple auxiliary function takes as input a Matrix Grid_data
      containing mostly NaNs and at some points measurement values.
      It will extract the row and column indices as well as the values of the
      measurements and put them into a List "Data", which can serve as input to
      Kriging_PPE.m
     
      The formats are:
            Grid_data            = [NaN .......         15]          (n,m)    
                                  [NaN .......        NaN]
                                  [12  .......        NaN]
                                  [NaN ......11       NaN]
            Data                 = [X_p1 .......        X_pn_data]   (3,n_data)    
                                  [Y_p1 .......        Y_pn_data]
                                  [Z_p1 .......        Z_pn_data]
    """
    
    [rows, cols]=np.where(np.isnan(Grid_data)==0)
    vals=np.zeros([rows.shape[0]]);
    for k in range(rows.shape[0]):
        vals[k]=Grid_data[rows[k],cols[k]]

    Data_list= np.vstack((rows, cols, vals))
    
    return Data_list



"""
6. Make_comparative_interpolation ---------------------------------------------
"""
    
def Make_comparative_interpolation(Data, Field_values_init):

      # 1: Convert Data list to grid and back to get rows and cols instead of
      # coordinates
    Data_grid,_,_,_,_ =From_list_to_grid(Data,Field_values_init)
    Data_list=From_grid_to_list(Data_grid)
    
    y=Data_list[0,:]
    x=Data_list[1,:]
    z=Data_list[2,:]
    
      # 2: Apply interpolation functions
    [n_y,n_x]=Data_grid.shape
    [Y, X] = np.mgrid[0:n_y,0:n_x]
    Cubic=griddata((x,y),z,(Y,X), method='cubic')
    Linear=griddata((x,y),z,(Y,X), method='linear')
    Nearest_Neighbor=griddata((x,y),z,(Y,X), method='nearest')
    
    return Nearest_Neighbor, Linear, Cubic


"""
7. Make_distance_matrix ---------------------------------------------
"""

def Make_distance_matrix(Point_list_1,Point_list_2):

    """
     This function takes as Input two Matrices containing Point Coordinates. 
     The output of this function is a Distance_Matrix containing at position
     (i,j) the distance between Point i in Point_List 1 and Point j in 
     Point_List 2.
     The input Lists should have the form: 
             PL 1                                    PL 2
     [X_s1   X_s2  X_s3 ... X_sn_est]          [X_p1   X_p2  X_p3 ... X_pn_data]
     [Y_s1   Y_s2  Y_s3 ... Y_sn_est]          [Y_p1   Y_p2  Y_p3 ... Y_pn_data]
     [...    ...    ... ...  ...    ]          [...    ...    ... ...      ... ]
    
     In the case of 2 dimensional Kriging these Lists are just
             PL 1                                    PL 2
     [X_s1   X_s2  X_s3 ... X_sn_est]          [X_1   X_2  X_3 ... X_n_data]
     [Y_s1   Y_s2  Y_s3 ... Y_sn_est]          [Y_1   Y_2  Y_3 ... Y_n_data]
    
     The distance between every point s_i (to be estimated) with coordinates 
     (X_si,Y_si) to every data point p_i (used for estimation) is calculated.
    
    """
    
    n_est=np.shape(Point_list_1)[1]
    n_data=np.shape(Point_list_2)[1]
    distance_matrix=np.zeros([n_est,n_data])
    
    for k in range(n_est):
        for l in range(n_data):
            distance_matrix[k,l]=np.linalg.norm(Point_list_1[:,k]-Point_list_2[:,l])
            
    return distance_matrix
    
    





















