function [Interpolated_grid_vals,Interpolated_grid_vars, Grid_parameters]=Kriging_PPE(Data, Field_values_init,variogram_type)

% [Interpolated_grid_vals,Interpolated_grid_vars, Grid_parameters]=Kriging_PPE(Data, Field_values_init)
%
% The goal of this function is to Compose all of the functions needed for
% Kriging. For this it effectively only needs the data and some initial
% Definition of the dimension of the output matrix. Its output consists of
% the completely interpolated grid, the associated variances and the
% parameters relating pixel numbers and real coordinates
%
% The formats are:
%
%       Data                   = [X_p1 .......        X_pn_data]   (3,n_data)    
%                                [Y_p1 .......        Y_pn_data]
%                                [Z_p1 .......        Z_pn_data]
%       Interpolated_grid_vals = [11 .......         15]              (m,n)    
%                                [09 .......         13]
%                                [..  .......        ..]
%       Interpolated_grid_vars = [11 .......         15]              (m,n)    
%                                [09 .......         13] 
%                                [..  .......        ..] 
%       Grid_parameters      = [min_X max_x delta_X]                  (2,3)
%                              [min_Y max_y delta_Y]
%       Field_values_init    = [NaN ....            ... NaN ]         (n,m)
%                              [NaN ....            ... NaN ]
%        variogram_type       =  Type of variogram function: either 
%                                'exponential' or 'squared_exponential'

% 1: Map data into grid
[ Field_values, Grid_parameters, coord_to_index, Data_to_estimate]...
    =From_list_to_grid(Data,Field_values_init);

% 2: Make experimental semivariogram and fit a model to it
Exp_sem=Make_experimental_semivariogram(Data);
semivar_fun=Fit_model_to_data(Exp_sem,variogram_type);

% 3: Create some Difference matrices allowing for later calculation of A and C
% from the differences and the model
Point_List_p=[Data(1,:);Data(2,:)];
Point_List_s=[Data_to_estimate(1,:);Data_to_estimate(2,:)];
Difference_matrix_pp=Make_Distance_Matrix(Point_List_p,Point_List_p);
Difference_matrix_sp=Make_Distance_Matrix(Point_List_p,Point_List_s);

% 4: Apply the model to the differences to derive the distance dependent
% semivariances, then solve the Kriging system
[A, A_cov, C, C_cov ,Z]=Make_Kriging_System(Data, ...
    Difference_matrix_pp, Difference_matrix_sp, semivar_fun);
[Data_estimated]=Solve_Kriging_System(Data_to_estimate, A, A_cov, C, C_cov ,Z);

% 5. map data from list into grid
[Interpolated_grid_vals]=(Write_list_in_grid(Field_values,coord_to_index,Data_estimated,3))';
[Interpolated_grid_vars]=(Write_list_in_grid(Field_values_init,coord_to_index,Data_estimated,4))';

end