function [A, A_cov, C, C_cov, Z]=Make_Kriging_System(Data, Difference_matrix_pp, Difference_matrix_ps, Gamma_model)

% [Data_estimated]=Make_Kriging_System(Data,Difference_matrix_pp, Difference_matrix_sp,, Gamma_Model)
% This function takes as input the Matrix of (vector or scalar) differences
% between the Locations s_i and the locations p_j to calculate the 
% semivariance between them using the theoretical semivariogram model 
% specified in "Gamma_Model".
%
% The outputs are 
% the Matrix A detailing the semivariances between the data points p_j 
% (where there are already measurements) (+A_cov)
% The Matrix C detailing the semivariances between the RV's at s_i (which 
%  are to be estimated) and the RV's at p_j (+C_cov)
% The Vector Z contains just the realizations of the RV's. Z_j= Z(p_j)
% The formats are:
%       Data_to_estimate     = [X_s1 .......        X_sn_est]    (2,n_est)    
%                              [Y_s1 .......        Y_sn_est]
%       Data                 = [X_p1 .......        X_pn_data]   (3,n_data)    
%                              [Y_p1 .......        Y_pn_data]
%                              [Z_p1 .......        Z_pn_data]
%       Difference_Matrix_pp = [ diff(pi,pj) ]                   (n_data,n_data)
%       Difference_Matrix_sp = [ diff(si,pj) ]                   (n_est,n_data)
%       Gamma_model          = function handle
%
%       A                    = [gamma(pi,pj) 1]                  (n_data+1,n_data+1)
%                              [    1        0]
%       C                    = [gamma(s_i,pj) ]                  (n_data+1,n_est+1)
%                              [       1      ]
%       Z                    = [    Z_pj      ]                  (n_data+1)

% Make Vector Z containing the Measurements
Z=Data(3,:)';

% Make A by applying the Semivariance function to each entry in the
% difference Matrix between points Difference_matrix_pp and adding the rows
% and columns guaranteeing unbiasedness
A_core= arrayfun(Gamma_model,Difference_matrix_pp);
A=[A_core;ones(1,size(A_core,2))];
A=[A,ones(size(A,1),1)];
A(size(A,1),size(A,2))=0;

% Make A_cov the same way but use sigma(X,Y)=sigma(X,X)-gamma(X,Y)
A_cov=var(Z)*ones(size(A_core))-A_core;
A_cov=[A_cov;ones(1,size(A_cov,2))];
A_cov=[A_cov,ones(size(A_cov,1),1)];
A_cov(size(A_cov,1),size(A_cov,2))=0;


% Make C by applying the Semivariance function to each entry in the mixed
% difference Matrix Difference_matrix_sp and adding a row of ones
C_core=arrayfun(Gamma_model,Difference_matrix_ps);
C=[C_core;ones(1,size(C_core,2))];

% Make C_core the same way but use sigma(X,Y)=sigma(X,X)-gamma(X,Y)
C_cov=var(Z)*ones(size(C_core))-C_core;
C_cov=[C_cov;ones(1,size(C_core,2))];

end