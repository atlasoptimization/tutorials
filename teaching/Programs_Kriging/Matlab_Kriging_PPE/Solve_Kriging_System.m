function [Data_estimated]=Solve_Kriging_System(Data_to_estimate, A, A_cov, C, C_cov ,Z)

% [Data_estimated]=Solve_Kriging_System(Data_to_estimate, A, A_cov, C, C_cov ,Z)
% This function takes as input the locations for which an estimation should
% be generated (Data_to_estimate) as well as the values of the predictor RV's.
% Also it needs 
% the Matrix A detailing the semivariances between the data points p_j 
%  (where there are already measurements)
% The Matrix C detailing the semivariances between the RV's at s_i (which 
%  are to be estimated) and the RV's at p_j
% The Vector Z containing just the realizations of the RV's. Z_j= Z(p_j)
%
% The output is the Matrix Data_estimated. It contains the locations at
% which Data was to be estimated as well as the values of this estimation
% together with confidence bounds in form of the estimated error variance
%
% The formats are:
%       Data_to_estimate     = [X_s1 .......        X_sn_est]    (2,n_est)    
%                              [Y_s1 .......        Y_sn_est]
%       A                    = [gamma(pi,pj) 1]                  (n_data+1,n_data+1)
%                              [    1        0]
%       C                    = [gamma(s_i,pj) ]                  (n_data+1,n_est+1)
%                              [       1      ]
%       Z                    = [    Z_pj      ]                  (n_data+1)
%
%       Data_estimated       = [X_s1 .......        X_sn_est]    (4,n_est)    
%                              [Y_s1 .......        Y_sn_est]
%                              [Z_s1 .......        Z_sn_est]
%                              [sigma_s1 ...... sigma_sn_est]

% Calculate the optimal weights lambda. Each column vector lambda_j is one
% set of optimal weights usable for predicting Z_sj. Delete lagrange
% multiplier
lambda=pinv(A)*C;
lambda(size(lambda,1),:)=[];

% Calculate now the predicted Z_sj for all j
Z_hat=lambda'*Z;

% Calculate now the estimated error variance sigma_e. Delete for this the
% parts of C_cov and A_cov responsible for unbiasedness.
A_cov(size(A_cov,1),:)=[];A_cov(:,size(A_cov,2))=[]; C_cov(size(C_cov,1),:)=[];
sigma_e=diag(lambda'*A_cov*lambda)-2*diag(lambda'*C_cov)+var(Z);

% Fill the Data into the Matrix "Data_estimated"
Data_estimated=Data_to_estimate;
Data_estimated=[Data_estimated ; Z_hat'];
Data_estimated=[Data_estimated; sigma_e'];





end
