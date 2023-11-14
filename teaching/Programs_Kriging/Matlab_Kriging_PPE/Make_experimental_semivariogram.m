function [experimental_semivariogram]=Make_experimental_semivariogram(Data)

% [experimental_semivariogram]=Make_experimental_semivariogram(Data)
%
% This function takes as an input the (3,n_data) matrix Data containing the
% coordinates of the points and the values at those points.
% The output is the empirical variogram, meaning a (2,n_bin) matrix
% containing in the first row the lag bins and in the second row the
% corresponding estimated semivariances.
% 
% The approach we follow here ist
% 1a: Make the distance Matrix and 
% 1b: distribute the distances into bins
% 2: Calculate the differences delta between values of pairs of points
% 3: Define a function that maps sets of delta's to a single semivariance
%    estimation (deltas_to_semivariance)
% 4: Calculate everything for accumarray and us it to map the differences
%    to corresponding bins before applying deltas_to_semivariance to every 
%    bin content.
% 5: Reformat everything, such that empirical_variogram is in the form
%    specified below.
%
% The formats are:
%       Data                 = [X_p1 .......         X_pn_data]   (3,n_data)    
%                              [Y_p1 .......         Y_pn_data]
%                              [Z_p1 .......         Z_pn_data]
%       empirical_variogram  = [Lag_bin1 .......  lag_binn_bin]   (2,n_bin)    
%                              [semi_bin1  ....  semi_binn_bin]

% 1a: ------------

% Get the number of data points
n_data=size(Data,2);

% Make the distance matrix. For that use only the coordinate values.
Dist_mat=Make_Distance_Matrix(Data([1, 2],:),Data([1,2],:));

% 1b: ------------

% Define bin_width; Use Rice rule for that and double the amount. the
% doubling is done, because in the end we want to cut at half the maximum
% distance but still want to have the right amount of bins.
% If you are unsatisfied with the bin size and want something more dynamic/
% a higher resolution you can modify this rule.
% Different rules below:
% n_bins=round(2*(n_data)^(1/3));    Rice Rule
% n_bins=2*sqrt(n_data);             Sqrt Rule
% n_bins=1+log2(n_data);             Sturges Rule
% n_bins=n_data;
n_bins=2*sqrt(n_data);
bin_width= max(max(Dist_mat))/(n_bins);

% Define function dist_to_bin that assigns a distance its corresponding bin
dist_to_bin= @(x) round(x/bin_width)+1;
bin_to_dist= @(x) (x-1)*bin_width;

% Apply function to map distances to distance bins
Dist_bins=arrayfun(dist_to_bin,Dist_mat);

% 2: ------------
% Calculate distance between Z values of all pairs of points
Delta_mat=Make_Distance_Matrix(Data(3,:),Data(3,:));

% 3: ------------
% Define function mapping sets of deltas to semivariance estimations
deltas_to_semivariance= @(x) (1/(2*numel(x)))*sum(x.^2);

% 4: ------------
% Prepare everything for accumarray calculation
Dist_bins=Dist_bins(:);
Delta_mat=Delta_mat(:);
Semivariances=accumarray(Dist_bins,Delta_mat,[],deltas_to_semivariance,NaN);
Semivariances=Semivariances';
bin_vector=1:numel(Semivariances);

% 5: ------------
% Reformat everything
lagvector=arrayfun(bin_to_dist, bin_vector);
experimental_semivariogram=[lagvector(1:round(end/2)); Semivariances(1:round(end/2))];



end