function [Sample_grid, Grid]=Grid_from_function(function_handle,m,n, sample_num)

% [Sampled_grid, Grid]=Grid_from_function(function_handle,m,n,sample_num)
%
% This small auxiliary function takes as an input a function handle and
% applies it to a matrix of dimension [n,m]. The idea is, that the function
% handle takes as input coordinates the row and column numbers of the
% matrix and produces in that way some exemplary datasets we can use in
% PPE. The sample_num input specifies, how many samples are drawn at random
% from the grid. Those serve as Input for Kriging_PPE.m
% The output is given by the whole grid "Grid" and the grid only containing
% values at randomly specified points ("Sample_grid").
%
% The formats are:
%
%       function_handle      = function handle type @(x,y) f(x,y)
%       m,n,sample_num       = just scalars
%       Grid                 = [11 .......         15]           (m,n)    
%                              [9 .......           6]
%                              [12  .......        14]
%                              [10 ......11        13]
%       Sample_grid          = [NaN .......         15]           (m,n)    
%                              [NaN .......        NaN]
%                              [12  .......        NaN]
%                              [NaN ......11       NaN]


% Create meshgrid
yy=1:m; xx=1:n;
[x,y]=meshgrid(xx,yy);

% Apply function
Grid=arrayfun(function_handle,x,y);

% Sample from grid
Rand_col=randi(n,sample_num);
Rand_row=randi(m,sample_num);
Rand_ind=(unique([Rand_row;Rand_col]','rows'))';

% Create Matrix with values at sample points
Sample_grid=NaN(m,n);
for k=1:size(Rand_ind,2)
    Sample_grid(Rand_ind(1,k),Rand_ind(2,k))=Grid(Rand_ind(1,k),Rand_ind(2,k));
end




end