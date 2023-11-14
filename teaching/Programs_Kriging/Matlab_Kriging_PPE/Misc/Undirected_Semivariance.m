function[Semivariance_undirected, lagvector]=Undirected_Semivariance(x_values,y_values,processvalues,lagmax,lagnumber)
% This function takes as input a list of x coordinates x_values, y
% coordinates y_values and the function values corresponding to the
% coordinates (processvalues). All three vectors have to have the same
% length. Also the biggest lag and the amount of calculations (resolution
% on the lag -axis) need to be given to calculate the undirected
% semivariance.  The result is a (1,lagnumber+1) vector containing the
% semivariance at lagnumber lags from 0 to lagmax

% Format to n,1 vectors
x_values=x_values(:);
y_values=y_values(:);
processvalues=processvalues(:);

% Make lagvector
lagvector=linspace(0,lagmax,floor(lagnumber)+1);

% Initialize
semivariance_accumulator=zeros(2,length(lagvector));

% For each point in the list of points, calculate for that point the
% semivariance estimations in all the distances specified in lagvector.
% This is done with the subfunction semivariance_part.m

for k=1:length(x_values);
point=[x_values(k), y_values(k),processvalues(k)];

semivariance_part=semivariance_fragment(x_values,y_values,processvalues,point,lagvector);
semivariance_accumulator=semivariance_accumulator+semivariance_part;


end

% Use classical formula to estimate the undirected semivariance
Semivariance_undirected=semivariance_accumulator(1,:)./(semivariance_accumulator(2,:)*2);


end