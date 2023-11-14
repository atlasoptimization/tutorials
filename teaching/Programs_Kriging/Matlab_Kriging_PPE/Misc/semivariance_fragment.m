function[semivariance_fragment] =semivariance_fragment(x_values,y_values,processvalues,point,lagvector)
% This function grabs a list of points (represented in x_values, y_values,
% processvalues ) and a special point. For all the distances defined in the
% vector lagvector, it calculates the squared processvalue-differences
% between the special point and all entries in the list in the specified
% distance. The result is an array called "semivariance_fragment"
% of length = length(lagvector). In each of its length(lagvetor) columns it
% contains the number of observations in the specific distance and an
% estimator for the semivariance in this distance.

% Rename for clarity
x_point=point(1);
y_point=point(2);
z_point=point(3);

lag_number=length(lagvector);

% Initialize
semivariance_fragment_temporary=zeros(2,lag_number);

% Calculate tolerance = value defining, how far the the coordinate
% difference between two points may stray from the designated difference
% noted in lagvector without this leading to them not being considered as
% in distance d to each other.
% For this take the mean of the difference of adjacent lags.

shifted_lagvector=circshift(lagvector,[0,-1]);
tolerance=0.5*mean(abs(lagvector(1,1:end-1)-shifted_lagvector(1,1:end-1)));

% Calculate distances between list of point and specific point
distance_vector=(sqrt((x_values-x_point).^2+(y_values-y_point).^2));

% Find all the points in the list with specific distance (lagvector(k)) to
% the special point for all k. Then note the number of those points and
% estimate the semivariance.

for k=1:lag_number
    indices=find(abs(distance_vector-lagvector(k))<tolerance);
    number_of_points=length(indices);
    semivariance_fragment_temporary(2,k)=number_of_points+1;
    semivariance_fragment_temporary(1,k)=sum((z_point*ones(number_of_points,1)-processvalues(indices)).^2);
    
end

% Generate output
semivariance_fragment=semivariance_fragment_temporary;

end