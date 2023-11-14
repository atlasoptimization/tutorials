% Example Skript
m=40; n=40; num_samp=40;
type='squared_exponential';
% Different test datasets / functions

% Edges and anisotropy
 function_handle= @(x,y) sin(x*y/100);
% function_handle= @(x,y) sin((x/(abs(sqrt(y))+1)));


% Make dataset
[Sample_grid, Grid]=Grid_from_function(function_handle,m,n, num_samp);
[Data]=From_grid_to_list(Sample_grid);
Field_values_init=NaN(size(Grid));

% Apply Kriging
[Interpolated_grid_vals, Interpolated_grid_vars, Grid_parameters]=Kriging_PPE(Data, Field_values_init,type);

% Apply other interpolation algorithms
[Nearest_Neighbor, Linear_Interp, Natural_Neighbor, Cubic]=Make_comparative_interpolation(Data, Field_values_init);

% Show grid, sampled grid and interpolated grid
figure(1);
subplot(2,2,1)
imagesc(Grid)
title('Underlying function')
subplot(2,2,2)
dummy=imagesc(Sample_grid);
set(dummy,'alphadata',~isnan(Sample_grid));
title('Sampled Data')
subplot(2,2,3)
imagesc(Interpolated_grid_vals)
title('Kriging estimations')
subplot(2,2,4)
imagesc(Interpolated_grid_vars)
title('Error variance')

figure(2);
subplot(2,2,1)
imagesc(Nearest_Neighbor)
title('Nearest Neighbor')
subplot(2,2,2)
imagesc(Linear_Interp)
title('Linear Interpolation')
subplot(2,2,3)
imagesc(Natural_Neighbor)
title('Natural Neighbor')
subplot(2,2,4)
imagesc(Cubic)
title('Cubic Spline')