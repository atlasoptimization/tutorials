% DEM Interpolation


% Load dataset
DTM_file=load('dtm_basedata_coarse_lossy.mat');
DTM_grid=DTM_file.dtm_basedata_coarse_lossy;
[Data]=From_grid_to_list(DTM_grid);
Field_values_init=NaN(size(DTM_grid));

% Also load truth
DTM_true_file=load('dtm_basedata_coarse.mat');
DTM_true=DTM_true_file.dtm_basedata_coarse;

% Apply Kriging
[Interpolated_grid_vals, Interpolated_grid_vars, Grid_parameters]=Kriging_PPE(Data, Field_values_init,'exponential');

% Apply other interpolation algorithms
[Nearest_Neighbor, Linear_Interp, Natural_Neighbor, Cubic]=Make_comparative_interpolation(Data, Field_values_init);

% Show grid, sampled grid and interpolated grid
figure(1);
subplot(2,2,1)
imagesc(DTM_true)
title('Underlying function')
subplot(2,2,2)
dummy=imagesc(DTM_grid);
set(dummy,'alphadata',~isnan(DTM_grid));
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