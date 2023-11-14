
% Example Skript
m=40; n=40; num_samp=100;
variogram_type='squared_exponential';

% Some function
 function_handle= @(x,y) sin(x*y/100);
 %function_handle= @(x,y) sin((x/(abs(sqrt(y))+1)));


% 1: Make dataset and map data into grid
[Sample_grid, Grid]=Grid_from_function(function_handle,m,n, num_samp);
[Data]=From_grid_to_list(Sample_grid);
Field_values_init=NaN(size(Grid));

[ Field_values, Grid_parameters, coord_to_index, Data_to_estimate]...
    =From_list_to_grid(Data,Field_values_init);

% Illu 1
figure(1);
imagesc(Grid)
title('Underlying function')
wait_var=input('Continue (y/n)? ','s');

% 2: Make experimental semivariogram and fit a model to it
Exp_sem=Make_experimental_semivariogram(Data);
semivar_fun=Fit_model_to_data(Exp_sem,variogram_type);

% Illu 2
figure(1);
plot(Exp_sem(1,:),Exp_sem(2,:));
title('Experimental Semivariogram')
wait_var=input('Continue (y/n)? ','s');
hold on
plot(Exp_sem(1,:),semivar_fun(Exp_sem(1,:)));
title('Experimental Semivariogram + Model')
hold off
wait_var=input('Continue (y/n)? ','s');

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
[Interpolated_grid_vars]=(Write_list_in_grid(Field_values,coord_to_index,Data_estimated,4))';

% Show grid, sampled grid and interpolated grid
min_grid=min(min(Grid));
max_grid=max(max(Grid));

figure(1);
subplot(2,2,1)
imagesc(Grid) 
caxis([min_grid max_grid]);
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
caxis([min_grid max_grid]);
title('Error variance')