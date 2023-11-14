% Processes with dice (I)
num_samp=30;  % Number of processvalues

% Simulate dicerolls and generate processvalues
dicerolls=randi(6,[1,num_samp+1]); % One value more for circshift
processvalues=0.5*dicerolls+0.5*circshift(dicerolls,[0 -1]);
processvalues=processvalues(1,1:num_samp); % Cut that last value

% Generate time_axis and a sequence of time differences
timevector=1:num_samp;
lagmax=floor(num_samp/2);

% Calculate semivariances and the variance
sigma_XX=std(processvalues)^2;
[lagvector, covariancevector, semivariancevector]=...
    calculate_cov_and_semivariance1D(timevector,processvalues,lagmax);

% calculate covariance with formula from slide
covariance_by_semivariance=sigma_XX*ones(1,lagmax+1)-semivariancevector;

% Plot semivariance and covariance
figure(1); hold on
plot(lagvector, semivariancevector)
plot(lagvector, covariance_by_semivariance);





