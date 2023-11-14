% Processes with dice (II)
num_samp=2000;  % Number of processvalues
alpha=0.9;

% Simulate dicerolls and generate processvalues
dicerolls=randi(6,[1,num_samp+1]); % One value more for circshift

processvalues=zeros(1,num_samp);
processvalues(1,1)=1.75;

for i=2:num_samp
processvalues(1,i)=alpha*processvalues(1,i-1)+(1-alpha)*dicerolls(1,i);
end


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

