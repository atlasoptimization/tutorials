% Simulate a random field with a certain covariance function. Just for
% purposes of fast and uncomplicated illustrations

% 1. Definitions and imports
d=5;
n=30;
x=1:n; y=1:n;
n_p=n^2;
[xx,yy]=meshgrid(x,y);

% 2. Creation of covariance matrix
 cov_fun=@(x,y) exp(-(norm(x-y)./d));      % exponential covariance
% cov_fun=@(x,y) exp(-(norm(x-y)./d).^2);  % squared exponential covariance

xx=xx(:);yy=yy(:);
coord=[xx';yy'];
Sigma_rf=zeros(n^2,n^2);

for i=1:n_p
    for j=1:n_p
        Sigma_rf(i,j)=cov_fun(coord(:,i),coord(:,j));
    end
end
 
% 3. Sampling and plotting
RF=mvnrnd(zeros(n_p,1),Sigma_rf);
RF=reshape(RF,n,n);

figure(1)
subplot(1,2,1)
imagesc(RF);
title('Realization RF exp-cov')
xlabel('X-Axis')
ylabel('Y-Axis')
subplot(1,2,2)
imagesc(Sigma_rf)
title('Covariance matrix exp-cov')
xlabel('Pnr')
ylabel('Pnr')















