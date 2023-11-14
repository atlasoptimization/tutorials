% Simulate a random field with a certain covariance function. Just for
% purposes of fast and uncomplicated illustrations

% 1. Definitions and imports
d=30;
n=300;
x=linspace(1,n,n); 

 cov_fun=@(x,y) exp(-(norm(x-y)./d));      % exponential covariance 
% cov_fun=@(x,y) exp(-(norm(x-y)./d).^2);  % squared exponential covariance
cov_fun=@(x,y) sin(0.1*cos((norm(x-y))^2)); 

% 2. Creation of covariance matrix
Sigma_sp=zeros(n,n);

for i=1:n
    for j=1:n
        Sigma_sp(i,j)=cov_fun(x(i),x(j));
    end
end
 
% 3. Simulate and plot
SP=mvnrnd(zeros(n,1),Sigma_sp);

figure(1)
subplot(1,2,1)
plot(x,SP);
title('Realization SP exp cov')
xlabel('X Axis')
ylabel('F-values')
subplot(1,2,2)
imagesc(Sigma_sp)
title('Covariance matrix exp cov')
xlabel('Pnr')
ylabel('Pnr')

% 4. Spectral analysis 
[U,S,V]=svd(Sigma_sp);
figure(2)
subplot(1,2,1)
surf(U(:,1:25),'linestyle','none')
title('Eigenvectors of covariance matrix')
subplot(1,2,2)
plot(diag(S(1:100,1:100)))
xlabel('Eigenvalue nr')
ylabel('Eigenvalue magnitude')
title('Spectrum of covariance matrix')
