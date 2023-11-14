% This script is supposed to simulate a stochastic process.
% For this, do the following:
%
% 1. Definitions and imports
% 2. Calculate Covariance matrix
% 3. Draw a sample
% 4. Plots and figures

% 1. Definitions and imports ---------------------------------------------
n=200;
t=linspace(0,1,n);

d=0.3;
%cov_fun=@(s,t) s==t;
%cov_fun=@(s,t) min(s,t);
%cov_fun=@(s,t) exp(-((abs(s-t)./d).^1));
%cov_fun=@(s,t) exp(-((abs(s-t)./d).^2));
%cov_fun=@(s,t) s.*t;
%cov_fun=@(s,t) 1./(1+s.^2+t.^2);
cov_fun=@(s,t) cos(4*pi.*(s-t));
% 2. Calculate Covariance matrix -----------------------------------------
C=zeros(n,n);
for i=1:n
    for j=1:n
        C(i,j)=cov_fun(t(i),t(j));
    end
end

% 3. Draw a sample --------------------------------------------------------
x=mvnrnd(zeros(n,1),C);

% 4. Plots and figures ----------------------------------------------------
figure(1)
plot(t,x)
title('Sample from process')
xlabel('Time axis')
ylabel('Function value')
set(gcf,'color',[1,1,1])





