% The goal of this script is to illustrate the Kriging procedure for some
% illustrative examples showcasing power and simplicity. We will perform
% estimation of 1 D processes, trajectories, and vector fields.


% Task 2 : Trajectory interpolation via Simple Kriging --------------------

% 1. Definitions and imports

clear all, close all

% i) Dimensions
n=100;
n_sample=10;

% ii) Indexset
t=linspace(0,1,n);
sample_index=round(linspace(1,n,n_sample));
t_sample=t(sample_index);


% 2. Simulate observations 

% i) Covariance function
d=0.3;
cov_fun=@(t1,t2) exp(-(norm(t1-t2)/d)^2);

% ii) Covariance matrix
C_full=zeros(n,n);
for k=1:n
    for l=1:n
        C_full(k,l)=cov_fun(t(k),t(l));
    end
end

% iii) Simulate
x=mvnrnd(zeros(n,1),C_full);
y=mvnrnd(zeros(n,1),C_full);
x_sample=x(sample_index)';
y_sample=y(sample_index)';


% -------------------------------------------------------------------------
% 3. Estimate at unobserved locations




% -------------------------------------------------------------------------

% 4. Plots and illustrations

scatter3(t_sample,x_sample,y_sample,[],[0,0,0])
hold on
xlabel('x coordinate')
ylabel('y coordinate')
title('Interpolation task')
set(gcf,'color','w');

% plot3(t,x_est,y_est,"Color",'k')











% 
% %  Solution
% 
% C_ij=zeros(n_sample,n_sample);
% c_t=zeros(n,n_sample);
% 
% for k=1:n_sample
%     for l=1:n_sample
%         C_ij(k,l)=cov_fun(t_sample(k),t_sample(l));
%     end
% end
% 
% for k=1:n
%     for l=1:n_sample
%         c_t(k,l)=cov_fun(t(k),t_sample(l));
%     end
% end
% 
% x_est=c_t*pinv(C_ij)*x_sample;
% y_est=c_t*pinv(C_ij)*y_sample;
% 
% 






















