% The goal of this script is to illustrate the Kriging procedure for some
% illustrative examples showcasing power and simplicity. We will perform
% estimation of 1 D processes, trajectories, and vector fields.


% Task 3 : Vector field interpolation via Simple Kriging ------------------

% 1. Definitions and imports

clear all, close all

% i) Dimensions
n=30;
n_sample=5;

% ii) Indexset
t=linspace(0,1,n);
[ss,tt]=meshgrid(t,t);

sample_index=round(linspace(1,n,n_sample));
[ss_i,tt_i]=meshgrid(sample_index);


t_sample=t(tt_i);
s_sample=t(ss_i);


% 2. Simulate observations 

% i) Covariance function
d=0.3;
cov_fun=@(t1,t2) exp(-(norm(t1-t2)/d)^2);

% ii) Covariance matrix
n_tot=n^2;

C_full=zeros(n_tot,n_tot);
for k=1:n_tot
    for l=1:n_tot
        C_full(k,l)=cov_fun([ss(k);tt(k)],[ss(l);tt(l)]);
    end
end

% iii) Simulate
x=mvnrnd(zeros(n^2,1),C_full);
y=mvnrnd(zeros(n^2,1),C_full);
x_reshape=reshape(x,[n,n]);
y_reshape=reshape(y,[n,n]);

x_sample=x_reshape(sub2ind([n,n],tt_i,ss_i));
y_sample=y_reshape(sub2ind([n,n],tt_i,ss_i));
x_sample_vec=x_sample(:);
y_sample_vec=y_sample(:);


% -------------------------------------------------------------------------
% 3. Estimate at unobserved locations





% -------------------------------------------------------------------------

% 4. Plots and illustrations


figure(1)
% subplot(1,3,1)
% quiver(ss,tt,x_reshape,y_reshape,'color',[0 0 0])
% hold on
% xlabel('x coordinate')
% ylabel('y coordinate')
% title('Vector field')
% set(gcf,'color','w');

subplot(1,3,2)
quiver(s_sample,t_sample,x_sample,y_sample,'color',[0 0 0])
hold on
xlabel('x coordinate')
ylabel('y coordinate')
title('Interpolation task')
set(gcf,'color','w');


% subplot(1,3,3)
% quiver(ss,tt,x_est_reshape,y_est_reshape,'color',[0 0 0])
% hold on
% xlabel('x coordinate')
% ylabel('y coordinate')
% title('Estimation')
% set(gcf,'color','w');


















% % Solution
% 
% n_s_tot=n_sample^2;
% 
% C_ij=zeros(n_s_tot,n_s_tot);
% c_t=zeros(n_tot,n_s_tot);
% 
% for k=1:n_s_tot
%     for l=1:n_s_tot
%         C_ij(k,l)=cov_fun([s_sample(k);t_sample(k)],[s_sample(l);t_sample(l)]);
%     end
% end
% 
% for k=1:n_tot
%     for l=1:n_s_tot
%         c_t(k,l)=cov_fun([ss(k);tt(k)],[s_sample(l);t_sample(l)]);
%     end
% end
% 
% x_est=c_t*pinv(C_ij)*x_sample_vec;
% y_est=c_t*pinv(C_ij)*y_sample_vec;
% x_est_reshape=reshape(x_est,[n,n]);
% y_est_reshape=reshape(y_est,[n,n]);
% 
% 
% 
% 













