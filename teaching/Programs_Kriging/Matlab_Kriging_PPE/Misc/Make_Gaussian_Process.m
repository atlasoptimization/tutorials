% Make gaussian process with covariance function and mean specified below

d=100;
t=linspace(0,1,d);

mu=zeros(1,d);
sigma=zeros(d,d);
sigma_zero=1;
range=0.5;


%covariance_func=@(x,y) sigma_zero.^2*(x==y);
%covariance_func=@(x,y) sigma_zero.^2*min(x,y);
%covariance_func=@(x,y) 1+0.5*((abs(x-y)).^2-(abs(x-y))+1/6);  %periodic sobolev m=1
%covariance_func=@(x,y) sigma_zero.^2*exp(-abs(x-y)/(range));
covariance_func=@(x,y) sigma_zero.^2*exp(-(abs(x-y)/(range)).^2);
% covariance_func=@(x,y) sigma_zero.^2*1./(1+0.01*(x-y).^2);
% covariance_func=@(x,y) sigma_zero.^2*1./(1+0.1*abs(x-y));
%covariance_func=@(x,y) sigma_zero.^2*exp(-abs(x-y)/(range)).*min(x,y)+100;
% covariance_func=@(x,y) sigma_zero.^2*x.*y*1./(1+0.01*(x-y).^2).*cos((x-y)/50);
% covariance_func=@(x,y) sigma_zero.^2*sqrt(x.*y)*1./(1+0.001*(x-y).^2).*cos((x-y)/3);
% covariance_func=@(x,y) sigma_zero.^2*(x.*y).^(1/3).*exp(-abs(x-y)/(range)).*cos((x-y)/8);
% covariance_func=@(t,s) sigma_zero.^2*(abs(t)+abs(s)-abs(t-s)).*exp(-abs(t-s)/range).*exp(-((t-s)/range).^2).*cos((t-s)/8);  % product of min exp and sq exp kernel

for i=1:d
    for j=1:d
        sigma(i,j)=covariance_func(t(i),t(j));
    end
end

process=mvnrnd(mu,sigma);

% Now calculate Covariances and plot them versus the lag
d_tenth=floor(d/10);
shifted_processes=zeros(d,d_tenth);

for k=1:d_tenth
    process_shifted=circshift(process,[0,-(k-1)]);
    shifted_processes(:,k)=process_shifted(:);
end
% Lets be lazy here und just circshift it. We should care about deleting
% the values at the end but we just ignore that error here

corr_mat=corrcoef(shifted_processes);
typical_decay=corr_mat(1,:);

distance=1:d_tenth;
% % Plot process
% figure(1)
% plot(process)
% 
% figure(2)
% plot(distance,typical_decay);
% 
% % Also plot the original covariance function from above
% figure(3)
% cov_values=covariance_func(distance);
% plot(distance,cov_values);

hold on
figure(1)
scatter(1:d,process,40)
axis off
set(gcf,'Color','white')
figure(2)
imagesc(sigma)
axis off
axis equal
set(gcf,'Color','white')