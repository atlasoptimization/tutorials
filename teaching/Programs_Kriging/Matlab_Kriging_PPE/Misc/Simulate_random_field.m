% The goal of this script is to simulate a random field. For this, do the
% following.
% 1. Definitions and imports
% 2. Generate covariance matrix
% 3. Simulate the field
% 4. Plots and figures


% 1. Definitions and imports ----------------------------------------------

n=30;
n_tot=n^2;

x=linspace(0,1,n);
y=linspace(0,1,n);
[xx,yy]=meshgrid(x,y);


% 2. Generate covariance matrix -------------------------------------------
range=0.2;
cov_atm=@(d) exp(-(d/range).^1);

C=zeros(n_tot,n_tot);
for i=1:n_tot
    for j=1:n_tot
        C(i,j)=cov_atm(norm([xx(i);yy(i)]-[xx(j);yy(j)]));
    end
end
 

% 3. Simulate the field --------------------------------------------------

Realization_rf=mvnrnd(zeros(n_tot,1),C);
Realization_rf=reshape(Realization_rf,[n,n]);


% 4. Plots and figures ----------------------------------------------------
ftsz=35;
figure(1)
imagesc(Realization_rf)
set(gcf,'color',[1,1,1])
xlabel('x coordinate','interpreter','latex')
ylabel('y coordinate','interpreter','latex')
title('Realization of RF','interpreter','latex')
set(gca,'fontsize',ftsz)
set(gca,'YTick',[])
set(gca,'XTick',[])
box on