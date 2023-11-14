function [Simulation]=Draw_from_TPspace_slow_robust(ax,bx,ay,by, nx,ny,explained_var,sigma_x,sigma_y)
% This function simulates drawing from tensor product spaces
% The output is a drawing from a 2d random field; its input consists of the
% output dimensions, the truncation variance and the covariance functions
% For this, do the following:
% 1. Definitions and imports
% 1.b Define the kernels
% 2. Construct a decomposition of covariance operator
% 3. Draw some random functions by KLE
% 4. Plot them

% 1. Definitions and imports
x_vector=linspace(ax,bx,nx);
y_vector=linspace(ay,by,ny);

% 2. Construct a decomposition of covariance operator
C_x=zeros(nx,nx);
for k=1:nx
    for l=1:nx
        C_x(k,l)=sigma_x(x_vector(k),x_vector(l));
    end
end

[U_x,S_x,V_x]=svd(C_x);
e_x_mat=U_x;

C_y=zeros(ny,ny);
for k=1:ny
    for l=1:ny
        C_y(k,l)=sigma_y(y_vector(k),y_vector(l));
    end
end
[U_y,S_y,V_y]=svd(C_y);
e_y_mat=U_y;

LAMBDA=zeros(nx*ny,1);  %  eigenvalues,
Ind=zeros(nx*ny,2);     %  i, j index

for i=1:nx
    for j=1:ny
        LAMBDA((i-1)*nx+j,1)=S_x(i,i)*S_y(j,j);
        Ind((i-1)*nx+j,:)=[i,j];
    end
end

[LAMBDA_sort , index]=sortrows(LAMBDA,[-1]);   % Sort descending eigenvalues
Ind_sort=Ind(index,:);                    % Remember sorting for eigenfunctions 

% 3. Draw some random functions by KLE
q=normrnd(0,1,[nx*ny,1]);
Acc=zeros(nx,ny);

tot_var=sum(LAMBDA_sort);
acc_lambda=cumsum(LAMBDA_sort);
trunc_ind=find(-acc_lambda/tot_var+explained_var<10*eps,1);

for k=1:trunc_ind
        delta=q(k)*sqrt(LAMBDA_sort(k))*e_y_mat(:,Ind_sort(k,1))*e_x_mat(:,Ind_sort(k,2))';  % x,y change because of illustration
        Acc=Acc+ delta;
end

Simulation=Acc;
end