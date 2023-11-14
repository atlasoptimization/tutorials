% Create some positive definite functions via map into sequence space and
% sample from the corresponding rkhs
% For this do the following:
% 1. Imports and definitions
% 2. Create kernel
% 3. Sample from rkhs
% 4. Plot results

% 1. Imports and definitions
n=50;
t=linspace(0,1,n);
sum_ind=10;

S=repmat(t',[1,n]);
T=repmat(t,[n,1]);


% 2. Create kernel
% i) Create sequence for series representation of kernel
% The following things can be manipulated:
%     a) any type of function
%     b) any sequence of functions / arbitrary manipulations to get vector
%     entries
%     c) any sequence of square integrable numbers as coefficients

scale_factors=zeros(sum_ind,1);
%phi_base=@(s) sawtooth(s)*(s<=5)+cos(s)+(s<=3);
phi_base=@(s) sawtooth(s);
%phi_base=@(s) sawtooth(s)*tanh(s);

phi=cell(sum_ind,1);
for k=1:sum_ind
    scale_factors(k)=k^(-0.2);
    phi{k}=@(s) phi_base(s*k)*scale_factors(k);
end

phi_vec=@(s) cell2mat(cellfun(@(dummy) dummy(s),phi,'UniformOutput',0));


sigma=@(s,t) sum(phi_vec(s).*phi_vec(t));


% 3. Sample from rkhs
mu=zeros(n,1);

% Build Covariance Matrix K_ij
K_ij=zeros(n,n);
for i=1:n
    for j=1:n
        K_ij(i,j)=sigma(t(i),t(j));
    end
end

X=mvnrnd(mu,K_ij);

% 4. Plot results
ftsz=35;

figure(1)
subplot(1,2,1)
plot(t,X)
xlabel('Time','interpreter','latex')
ylabel('Function value','interpreter','latex')
title('Realization of SP','interpreter','latex')
set(gca,'fontsize',ftsz)
set(gca,'YTick',[])
set(gca,'XTick',[])
box on
subplot(1,2,2)
imagesc(K_ij)
xlabel('Point nr','interpreter','latex')
ylabel('Point nr','interpreter','latex')
title('Covariance matrix','interpreter','latex')
set(gca,'fontsize',ftsz)
set(gca,'YTick',[])
set(gca,'XTick',[])
box on
set(gcf,'color',[1,1,1]);

% Now 2D
ax=0; bx=1;  % intervall borders, number of points along x
ay=0; by=1;  % intervall borders, number of points along y
nx=n;
ny=n;
explained_var=1;

[Simulation_1]=Draw_from_TPspace_slow_robust(ax,bx,ay,by, nx,ny,explained_var,sigma,sigma);

figure(2)
surf(Simulation_1,'linestyle','none')
xlabel('x coordinate','interpreter','latex')
ylabel('y coordinate','interpreter','latex')
title('Realization of RF','interpreter','latex')
set(gca,'fontsize',ftsz)
set(gca,'YTick',[])
set(gca,'XTick',[])
set(gca,'ZTick',[])
box on
set(gcf,'color',[1,1,1]);
figure(3)
imagesc(kron(K_ij,K_ij));
xlabel('Point nr','interpreter','latex')
ylabel('Point nr','interpreter','latex')
title('Covariance matrix','interpreter','latex')
set(gca,'fontsize',ftsz)
set(gca,'YTick',[])
set(gca,'XTick',[])
box on
set(gcf,'color',[1,1,1]);
