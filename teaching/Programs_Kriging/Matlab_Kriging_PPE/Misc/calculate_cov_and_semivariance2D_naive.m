function [ lagmatrix, covariancematrix, semivariancematrix ]=...
    calculate_cov_and_semivariance2D_naive(xvalues,yvalues,processvalues,lagmaxvector)

% 1. Calculate needed sizes and initialize variables. Only allow lagmax to
% be a positive integer.
lagmax_x=round(lagmax(1));
lagmax_y=round(lagmax(2));

covariancematrix=zeros(lagmax_y+1,lagmax_x+1);
semivariancematrix=zeros(lagmax_y+1,lagmax_x+1);


numberofsteps_x=size(xvalues);
numberofsteps_x=numberofsteps_x(1,2);
numberofsteps_y=size(yvalues);
numberofsteps_y=numberofsteps_y(1,1);

xmax=max(xvalues);
xmin=min(xvalues);

ymax=max(yvalues);
ymin=min(yvalues);

stepsfordelta_x=numberofsteps_x/(xmax-xmin);
stepsfordelta_y=numberofsteps_y/(ymax-ymin);


% 2. Shift needed vectors

delta_x_shiftvector=linspace(0,lagmax_x,lagmax_x+1);
x_step_shiftvector=round(stepsfordelta_x*delta_x_shiftvector);
lagvector_x=x_step_shiftvector;

x_lagvector=lagvector_x*1/(stepsfordelta_x);

delta_y_shiftvector=linspace(0,lagmax_y,lagmax_y+1);
y_step_shiftvector=round(stepsfordelta_y*delta_y_shiftvector);
lagvector_y=y_step_shiftvector;

y_lagvector=lagvector_y*1/(stepsfordelta_y);

lagmatrix=[x_lagvector;y_lagvector];

%shiftedprocessvectorcell=cell(1,lagmax+1);

for i=1:lagmax+1
    lag=lagvector(1,i);
    shifted_processvector=circshift(processvaluevector,[0,-lag]);
    reduced_processvector=processvaluevector;
    if lag==0
        % no need for eliminating elements of the vector
    else
        shifted_processvector(:,(numberofsteps -lag)+1 : numberofsteps)=[];
        reduced_processvector(:,(numberofsteps -lag)+1 : numberofsteps)=[];
    end
    %shiftedprocessvectorcell(1,i)={shiftedprocessvector};
    mean_shifted_processvector=mean(shifted_processvector);
    mean_reduced_processvector=mean(reduced_processvector);
    
    shifted_processvector_centralized=shifted_processvector-mean_shifted_processvector;
    reduced_processvector_centralized=reduced_processvector-mean_reduced_processvector;
    
    covariance=(1/(length(reduced_processvector)-1))*...
        (sum(shifted_processvector_centralized.*reduced_processvector_centralized));
    covariancematrix(1,i)=covariance;
    
    semivariance=1/(length(reduced_processvector))*...
        (sum((shifted_processvector_centralized-reduced_processvector_centralized).^2));
    semivariancematrix(1,i)=semivariance;
    
end





end