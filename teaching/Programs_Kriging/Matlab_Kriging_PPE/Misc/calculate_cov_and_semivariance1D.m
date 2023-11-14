function [timelagvector, covariancevector, semivariancevector]=...
    calculate_cov_and_semivariance1D(timevector,processvaluevector,lagmax)

% 1. Calculate needed sizes and initialize variables. Only allow lagmax to
% be a positive integer.
covariancevector=zeros(1,lagmax+1);
semivariancevector=zeros(1,lagmax+1);

lagmax=round(lagmax);
numberofsteps=size(timevector);
numberofsteps=numberofsteps(1,2);

tmax=max(timevector);
tmin=min(timevector);

stepsfordeltat=numberofsteps/(tmax-tmin);


% 2. Shift needed vectors

deltatshiftvector=linspace(0,lagmax,lagmax+1);
stepshiftvector=round(stepsfordeltat*deltatshiftvector);
lagvector=stepshiftvector;

timelagvector=lagvector*1/(stepsfordeltat);

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
    covariancevector(1,i)=covariance;
    
    semivariance=1/(2*length(reduced_processvector))*...
        (sum((shifted_processvector_centralized-reduced_processvector_centralized).^2));
    semivariancevector(1,i)=semivariance;
    
end



end