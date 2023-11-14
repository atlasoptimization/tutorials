function[ semivariance_estimation]=gamma_spherical(r,c,h)
% This function takes a positive lag h and uses it to estimate the
% semivariance in the lag h by using the spherical variogram model with
% parameters a= range and c=sill. The Result is just a number.

semivariance_estimation=zeros(1,length(h(:)));
for k=1:length(h(:))
    
if  h(k) < r && h(k) > 0
semivariance_estimation(k)=c*((3*h(k)./(2*r))-1/2*(h(k)./r).^3);

elseif h(k)==0
    semivariance_estimation(k)=0;
elseif h(k) >= r
   semivariance_estimation(k)=c; 
end
end
semivariance_estimation=reshape(semivariance_estimation,[size(h)]);


end