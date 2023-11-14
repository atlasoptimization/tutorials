function ProcessMean = PlotWienerProcesses(number_of_processes ... 
                                                , stepcount, sigma, tmax)
%
% This function should generate a realization of the Wiener process 
% number_of_processes times and plot those realizations in a scatterplot.
% Furthermore the mean value of all those realizations for every point is
% calculated and drawn as a line representing the expected value for each
% random variable X_t.
% The realizations of the Wiener Process are stored in
% Wiener_processes_collection.
                                            
                                            
  Wiener_processes_collection=zeros(number_of_processes,stepcount);
                                            
for i= 1:number_of_processes
    [stepcountvector, timevector, processvaluevector] = ...
    create_wiener_process1D(sigma, tmax, stepcount);
    Wiener_processes_collection(i,:)=processvaluevector;
    scatter(timevector,processvaluevector,3,1:stepcount,'filled');
    hold on                                    
                                            
end

red=linspace(0,0,stepcount)';
green=linspace(0.6,0.6,stepcount)';
blue=linspace(0.7,0.7,stepcount)';

% Explanation: Colormap is Matrix with values between 0 and 1.
% Dimensions: (stepcount, 3) ; first dimension: for each point in time one
% entry. Also second dimension contains r,g,b values for every point in time.
% After initialization, unite three seperate vectors into single
% colormapmatrix.

colors_greenblue_spec=[red green blue];
colormap(colors_greenblue_spec);



% Explanation: Label the scatterplot. sprintf command is used because the
% value of the variable sigma should be incorporated into the title-string.

xlabel('Time');
ylabel('Process Value')
titlename=sprintf('Wiener process with mu = 0 and sigma = %d', sigma);
title(titlename)

% Explanation: Calculate for each time step the mean of all values of the
% different realizations of the Wiener process at that time step. Draw
% those mean values into the scatterplot containing all the processvalues.

ProcessMean=mean(Wiener_processes_collection,1);

plot(timevector,ProcessMean,'Color',[0 0.3 0.9],'Linewidth',2);